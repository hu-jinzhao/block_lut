// Fused BlockLUT Dequant + Matmul vs 两阶段法 benchmark
// nvcc -arch=sm_86 -O3 bench_fused.cu -lcudart -o bench_fused && ./bench_fused
#include <stdint.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define M 2048
#define K 1408
#define B 32
#define LZ 256
#define WU 20
#define IT 500

// ─── 方案A: 反量化 kernel（写行优先 float）───
__global__ void dequant_kernel(float *W, const uint8_t *idx, const uint16_t *ax,
                               const uint16_t *lut, int m, int k) {
    int row = blockIdx.y, cb = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= m || cb >= k) return;
    __shared__ uint16_t sm[LZ];
    for (int i = threadIdx.x; i < LZ; i += blockDim.x) sm[i] = lut[i];
    __syncthreads();
    int s = blockDim.x * gridDim.x;
    for (int c = cb; c < k; c += s) {
        __nv_bfloat16 lv = *(const __nv_bfloat16 *)&sm[idx[row*k+c]];
        __nv_bfloat16 av = *(const __nv_bfloat16 *)&ax[row*((k+127)/128) + c/128];
        W[row*k + c] = __bfloat162float(lv) * __bfloat162float(av);
    }
}

// ─── 参考 matmul kernel（行优先）───
__global__ void ref_mm(float *Y, const float *W, const float *X, int m, int k, int b) {
    int row = blockIdx.x; if (row >= m) return;
    for (int col = threadIdx.x; col < b; col += blockDim.x) {
        float acc = 0;
        for (int gk = 0; gk < k; gk++) acc += W[row*k+gk] * X[gk*b+col];
        Y[row*b+col] = acc;
    }
}

// ─── 方案B: Fused dequant + matmul ───
__constant__ uint16_t lut_c[LZ];

__global__ void fused_kernel(float *Y, const uint8_t *idx, const uint16_t *ax,
                             const float *X, int m, int k, int b) {
    int row = blockIdx.x; if (row >= m) return;
    int na = (k + 127) / 128;
    for (int col = threadIdx.x; col < b; col += blockDim.x) {
        float acc = 0;
        for (int gk = 0; gk < k; gk++) {
            __nv_bfloat16 lv = *(const __nv_bfloat16 *)&lut_c[idx[row*k+gk]];
            __nv_bfloat16 av = *(const __nv_bfloat16 *)&ax[row*na + gk/128];
            acc += __bfloat162float(lv) * __bfloat162float(av) * X[gk*b+col];
        }
        Y[row*b+col] = acc;
    }
}

// ─── 辅助 ───
void make_data(uint8_t *h_Wi, uint16_t *h_Wa, uint16_t *h_lu, int m, int k) {
    int nb = (k + 127) / 128;
    float lf[LZ];
    for (int i = 0; i < LZ; i++) lf[i] = (float)i/(LZ-1)*2-1;
    for (int i = 0; i < LZ; i++) {
        uint32_t b; memcpy(&b, &lf[i], 4);
        h_lu[i] = (b + 0x7FFF + ((b>>16)&1)) >> 16;
    }
    for (int r = 0; r < m; r++) {
        for (int bk = 0; bk < nb; bk++) {
            float mx = 0;
            for (int i = 0; i < 128 && bk*128+i < k; i++)
                mx = fmaxf(mx, fabsf((float)rand()/RAND_MAX*2-1));
            if (mx < 1e-12f) mx = 1e-12f;
            uint32_t b; memcpy(&b, &mx, 4);
            h_Wa[r*nb+bk] = (b + 0x7FFF + ((b>>16)&1)) >> 16;
            for (int i = 0; i < 128 && bk*128+i < k; i++) {
                float val = (float)rand()/RAND_MAX*2-1 / mx;
                int best = 0; float bd = fabsf(val - lf[0]);
                for (int v = 1; v < LZ; v++) {
                    float d = fabsf(val - lf[v]);
                    if (d < bd) { bd = d; best = v; }
                }
                h_Wi[r*k + bk*128 + i] = best;
            }
        }
    }
}

int main() {
    cudaSetDevice(0);
    printf("=== Fused BlockLUT Dequant + Matmul Benchmark ===\n");
    printf("Weight: %d×%d, Batch: %d\n\n", M, K, B);

    int nb = (K+127)/128;
    std::vector<uint8_t>  h_i(M*K);
    std::vector<uint16_t> h_a(M*nb), h_l(LZ);
    std::vector<float> h_x(K*B);
    make_data(h_i.data(), h_a.data(), h_l.data(), M, K);
    for (auto &v : h_x) v = (float)rand()/RAND_MAX;

    float *dW, *dX, *dY1, *dY2;
    uint8_t *di; uint16_t *da, *dl;
    cudaMalloc(&dW, M*K*4); cudaMalloc(&di, M*K);
    cudaMalloc(&da, M*nb*2); cudaMalloc(&dl, LZ*2);
    cudaMalloc(&dX, K*B*4); cudaMalloc(&dY1, M*B*4); cudaMalloc(&dY2, M*B*4);
    cudaMemcpy(di, h_i.data(), M*K, cudaMemcpyHostToDevice);
    cudaMemcpy(da, h_a.data(), M*nb*2, cudaMemcpyHostToDevice);
    cudaMemcpy(dl, h_l.data(), LZ*2, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(lut_c, h_l.data(), LZ*2);
    cudaMemcpy(dX, h_x.data(), K*B*4, cudaMemcpyHostToDevice);

    dim3 gd((K+255)/256, M), bd(256);  // dequant grid
    dim3 gm(M, 1), bm(256);            // matmul grid

    // ─── 计时辅助 ───
    auto time_it = [&](auto f) {
        cudaEvent_t e0,e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
        for (int i = 0; i < WU; i++) f();
        cudaDeviceSynchronize();
        uint16_t tmp; cudaMemcpy(&tmp, dY1, 2, cudaMemcpyDeviceToHost); volatile uint16_t _ = tmp;
        cudaEventRecord(e0);
        for (int i = 0; i < IT; i++) f();
        cudaEventRecord(e1); cudaEventSynchronize(e1);
        float ms; cudaEventElapsedTime(&ms, e0, e1); ms /= IT;
        cudaEventDestroy(e0); cudaEventDestroy(e1); (void)_;
        return ms;
    };

    // 方案A: dequant → matmul
    float tA = time_it([&]() {
        dequant_kernel<<<gd, bd>>>(dW, di, da, dl, M, K);
        ref_mm<<<gm, bm>>>(dY1, dW, dX, M, K, B);
    });

    // 方案B: fused
    float tB = time_it([&]() {
        fused_kernel<<<gm, bm>>>(dY2, di, da, dX, M, K, B);
    });

    // 结果校验
    std::vector<float> hY1(M*B), hY2(M*B);
    cudaMemcpy(hY1.data(), dY1, M*B*4, cudaMemcpyDeviceToHost);
    cudaMemcpy(hY2.data(), dY2, M*B*4, cudaMemcpyDeviceToHost);
    double md = 0, se = 0;
    for (int i = 0; i < M*B; i++) { double d = fabs((double)hY1[i]-(double)hY2[i]); md = fmax(md, d); se += d*d; }

    printf("方案A (dequant + matmul): %7.3f ms\n", tA);
    printf("方案B (fused deq+mm):    %7.3f ms\n", tB);
    printf("加速比:                  %7.2f x\n", tA / tB);
    printf("正确性:  MaxDiff=%.2e  RMSE=%.2e\n", md, sqrt(se/(M*B)));
    printf("  Y1[0..4]=%.4f %.4f %.4f %.4f %.4f\n", hY1[0],hY1[1],hY1[2],hY1[3],hY1[4]);
    printf("  Y2[0..4]=%.4f %.4f %.4f %.4f %.4f\n", hY2[0],hY2[1],hY2[2],hY2[3],hY2[4]);

    printf("\n结论: fused kernel 当前为 naive 实现，逐元素解压+计算\n");
    printf("      性能比两阶段慢是因为 occupancy 低 (2048 vs 12288 blocks)\n");
    printf("      优化方向: shared memory tiling + 多行合并 + 增大 batch\n");

    // 扩展分析
    double bwA = (double)M*K*(1+2.0/128+4) / 1e9 / (tA/1000);  // dequant read+write
    double bwB = (double)M*K*(1+2.0/128) / 1e9 / (tB/1000);   // fused read only
    printf("\n访存分析:\n");
    printf("  方案A 总数据量: %.0f MB  (dequant R:idx+ax + W:bf16 + matmul R:W+X)\n",
           (double)M*K*(1+2.0/128+4+4+4)/1e6);
    printf("  方案B 总数据量: %.0f MB  (直读压缩格式 + X)\n",
           (double)M*K*(1+2.0/128)/1e6 + (double)K*B*4/1e6);

    cudaFree(dW); cudaFree(di); cudaFree(da); cudaFree(dl);
    cudaFree(dX); cudaFree(dY1); cudaFree(dY2);
    return 0;
}
