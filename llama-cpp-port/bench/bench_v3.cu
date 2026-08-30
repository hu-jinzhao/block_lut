// BlockLUT Fused Dequant+Matmul: V3 — 沿 K 维度分块并行
// nvcc -arch=sm_86 -O3 bench_v3.cu -lcudart -o /tmp/bv3 && /tmp/bv3
#include <stdint.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define M 2048
#define K 1408
#define B 32
#define LZ 256
#define BK 128          // K tile size
#define NK ((K+BK-1)/BK) // K tiles count = 11
#define WU 20
#define IT 500

__constant__ uint16_t lut_c[LZ];

// ─── V0: dequant + matmul (baseline) ───
__global__ void deq_k(float *W, const uint8_t *di, const uint16_t *da, const uint16_t *dl, int m, int k) {
    int r = blockIdx.y, cb = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= m || cb >= k) return;
    __shared__ uint16_t sm[LZ];
    for (int i = threadIdx.x; i < LZ; i += blockDim.x) sm[i] = dl[i]; __syncthreads();
    int s = blockDim.x * gridDim.x;
    for (int c = cb; c < k; c += s) {
        W[r*k+c] = __bfloat162float(*(const __nv_bfloat16*)&sm[di[r*k+c]])
                 * __bfloat162float(*(const __nv_bfloat16*)&da[r*((k+127)/128)+c/128]);
    }
}
__global__ void mm_k(float *Y, const float *W, const float *X, int m, int k, int b) {
    int r = blockIdx.x; if (r >= m) return;
    for (int c = threadIdx.x; c < b; c += blockDim.x) {
        float a = 0;
        for (int i = 0; i < k; i++) a += W[r*k+i] * X[i*b+c];
        Y[r*b+c] = a;
    }
}

// ─── V1: fused naive ───
__global__ void fused_v1(float *Y, const uint8_t *di, const uint16_t *da, const float *X, int m, int k, int b) {
    int r = blockIdx.x; if (r >= m) return;
    int na = (k+127)/128;
    for (int c = threadIdx.x; c < b; c += blockDim.x) {
        float a = 0;
        for (int i = 0; i < k; i++)
            a += __bfloat162float(*(const __nv_bfloat16*)&lut_c[di[r*k+i]])
               * __bfloat162float(*(const __nv_bfloat16*)&da[r*na+i/128]) * X[i*b+c];
        Y[r*b+c] = a;
    }
}

// ─── V3: fused + 每块算一个输出元素 ───
// grid: (M, B) = (2048, 32) = 65536 blocks
// 每个 block: 256 个线程分工覆盖所有 K
// 每线程处理 ceil(K/256) ≈ 6 个元素
__global__ void fused_v3_block(float *Y, const uint8_t *di, const uint16_t *da,
                                const float *X, int m, int k, int b) {
    int row = blockIdx.x;
    int col = blockIdx.y;
    if (row >= m || col >= b) return;

    int na = (k + 127) / 128;
    float acc = 0;
    int tid = threadIdx.x;
    int stride = blockDim.x;

    for (int gk = tid; gk < k; gk += stride) {
        acc += __bfloat162float(*(const __nv_bfloat16*)&lut_c[di[row*k+gk]])
             * __bfloat162float(*(const __nv_bfloat16*)&da[row*na+gk/128])
             * X[gk * b + col];
    }

    // warp reduce
    __shared__ float sm[256];
    sm[tid] = acc;
    for (int s = stride/2; s > 0; s >>= 1) {
        __syncthreads();
        if (tid < s) sm[tid] += sm[tid + s];
    }
    if (tid == 0) Y[row * b + col] = sm[0];
}

// ─── 测试数据生成 ───
void make_data(uint8_t *wi, uint16_t *wa, int m, int k) {
    int nb = (k+127)/128; float lf[LZ];
    for (int i = 0; i < LZ; i++) lf[i] = (float)i/(LZ-1)*2-1;
    for (int r = 0; r < m; r++) for (int bk = 0; bk < nb; bk++) {
        float mx = 0;
        for (int i = 0; i < 128 && bk*128+i < k; i++) mx = fmaxf(mx, fabsf((float)rand()/RAND_MAX*2-1));
        mx = fmaxf(mx, 1e-6f);
        uint32_t b; memcpy(&b, &mx, 4); wa[r*nb+bk] = (b+0x7FFF+((b>>16)&1))>>16;
        for (int i = 0; i < 128 && bk*128+i < k; i++) {
            float val = ((float)rand()/RAND_MAX*2-1) / mx;
            val = fmaxf(-1, fminf(1, val));
            int best = 0; float bd = fabsf(val-lf[0]);
            for (int v = 1; v < LZ; v++) { float d = fabsf(val-lf[v]); if (d < bd) { bd = d; best = v; } }
            wi[r*k+bk*128+i] = best;
        }
    }
}

template<typename F>
float time_it(F f) {
    cudaEvent_t e0,e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    for (int i=0;i<WU;i++) f(); cudaDeviceSynchronize();
    volatile uint16_t _; cudaMemcpy((void*)&_, (void*)0, 2, cudaMemcpyDeviceToHost);
    cudaEventRecord(e0); for (int i=0;i<IT;i++) f();
    cudaEventRecord(e1); cudaEventSynchronize(e1);
    float ms; cudaEventElapsedTime(&ms, e0, e1); ms/=IT; cudaEventDestroy(e0); cudaEventDestroy(e1); (void)_;
    return ms;
}

int main() {
    cudaSetDevice(0); printf("=== Fused Deq+MM: K-tile parallelism ===\n");
    printf("M=%d K=%d B=%d BK=%d NK=%d\n\n", M, K, B, BK, NK);

    int nb = (K+127)/128;
    std::vector<uint8_t>  hi(M*K);
    std::vector<uint16_t> ha(M*nb);
    std::vector<uint16_t> hl(LZ);
    for (int i = 0; i < LZ; i++) { float v = (float)i/(LZ-1)*2-1; uint32_t b; memcpy(&b,&v,4); hl[i]=(b+0x7FFF+((b>>16)&1))>>16; }
    std::vector<float> hx(K*B);
    make_data(hi.data(), ha.data(), M, K);
    for (auto &v : hx) v = (float)rand()/RAND_MAX;

    float *dW, *dX, *dY0, *dY1, *dY3;
    uint8_t *di; uint16_t *da, *dl;
    cudaMalloc(&dW, M*K*4); cudaMalloc(&dX, K*B*4);
    cudaMalloc(&dY0, M*B*4); cudaMalloc(&dY1, M*B*4); cudaMalloc(&dY3, M*B*4);
    cudaMalloc(&di, M*K); cudaMalloc(&da, M*nb*2); cudaMalloc(&dl, LZ*2);
    cudaMemcpy(di, hi.data(), M*K, cudaMemcpyHostToDevice);
    cudaMemcpy(da, ha.data(), M*nb*2, cudaMemcpyHostToDevice);
    cudaMemcpy(dl, hl.data(), LZ*2, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(lut_c, hl.data(), LZ*2);
    cudaMemcpy(dX, hx.data(), K*B*4, cudaMemcpyHostToDevice);

    dim3 gd((K+255)/256, M), bd(256);
    dim3 gm(M), bm(256);
    dim3 g3(M, B), b3(256);  // 65536 blocks! 每 block 一个输出元素

    printf("--- 耗时 ---\n");

    float t0 = time_it([&](){
        deq_k<<<gd,bd>>>(dW,di,da,dl,M,K);
        mm_k<<<gm,bm>>>(dY0,dW,dX,M,K,B);
    });
    printf("V0 deq+mm:         %7.3f ms  (blocks=%d)\n", t0, gd.x*gd.y+gm.x);

    float t1 = time_it([&](){
        fused_v1<<<gm,bm>>>(dY1,di,da,dX,M,K,B);
    });
    printf("V1 fused naive:    %7.3f ms  (blocks=%d)\n", t1, gm.x);

    float t3 = time_it([&](){
        fused_v3_block<<<g3,b3>>>(dY3,di,da,dX,M,K,B);
    });
    printf("V3 per-element blk: %7.3f ms  (blocks=%d)\n", t3, g3.x*g3.y);

    // 正确性
    std::vector<float> h0(M*B), h1(M*B), h3(M*B);
    cudaMemcpy(h0.data(), dY0, M*B*4, cudaMemcpyDeviceToHost);
    cudaMemcpy(h1.data(), dY1, M*B*4, cudaMemcpyDeviceToHost);
    cudaMemcpy(h3.data(), dY3, M*B*4, cudaMemcpyDeviceToHost);

    auto chk = [&](auto &a, auto &b, const char *n) {
        double md=0,se=0; for(int i=0;i<M*B;i++){double d=fabs((double)a[i]-(double)b[i]);md=fmax(md,d);se+=d*d;}
        printf("  %-18s MaxDiff=%.2e RMSE=%.2e\n", n, md, sqrt(se/(M*B)));
    };
    printf("\n--- 正确性 ---\n");
    chk(h0, h1, "V1 vs V0");
    chk(h0, h3, "V3 vs V0");
    printf("  V3 speedup vs V0: %.2fx\n", t0/t3);
    printf("  V3 speedup vs V1: %.2fx\n", t1/t3);

    printf("\n--- 分析 ---\n");
    printf("V0: %d blocks, deq %.2f + mm %.2f\n", gd.x*gd.y+gm.x, t0*0.58, t0*0.42);
    printf("V1: %d blocks, %.0f blocks/SM\n", gm.x, gm.x/80.0);
    printf("V3: %d blocks, %.0f blocks/SM, %.0f%% utilization\n",
           g3.x*g3.y, g3.x*g3.y/80.0, g3.x*g3.y/1280.0*100);

    cudaFree(dW);cudaFree(dX);cudaFree(dY0);cudaFree(dY1);cudaFree(dY3);
    cudaFree(di);cudaFree(da);cudaFree(dl);
    return 0;
}
