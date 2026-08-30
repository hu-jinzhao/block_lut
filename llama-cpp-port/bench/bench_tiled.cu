// BlockLUT Fused Dequant+Matmul: Naive vs Tiled
// nvcc -arch=sm_86 -O3 bench_tiled.cu -lcudart -o bench_tiled && ./bench_tiled
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
#define WU 20
#define IT 500

__constant__ uint16_t lut_c[LZ];

// ─── V1: 反量化 → matmul（基准） ───
__global__ void deq_k(float *W, const uint8_t *di, const uint16_t *da, const uint16_t *dl, int m, int k) {
    int r = blockIdx.y, cb = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= m || cb >= k) return;
    __shared__ uint16_t sm[LZ];
    for (int i = threadIdx.x; i < LZ; i += blockDim.x) sm[i] = dl[i];
    __syncthreads();
    int s = blockDim.x * gridDim.x;
    for (int c = cb; c < k; c += s) {
        float w = __bfloat162float(*(const __nv_bfloat16*)&sm[di[r*k+c]])
                * __bfloat162float(*(const __nv_bfloat16*)&da[r*((k+127)/128)+c/128]);
        W[r*k+c] = w;
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

// ─── V2: Fused naive ───
__global__ void fused_naive(float *Y, const uint8_t *di, const uint16_t *da, const float *X, int m, int k, int b) {
    int r = blockIdx.x; if (r >= m) return;
    int na = (k+127)/128;
    for (int c = threadIdx.x; c < b; c += blockDim.x) {
        float a = 0;
        for (int i = 0; i < k; i++)
            a += __bfloat162float(*(const __nv_bfloat16*)&lut_c[di[r*k+i]])
               * __bfloat162float(*(const __nv_bfloat16*)&da[r*na+i/128])
               * X[i*b+c];
        Y[r*b+c] = a;
    }
}

// ─── V3: Fused tiled (smem caching of X) ───
// Block: (TILE_B, TILE_M) = (32, 8) threads = 256
// Grid:  (B/TILE_B, M/TILE_M) = (1, 256)
// smem:  X_sm[TILE_K][TILE_B] = 128×32×4 = 16KB
__global__ void fused_tiled(float *Y, const uint8_t *di, const uint16_t *da, const float *X, int m, int k, int b) {
    const int BM = 8, BK = 128, BN = 32;
    __shared__ float xsm[BK][BN];

    int cr = blockIdx.y * BM;  // 当前 block 起始行
    int cc = blockIdx.x * BN + threadIdx.x;  // 当前 thread 列
    int tx = threadIdx.x, ty = threadIdx.y;

    float acc[BM] = {0};
    int na = (k+127)/128;

    for (int kt = 0; kt < k; kt += BK) {
        // 协同加载 X tile → smem
        for (int i = ty; i < BK; i += blockDim.y) {
            int gk = kt + i;
            xsm[i][tx] = (gk < k && cc < b) ? X[gk * b + cc] : 0;
        }
        __syncthreads();

        // 对当前 tile 内的 BM 行，逐行计算
        for (int rr = 0; rr < BM; rr++) {
            int row = cr + rr;
            if (row >= m) continue;
            float sum = 0;
            for (int i = 0; i < BK; i++) {
                int gk = kt + i;
                if (gk >= k) break;
                float w = __bfloat162float(*(const __nv_bfloat16*)&lut_c[di[row*k+gk]])
                        * __bfloat162float(*(const __nv_bfloat16*)&da[row*na+gk/128]);
                sum += w * xsm[i][tx];
            }
            acc[rr] += sum;
        }
        __syncthreads();
    }

    for (int rr = 0; rr < BM; rr++) {
        int row = cr + rr;
        if (row < m && cc < b) Y[row * b + cc] = acc[rr];
    }
}

// ─── 测试数据生成 ───
void make_data(uint8_t *wi, uint16_t *wa, uint16_t *hl, int m, int k) {
    int nb = (k+127)/128; float lf[LZ];
    for (int i = 0; i < LZ; i++) lf[i] = (float)i/(LZ-1)*2-1;
    for (int i = 0; i < LZ; i++) { uint32_t b; memcpy(&b, &lf[i], 4); hl[i] = (b+0x7FFF+((b>>16)&1))>>16; }
    for (int r = 0; r < m; r++) for (int bk = 0; bk < nb; bk++) {
        float mx = 0;
        for (int i = 0; i < 128 && bk*128+i < k; i++) mx = fmaxf(mx, fabsf((float)rand()/RAND_MAX*2-1));
        mx = fmaxf(mx, 1e-6f);
        uint32_t b; memcpy(&b, &mx, 4); wa[r*nb+bk] = (b+0x7FFF+((b>>16)&1))>>16;
        for (int i = 0; i < 128 && bk*128+i < k; i++) {
            float val = (float)rand()/RAND_MAX*2-1/mx;
            if (val < -1) val = -1; if (val > 1) val = 1;
            int best = 0; float bd = fabsf(val-lf[0]);
            for (int v = 1; v < LZ; v++) { float d = fabsf(val-lf[v]); if (d < bd) { bd = d; best = v; } }
            wi[r*k+bk*128+i] = best;
        }
    }
}

int main() {
    cudaSetDevice(0); printf("=== Fused Dequant+Matmul: Naive vs Tiled ===\nM=%d K=%d B=%d\n\n", M, K, B);

    int nb = (K+127)/128;
    std::vector<uint8_t>  hi(M*K);
    std::vector<uint16_t> ha(M*nb), hl(LZ);
    std::vector<float> hx(K*B);
    make_data(hi.data(), ha.data(), hl.data(), M, K);
    for (auto &v : hx) v = (float)rand()/RAND_MAX;

    float *dW, *dX, *dY0, *dY1, *dY2; uint8_t *di; uint16_t *da, *dl;
    cudaMalloc(&dW, M*K*4); cudaMalloc(&dX, K*B*4);
    cudaMalloc(&dY0, M*B*4); cudaMalloc(&dY1, M*B*4); cudaMalloc(&dY2, M*B*4);
    cudaMalloc(&di, M*K); cudaMalloc(&da, M*nb*2); cudaMalloc(&dl, LZ*2);
    cudaMemcpy(di, hi.data(), M*K, cudaMemcpyHostToDevice);
    cudaMemcpy(da, ha.data(), M*nb*2, cudaMemcpyHostToDevice);
    cudaMemcpy(dl, hl.data(), LZ*2, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(lut_c, hl.data(), LZ*2);
    cudaMemcpy(dX, hx.data(), K*B*4, cudaMemcpyHostToDevice);

    dim3 gd((K+255)/256, M), bd(256);
    dim3 gm(M), bm(256);
    dim3 gf(1, M/8), bf(32, 8);  // fused tiled: (TILE_B, TILE_M)

    auto time_it = [&](auto f) {
        cudaEvent_t e0,e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
        for (int i=0;i<WU;i++) f(); cudaDeviceSynchronize();
        volatile uint16_t _; cudaMemcpy((void*)&_, dY0, 2, cudaMemcpyDeviceToHost);
        cudaEventRecord(e0); for (int i=0;i<IT;i++) f();
        cudaEventRecord(e1); cudaEventSynchronize(e1);
        float ms; cudaEventElapsedTime(&ms, e0, e1); ms/=IT; cudaEventDestroy(e0); cudaEventDestroy(e1); (void)_;
        return ms;
    };

    float t0 = time_it([&](){ deq_k<<<gd,bd>>>(dW,di,da,dl,M,K); mm_k<<<gm,bm>>>(dY0,dW,dX,M,K,B); });
    float t1 = time_it([&](){ fused_naive<<<gm,bm>>>(dY1,di,da,dX,M,K,B); });
    float t2 = time_it([&](){ fused_tiled<<<gf,bf>>>(dY2,di,da,dX,M,K,B); });

    std::vector<float> h0(M*B), h1(M*B), h2(M*B);
    cudaMemcpy(h0.data(), dY0, M*B*4, cudaMemcpyDeviceToHost);
    cudaMemcpy(h1.data(), dY1, M*B*4, cudaMemcpyDeviceToHost);
    cudaMemcpy(h2.data(), dY2, M*B*4, cudaMemcpyDeviceToHost);

    auto verify = [](auto &a, auto &b, const char *name) {
        double md=0, se=0; for (int i=0;i<M*B;i++){double d=fabs((double)a[i]-(double)b[i]);md=fmax(md,d);se+=d*d;}
        printf("  %-16s: MaxDiff=%.2e RMSE=%.2e %s\n", name, md, sqrt(se/(M*B)),
               (md<1e-3?"✅":"⚠️"));
    };

    printf("--- 耗时 ---\n");
    printf("V0 deq+mm   (baseline):  %7.3f ms\n", t0);
    printf("V1 fused naive:          %7.3f ms\n", t1);
    printf("V2 fused tiled (smem):   %7.3f ms\n", t2);
    printf("\n加速比:  V2/V0=%.2fx  V2/V1=%.2fx\n", t0/t2, t1/t2);

    printf("\n--- 正确性 ---\n");
    verify(h0, h1, "V1 vs V0 (baseline)");
    verify(h0, h2, "V2 vs V0 (baseline)");

    printf("\n--- 分析 ---\n");
    printf("V0: deq(%d blk) + mm(%d blk) = %d total blocks\n", gd.x*gd.y, gm.x, gd.x*gd.y+gm.x);
    printf("V1: %d blocks, %d thr/block\n", gm.x, bm.x);
    printf("V2: %d blocks, %d thr/block, smem=%zu bytes\n", gf.x*gf.y, bf.x*bf.y, 128*32*4);

    cudaFree(dW);cudaFree(dX);cudaFree(dY0);cudaFree(dY1);cudaFree(dY2);
    cudaFree(di);cudaFree(da);cudaFree(dl);
    return 0;
}
