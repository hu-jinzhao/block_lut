#include <stdint.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

#define NX  (1408*2048)
#define NT  72
#define NA  (NX*NT)
#define NB  (NX/128)
#define LZ  256
#define WU  20
#define IT  500

// ─── Naive kernel ───
__global__ void kernel_naive(
    uint16_t * __restrict__ o,
    const uint8_t * __restrict__ x,
    const uint16_t * __restrict__ a,
    const uint16_t * __restrict__ l,
    int n)
{
    __shared__ uint16_t sm[LZ];
    for (int i = threadIdx.x; i < LZ; i += blockDim.x) sm[i] = l[i];
    __syncthreads();

    int base = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
    if (base + 7 < n) {
        uint64_t v = *(const uint64_t *)(x + base);
        __nv_bfloat16 r[8];
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            int elem = base + i;
            uint8_t lidx = (v >> (i * 8)) & 0xFF;
            __nv_bfloat16 lv = *(const __nv_bfloat16 *)&sm[lidx];
            __nv_bfloat16 av = *(const __nv_bfloat16 *)&a[elem >> 7];
            r[i] = __float2bfloat16(__bfloat162float(lv) * __bfloat162float(av));
        }
        *(uint64_t*)(o + base)     = *(uint64_t*)&r[0];
        *(uint64_t*)(o + base + 4) = *(uint64_t*)&r[4];
    } else {
        for (int i = base; i < n; ++i) {
            __nv_bfloat16 lv = *(const __nv_bfloat16 *)&sm[x[i]];
            __nv_bfloat16 av = *(const __nv_bfloat16 *)&a[i >> 7];
            *(__nv_bfloat16*)(o + i) = __float2bfloat16(__bfloat162float(lv) * __bfloat162float(av));
        }
    }
}

// ─── __constant__ LUT ───
__constant__ uint16_t lut_const[LZ];

// ─── Optimized kernel ───
__global__ void kernel_opt(
    uint16_t * __restrict__ o,
    const uint8_t * __restrict__ x,
    const uint16_t * __restrict__ a,
    int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = tid * 16; i < n; i += stride * 16) {
        int rem = n - i;
        if (rem > 16) rem = 16;

        #pragma unroll
        for (int j = 0; j < rem; j += 4) {
            if (j + 4 > rem) break;

            uint4 u = *(const uint4 *)(x + i + j);
            int b0 = (i + j + 0) >> 7;
            int b1 = (i + j + 1) >> 7;
            int b2 = (i + j + 2) >> 7;
            int b3 = (i + j + 3) >> 7;

            __nv_bfloat16 l0 = *(const __nv_bfloat16 *)&lut_const[u.x];
            __nv_bfloat16 l1 = *(const __nv_bfloat16 *)&lut_const[u.y];
            __nv_bfloat16 l2 = *(const __nv_bfloat16 *)&lut_const[u.z];
            __nv_bfloat16 l3 = *(const __nv_bfloat16 *)&lut_const[u.w];
            __nv_bfloat16 a0 = *(const __nv_bfloat16 *)&a[b0];
            __nv_bfloat16 a1 = *(const __nv_bfloat16 *)&a[b1];
            __nv_bfloat16 a2 = *(const __nv_bfloat16 *)&a[b2];
            __nv_bfloat16 a3 = *(const __nv_bfloat16 *)&a[b3];

            __nv_bfloat162 p01 = __hmul2(
                __floats2bfloat162_rn(__bfloat162float(l0), __bfloat162float(l1)),
                __floats2bfloat162_rn(__bfloat162float(a0), __bfloat162float(a1)));
            __nv_bfloat162 p23 = __hmul2(
                __floats2bfloat162_rn(__bfloat162float(l2), __bfloat162float(l3)),
                __floats2bfloat162_rn(__bfloat162float(a2), __bfloat162float(a3)));

            *(uint64_t*)(o + i + j + 0) = *(uint64_t*)&p01;
            *(uint64_t*)(o + i + j + 2) = *(uint64_t*)&p23;
        }
        for (int j = (rem / 4) * 4; j < rem; ++j) {
            __nv_bfloat16 lv = *(const __nv_bfloat16 *)&lut_const[x[i + j]];
            __nv_bfloat16 av = *(const __nv_bfloat16 *)&a[(i + j) >> 7];
            *(__nv_bfloat16*)(o + i + j) = __float2bfloat16(__bfloat162float(lv) * __bfloat162float(av));
        }
    }
}

// ─── Benchmark ───
double bench_kernel(const char *name, const void *k, dim3 g, dim3 b,
                    uint16_t *o, const uint8_t *x, const uint16_t *a, const uint16_t *l, int n, bool use_opt)
{
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);

    auto launch = [&]() {
        if (use_opt) kernel_opt<<<g,b>>>(o, x, a, n);
        else         kernel_naive<<<g,b>>>(o, x, a, l, n);
    };

    for (int i = 0; i < WU; i++) launch();
    cudaDeviceSynchronize();

    uint16_t chk;
    cudaMemcpy(&chk, o, 2, cudaMemcpyDeviceToHost);
    volatile uint16_t dummy = chk;

    cudaEventRecord(e0);
    for (int i = 0; i < IT; i++) launch();
    cudaEventRecord(e1);
    cudaEventSynchronize(e1);

    float ms;
    cudaEventElapsedTime(&ms, e0, e1);
    double avg = ms / IT;
    double bw  = (double)n * 2 / 1e9 / (avg / 1000);

    printf("%-18s %8.4f ms  %7.1f GB/s  %5.1f Gelem/s  (%d blk x %d thr)\n",
           name, avg, bw, (double)n/1e9/(avg/1000), g.x, b.x);

    cudaEventDestroy(e0); cudaEventDestroy(e1);
    (void)dummy;
    return avg;
}

int main()
{
    cudaSetDevice(0);

    printf("BlockLUT Dequantize Benchmark — A5000\n");
    printf("Elements total: %d (72 experts pooled)\n\n", NA);

    // Test data
    std::vector<uint8_t>  hi(NA);
    std::vector<uint16_t> ha(NB * NT);
    std::vector<uint16_t> hl(LZ);
    for (auto &v : hi) v = rand() % 256;
    for (auto &v : ha) { float f = (float)rand()/RAND_MAX*0.5f+0.01f; uint32_t b; memcpy(&b,&f,4); v = (b + 0x7FFF + ((b>>16)&1)) >> 16; }
    for (int i = 0; i < LZ; i++) { float f = (float)i/255*2-1; uint32_t b; memcpy(&b,&f,4); hl[i] = (b + 0x7FFF + ((b>>16)&1)) >> 16; }

    uint8_t  *di; cudaMalloc(&di, NA);
    uint16_t *da; cudaMalloc(&da, NB * NT * 2);
    uint16_t *dl; cudaMalloc(&dl, LZ * 2);
    uint16_t *d_out; cudaMalloc(&d_out, NA * 2);
    cudaMemcpy(di, hi.data(), NA, cudaMemcpyHostToDevice);
    cudaMemcpy(da, ha.data(), NB * NT * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dl, hl.data(), LZ * 2, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(lut_const, hl.data(), LZ * 2);

    dim3 b_naive(256), g_naive((NA + 256*8 - 1) / (256*8));
    dim3 b_opt(128),   g_opt((NA + 128*16 - 1) / (128*16));

    double tn = bench_kernel("naive (smem)", nullptr, g_naive, b_naive, d_out, di, da, dl, NA, false);
    double to = bench_kernel("optimized (cc+u4)", nullptr, g_opt, b_opt, d_out, di, da, nullptr, NA, true);

    double bw_peak = 768.0;  // A5000 peak BW
    printf("\n  Speedup:         %.2fx\n", tn / to);
    printf("  Mem BW (naive):  %.0f GB/s (%.0f%% of peak %.0f GB/s)\n",
           (double)NA*2/1e9/(tn/1000), (double)NA*2/1e9/(tn/1000)/bw_peak*100, bw_peak);
    printf("  Mem BW (opt):    %.0f GB/s (%.0f%% of peak)\n",
           (double)NA*2/1e9/(to/1000), (double)NA*2/1e9/(to/1000)/bw_peak*100);

    cudaFree(di); cudaFree(da); cudaFree(dl); cudaFree(d_out);
    return 0;
}
