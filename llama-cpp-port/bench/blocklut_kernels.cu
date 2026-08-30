// BlockLUT Kernel Collection — 供 ncu 分析
// nvcc -arch=sm_86 -O3 blocklut_kernels.cu -lcudart -o blocklut_kernels && ./blocklut_kernels
// ncu --set full -o profile_k1 --kernel-name "fused_naive" ./blocklut_kernels
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
#define WU 10
#define IT 200

__constant__ uint16_t lut_const[LZ];

// K0: dequant (smem LUT)
__global__ void k0_deq(float *W, const uint8_t *d, const uint16_t *a, const uint16_t *l, int m, int k) {
    int r=blockIdx.y,cb=blockIdx.x*blockDim.x+threadIdx.x; if(r>=m||cb>=k)return;
    __shared__ uint16_t s[LZ]; for(int i=threadIdx.x;i<LZ;i+=blockDim.x) s[i]=l[i]; __syncthreads();
    int st=blockDim.x*gridDim.x;
    for(int c=cb;c<k;c+=st) W[r*k+c]=__bfloat162float(*(const __nv_bfloat16*)&s[d[r*k+c]])*__bfloat162float(*(const __nv_bfloat16*)&a[r*((k+127)/128)+c/128]);
}
__global__ void k0_mm(float *Y, const float *W, const float *X, int m, int k, int b) {
    int r=blockIdx.x;if(r>=m)return;
    for(int c=threadIdx.x;c<b;c+=blockDim.x){float s=0;for(int i=0;i<k;i++) s+=W[r*k+i]*X[i*b+c];Y[r*b+c]=s;}
}

// K1: fused naive (const LUT)
__global__ void k1_fused_naive(float *Y, const uint8_t *d, const uint16_t *a, const float *X, int m, int k, int b) {
    int r=blockIdx.x;if(r>=m)return;int na=(k+127)/128;
    for(int c=threadIdx.x;c<b;c+=blockDim.x){float s=0;for(int i=0;i<k;i++)s+=__bfloat162float(*(const __nv_bfloat16*)&lut_const[d[r*k+i]])*__bfloat162float(*(const __nv_bfloat16*)&a[r*na+i/128])*X[i*b+c];Y[r*b+c]=s;}
}

// K2: fused smem LUT
__global__ void k2_smem_lut(float *Y, const uint8_t *d, const uint16_t *a, const uint16_t *l, const float *X, int m, int k, int b) {
    __shared__ uint16_t s[LZ]; for(int i=threadIdx.x;i<LZ;i+=blockDim.x) s[i]=l[i]; __syncthreads();
    int r=blockIdx.x;if(r>=m)return;int na=(k+127)/128;
    for(int c=threadIdx.x;c<b;c+=blockDim.x){float sum=0;for(int i=0;i<k;i++)sum+=__bfloat162float(*(const __nv_bfloat16*)&s[d[r*k+i]])*__bfloat162float(*(const __nv_bfloat16*)&a[r*na+i/128])*X[i*b+c];Y[r*b+c]=sum;}
}

// K3: fused uint4
__global__ void k3_uint4(float *Y, const uint8_t *d, const uint16_t *a, const float *X, int m, int k, int b) {
    int r=blockIdx.x;if(r>=m)return;int na=(k+127)/128;
    for(int c=threadIdx.x;c<b;c+=blockDim.x){float s=0;int i;
        for(i=0;i+4<=k;i+=4){uint4 u=*(const uint4*)(d+r*k+i);
            s+=__bfloat162float(*(const __nv_bfloat16*)&lut_const[u.x])*__bfloat162float(*(const __nv_bfloat16*)&a[r*na+(i+0)/128])*X[(i+0)*b+c];
            s+=__bfloat162float(*(const __nv_bfloat16*)&lut_const[u.y])*__bfloat162float(*(const __nv_bfloat16*)&a[r*na+(i+1)/128])*X[(i+1)*b+c];
            s+=__bfloat162float(*(const __nv_bfloat16*)&lut_const[u.z])*__bfloat162float(*(const __nv_bfloat16*)&a[r*na+(i+2)/128])*X[(i+2)*b+c];
            s+=__bfloat162float(*(const __nv_bfloat16*)&lut_const[u.w])*__bfloat162float(*(const __nv_bfloat16*)&a[r*na+(i+3)/128])*X[(i+3)*b+c];}
        for(;i<k;i++)s+=__bfloat162float(*(const __nv_bfloat16*)&lut_const[d[r*k+i]])*__bfloat162float(*(const __nv_bfloat16*)&a[r*na+i/128])*X[i*b+c];Y[r*b+c]=s;}
}

// K4: per-element
__global__ void k4_per_elem(float *Y, const uint8_t *d, const uint16_t *a, const float *X, int m, int k, int b) {
    int r=blockIdx.x,c=blockIdx.y;if(r>=m||c>=b)return;int na=(k+127)/128;float s=0;
    for(int i=threadIdx.x;i<k;i+=blockDim.x)s+=__bfloat162float(*(const __nv_bfloat16*)&lut_const[d[r*k+i]])*__bfloat162float(*(const __nv_bfloat16*)&a[r*na+i/128])*X[i*b+c];
    __shared__ float sm[256];sm[threadIdx.x]=s;
    for(int st=blockDim.x/2;st>0;st>>=1){__syncthreads();if(threadIdx.x<st)sm[threadIdx.x]+=sm[threadIdx.x+st];}
    if(threadIdx.x==0)Y[r*b+c]=sm[0];
}

void gen(uint8_t *w, uint16_t *a, uint16_t *l, int m, int k) {
    int nb=(k+127)/128;float lf[LZ];for(int i=0;i<LZ;i++)lf[i]=(float)i/(LZ-1)*2-1;
    for(int i=0;i<LZ;i++){uint32_t b;memcpy(&b,&lf[i],4);l[i]=(b+0x7FFF+((b>>16)&1))>>16;}
    for(int r=0;r<m;r++)for(int b=0;b<nb;b++){
        float mx=0;for(int i=0;i<128&&b*128+i<k;i++)mx=fmaxf(mx,fabsf((float)rand()/RAND_MAX*2-1));
        mx=fmaxf(mx,1e-6f);uint32_t x;memcpy(&x,&mx,4);a[r*nb+b]=(x+0x7FFF+((x>>16)&1))>>16;
        for(int i=0;i<128&&b*128+i<k;i++){float v=((float)rand()/RAND_MAX*2-1)/mx;v=fmaxf(-1,fminf(1,v));
            int bi=0;float bd=fabsf(v-lf[0]);for(int j=1;j<LZ;j++){float d=fabsf(v-lf[j]);if(d<bd){bd=d;bi=j;}}w[r*k+b*128+i]=bi;}}
}

int main() {
    cudaSetDevice(0);
    printf("=== BlockLUT Kernels ===\nM=%d K=%d B=%d\n\n", M, K, B);

    int nb=(K+127)/128;
    std::vector<uint8_t> hi(M*K); std::vector<uint16_t> ha(M*nb), hl(LZ); std::vector<float> hx(K*B);
    gen(hi.data(),ha.data(),hl.data(),M,K); for(auto&v:hx)v=(float)rand()/RAND_MAX;

    float *w,*x0,*x1,*x2,*x3,*x4,*y0,*y1,*y2,*y3,*y4; uint8_t *di; uint16_t *da,*dl;
    cudaMalloc(&w,M*K*4);cudaMalloc(&x0,K*B*4);cudaMalloc(&x1,K*B*4);cudaMalloc(&x2,K*B*4);cudaMalloc(&x3,K*B*4);cudaMalloc(&x4,K*B*4);
    cudaMalloc(&y0,M*B*4);cudaMalloc(&y1,M*B*4);cudaMalloc(&y2,M*B*4);cudaMalloc(&y3,M*B*4);cudaMalloc(&y4,M*B*4);
    cudaMalloc(&di,M*K);cudaMalloc(&da,M*nb*2);cudaMalloc(&dl,LZ*2);
    cudaMemcpy(di,hi.data(),M*K,cudaMemcpyHostToDevice);cudaMemcpy(da,ha.data(),M*nb*2,cudaMemcpyHostToDevice);
    cudaMemcpy(dl,hl.data(),LZ*2,cudaMemcpyHostToDevice);cudaMemcpyToSymbol(lut_const,hl.data(),LZ*2);
    cudaMemcpy(x0,hx.data(),K*B*4,cudaMemcpyHostToDevice);cudaMemcpy(x1,hx.data(),K*B*4,cudaMemcpyHostToDevice);
    cudaMemcpy(x2,hx.data(),K*B*4,cudaMemcpyHostToDevice);cudaMemcpy(x3,hx.data(),K*B*4,cudaMemcpyHostToDevice);cudaMemcpy(x4,hx.data(),K*B*4,cudaMemcpyHostToDevice);

    dim3 gd((K+255)/256,M),bd(256),gm(M),bm(256),ge(M,B),be(256);

    auto tm = [&](auto f) {
        cudaEvent_t a,b;cudaEventCreate(&a);cudaEventCreate(&b);
        for(int i=0;i<WU;i++)f();cudaDeviceSynchronize();cudaEventRecord(a);
        for(int i=0;i<IT;i++)f();cudaEventRecord(b);cudaEventSynchronize(b);
        float ms;cudaEventElapsedTime(&ms,a,b);ms/=IT;
        cudaEventDestroy(a);cudaEventDestroy(b);return ms;
    };

    printf("--- 耗时 (O3, %d iters) ---\n", IT);

    // 手动计时，不用宏（避开花括号逗号冲突）
    {
        cudaEvent_t e0,e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
        auto go = [&](const char *n, auto f, dim3 g, dim3 b) {
            for(int i=0;i<WU;i++)f(); cudaDeviceSynchronize();
            cudaEventRecord(e0);
            for(int i=0;i<IT;i++)f();
            cudaEventRecord(e1); cudaEventSynchronize(e1);
            float ms; cudaEventElapsedTime(&ms,e0,e1); ms/=IT;
            cudaError_t e = cudaGetLastError();
            printf("  %-20s %8.3f ms grid=%-5d block=%d", n, ms, g.x*g.y, b.x*b.y);
            if (e) printf("  ERR: %s", cudaGetErrorString(e));
            printf("\n");
        };
        go("K0 deq+matmul", [&]{k0_deq<<<gd,bd>>>(w,di,da,dl,M,K);k0_mm<<<gm,bm>>>(y0,w,x0,M,K,B);}, gd, bd);
        go("K1 fused naive", [&]{k1_fused_naive<<<gm,bm>>>(y1,di,da,x1,M,K,B);}, gm, bm);
        go("K2 fused smemLUT", [&]{k2_smem_lut<<<gm,bm>>>(y2,di,da,dl,x2,M,K,B);}, gm, bm);
        go("K3 fused uint4", [&]{k3_uint4<<<gm,bm>>>(y3,di,da,x3,M,K,B);}, gm, bm);
        go("K4 fused per-elem", [&]{k4_per_elem<<<ge,be>>>(y4,di,da,x4,M,K,B);}, ge, be);
        cudaEventDestroy(e0); cudaEventDestroy(e1);
    }
    printf("\nncu 分析:\n  ncu --set full -o profile_k1 --kernel-name \"k1_fused_naive\" ./blocklut_kernels\n");
    cudaFree(w);cudaFree(x0);cudaFree(x1);cudaFree(x2);cudaFree(x3);cudaFree(x4);
    cudaFree(y0);cudaFree(y1);cudaFree(y2);cudaFree(y3);cudaFree(y4);
    cudaFree(di);cudaFree(da);cudaFree(dl);
    return 0;
}
