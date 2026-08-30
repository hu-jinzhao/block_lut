// 单次 launch，专为 ncu 分析
// nvcc -arch=sm_86 -O3 profile_target.cu -lcudart -o /tmp/pt && ncu -o prof ./pt
#include <stdint.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#define M 2048
#define K 1408
#define B 32
#define LZ 256
__constant__ uint16_t lc[LZ];

// 只包含 K0 的两个 kernel（dequant + matmul）
__global__ void deq(float *W, const uint8_t *d, const uint16_t *a, const uint16_t *l) {
    int r=blockIdx.y,cb=blockIdx.x*blockDim.x+threadIdx.x; if(r>=M||cb>=K)return;
    __shared__ uint16_t s[LZ]; for(int i=threadIdx.x;i<LZ;i+=blockDim.x) s[i]=l[i]; __syncthreads();
    for(int c=cb;c<K;c+=blockDim.x*gridDim.x) W[r*K+c]=__bfloat162float(*(const __nv_bfloat16*)&s[d[r*K+c]])*__bfloat162float(*(const __nv_bfloat16*)&a[r*((K+127)/128)+c/128]);
}
__global__ void mm(float *Y, const float *W, const float *X) {
    int r=blockIdx.x;if(r>=M)return;
    for(int c=threadIdx.x;c<B;c+=blockDim.x){float s=0;for(int i=0;i<K;i++) s+=W[r*K+i]*X[i*B+c];Y[r*B+c]=s;}
}
// fused smem LUT
__global__ void fused(float *Y, const uint8_t *d, const uint16_t *a, const uint16_t *l, const float *X) {
    __shared__ uint16_t s[LZ]; for(int i=threadIdx.x;i<LZ;i+=blockDim.x) s[i]=l[i]; __syncthreads();
    int r=blockIdx.x;if(r>=M)return;int na=(K+127)/128;
    for(int c=threadIdx.x;c<B;c+=blockDim.x){float sum=0;for(int i=0;i<K;i++)sum+=__bfloat162float(*(const __nv_bfloat16*)&s[d[r*K+i]])*__bfloat162float(*(const __nv_bfloat16*)&a[r*na+i/128])*X[i*B+c];Y[r*B+c]=sum;}
}

int main(int argc, char **argv) {
    cudaSetDevice(0);
    int mode = argc > 1 ? atoi(argv[1]) : 0;

    int nb=(K+127)/128; size_t szW=M*K*4,szX=K*B*4,szY=M*B*4;
    float *W,*X,*Y; uint8_t *di; uint16_t *da,*dl;
    cudaMalloc(&W,szW);cudaMalloc(&X,szX);cudaMalloc(&Y,szY);
    cudaMalloc(&di,M*K);cudaMalloc(&da,M*nb*2);cudaMalloc(&dl,LZ*2);
    cudaMemset(di,0,M*K);cudaMemset(da,0,M*nb*2);
    float hl[LZ]; for(int i=0;i<LZ;i++){float v=(float)i/255*2-1;uint32_t b;memcpy(&b,&v,4);hl[i]=(b+0x7FFF+((b>>16)&1))>>16;}
    cudaMemcpy(dl,hl,LZ*2,cudaMemcpyHostToDevice);cudaMemcpyToSymbol(lc,hl,LZ*2);
    cudaMemset(X,0,szX);

    dim3 gd((K+255)/256,M),bd(256),gm(M),bm(256);

    // kernel launch
    if (mode == 0) { deq<<<gd,bd>>>(W,di,da,dl); mm<<<gm,bm>>>(Y,W,X); }
    else if (mode == 1) { mm<<<gm,bm>>>(Y,W,X); }
    else { fused<<<gm,bm>>>(Y,di,da,dl,X); }
    cudaDeviceSynchronize();
    return 0;
}
