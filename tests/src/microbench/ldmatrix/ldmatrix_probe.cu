#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#define CK(x) do { cudaError_t e=(x);if(e!=cudaSuccess){fprintf(stderr,"cuda: %s\n",cudaGetErrorString(e));exit(2);}}while(0)
__global__ void ldm_probe(uint32_t* out,unsigned long long* clk,int steps,int trans){
 __shared__ uint32_t s[128];int t=threadIdx.x, lane=t&31, warp=t>>5;
 for(int i=t;i<128;i+=blockDim.x)s[i]=(0x03010000u ^ (i*0x10003u));
 __syncthreads();
 unsigned addr=(unsigned)__cvta_generic_to_shared(&s[lane*4]);
 unsigned a=0,b=0,c=0,d=0; unsigned long long begin=clock64();
 #pragma unroll 1
 for(int k=0;k<steps;k++) {
  unsigned x0,x1,x2,x3;
  if(trans) asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];":"=r"(x0),"=r"(x1),"=r"(x2),"=r"(x3):"r"(addr):"memory");
  else asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];":"=r"(x0),"=r"(x1),"=r"(x2),"=r"(x3):"r"(addr):"memory");
  a+=x0;b+=x1;c+=x2;d+=x3;
 }
 unsigned long long end=clock64();int pos=((blockIdx.x*blockDim.x+t)*4);out[pos]=a;out[pos+1]=b;out[pos+2]=c;out[pos+3]=d;
 if(lane==0)clk[blockIdx.x*(blockDim.x/32)+warp]=end-begin;
}
int main(int argc,char**argv){
 int steps=argc>1?atoi(argv[1]):32,warps=argc>2?atoi(argv[2]):1,trans=argc>3?atoi(argv[3]):0,blocks=argc>4?atoi(argv[4]):1;
 const char* path=argc>5?argv[5]:"output.bin";
 if(steps<1||warps<1||warps>4||blocks<1||blocks>64||(trans!=0&&trans!=1))return 3;
 int threads=warps*32;size_t n=blocks*threads*4;std::vector<uint32_t>out(n);std::vector<unsigned long long>clk(blocks*warps);
 uint32_t *d_out;unsigned long long*d_clk;CK(cudaMalloc(&d_out,n*sizeof(uint32_t)));CK(cudaMalloc(&d_clk,clk.size()*sizeof(unsigned long long)));
 ldm_probe<<<blocks,threads>>>(d_out,d_clk,steps,trans);CK(cudaDeviceSynchronize());CK(cudaMemcpy(out.data(),d_out,n*sizeof(uint32_t),cudaMemcpyDeviceToHost));CK(cudaMemcpy(clk.data(),d_clk,clk.size()*sizeof(unsigned long long),cudaMemcpyDeviceToHost));
 FILE*f=fopen(path,"wb");if(!f)return 4;if(fwrite(out.data(),sizeof(uint32_t),n,f)!=n)return 5;fclose(f);
 printf("steps=%d warps=%d trans=%d blocks=%d clock0=%llu output_words=%zu\n",steps,warps,trans,blocks,clk[0],n);
 CK(cudaFree(d_out));CK(cudaFree(d_clk));return 0;
}
