// Standalone 16dp transfer and shared->TMEM copy service/drain probes.
#include "tmem_ops.cuh"
#include <cstdio>
#include <cstdlib>
#include <vector>
#define CK(x) do { auto e=(x); if(e!=cudaSuccess) { \
  fprintf(stderr,"%s: %s\n",#x,cudaGetErrorString(e)); exit(2); } } while(0)
__host__ __device__ unsigned value(unsigned row, unsigned col) {
  return 0x100000u + row*1024 + col;
}
template<int F> __host__ __device__ unsigned physical_row(unsigned tid, unsigned r) {
  unsigned lane=tid%32;
  if constexpr(F==0) return tid/32*32 + lane/4 + lane%2*8;
  if constexpr(F==1) return tid/32*32 + lane/4 + r%2*8;
  return tid/32*32 + lane/4 + (r%4)/2*8;
}
template<int F> __host__ __device__ unsigned physical_col(unsigned tid, unsigned r) {
  unsigned lane=tid%32;
  if constexpr(F==0) return lane%4/2 + r*2;
  if constexpr(F==1) return lane%4 + r/2*4;
  return lane%4*2 + r%2 + r/4*8;
}
template<int F, bool Store> __device__ __forceinline__ void transfer(unsigned a,unsigned* v) {
#define FOUR(SHAPE,REPEAT) \
  if constexpr(Store) asm volatile("tcgen05.st.sync.aligned." SHAPE "." REPEAT ".b32 [%0],{%1,%2,%3,%4};" \
    ::"r"(a),"r"(v[0]),"r"(v[1]),"r"(v[2]),"r"(v[3]):"memory"); \
  else asm volatile("tcgen05.ld.sync.aligned." SHAPE "." REPEAT ".b32 {%0,%1,%2,%3},[%4];" \
    :"=r"(v[0]),"=r"(v[1]),"=r"(v[2]),"=r"(v[3]):"r"(a):"memory")
  if constexpr(F==0) { FOUR("16x64b","x4"); }
  if constexpr(F==1) { FOUR("16x128b","x2"); }
  if constexpr(F==2) { FOUR("16x256b","x1"); }
#undef FOUR
  if constexpr(F==3) {
    if constexpr(Store) asm volatile("tcgen05.st.sync.aligned.16x256b.x2.b32 [%0],{%1,%2,%3,%4,%5,%6,%7,%8};"
      ::"r"(a),"r"(v[0]),"r"(v[1]),"r"(v[2]),"r"(v[3]),"r"(v[4]),"r"(v[5]),"r"(v[6]),"r"(v[7]):"memory");
    else asm volatile("tcgen05.ld.sync.aligned.16x256b.x2.b32 {%0,%1,%2,%3,%4,%5,%6,%7},[%8];"
      :"=r"(v[0]),"=r"(v[1]),"=r"(v[2]),"=r"(v[3]),"=r"(v[4]),"=r"(v[5]),"=r"(v[6]),"=r"(v[7]):"r"(a):"memory");
  }
}
template<int F> __device__ __forceinline__ void copy(unsigned dest,unsigned long long desc) {
#define COPY(SHAPE) asm volatile("tcgen05.cp.cta_group::1." SHAPE " [%0],%1;"::"r"(dest),"l"(desc):"memory")
  if constexpr(F==0) { COPY("128x256b"); }
  if constexpr(F==1) { COPY("128x128b"); }
  if constexpr(F==2) { COPY("64x128b.warpx2::02_13"); }
  if constexpr(F==3) { COPY("64x128b.warpx2::01_23"); }
  if constexpr(F==4) { COPY("32x128b.warpx4"); }
#undef COPY
}
__device__ __forceinline__ void commit_wait(unsigned bar,unsigned& parity) {
  asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];"::"r"(bar):"memory");
  asm volatile("{ .reg .pred p; again: mbarrier.try_wait.parity.shared::cta.b64 p,[%0],%1; @!p bra again; }"
    ::"r"(bar),"r"(parity):"memory");
  parity^=1;
  asm volatile("tcgen05.fence::after_thread_sync;":::"memory");
}
template<int Kind,int F,int Depth> __global__ void probe(unsigned* out,unsigned long long* clocks,
                                                        unsigned iterations,unsigned warps,bool drain) {
  __shared__ unsigned base;
  __shared__ __align__(1024) unsigned input[4096];
  __shared__ __align__(8) unsigned long long barrier;
  unsigned tid=threadIdx.x,bar=__cvta_generic_to_shared(&barrier);
  for(unsigned i=tid;i<4096;i+=128) input[i]=17+i;
  if(tid==0) asm volatile("mbarrier.init.shared::cta.b64 [%0],1;"::"r"(bar):"memory");
  if(tid<32) asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0],128;"
    ::"r"((unsigned)__cvta_generic_to_shared(&base)):"memory");
  __syncthreads();
  for(unsigned col=0;col<64;col+=4) {
    unsigned v[4];
    for(unsigned j=0;j<4;++j) v[j]=Kind==0?value(tid,col+j):~value(tid,col+j);
    tstore<4>(base+col,v);
  }
  asm volatile("tcgen05.wait::st.sync.aligned; tcgen05.fence::before_thread_sync; fence.proxy.async.shared::cta;":::"memory");
  __syncthreads();
  asm volatile("tcgen05.fence::after_thread_sync;":::"memory");
  unsigned long long begin=0,end=0;
  unsigned v[Depth][8]={},sum=0;
  if constexpr(Kind<2) {
    if(tid<warps*32) {
      for(unsigned d=0;d<Depth;++d) for(unsigned j=0;j<(F==3?8:4);++j)
        v[d][j]=value(physical_row<F>(tid,j),d*16+physical_col<F>(tid,j));
      begin=clock64();
#pragma unroll 1
      for(unsigned i=0;i<iterations;++i) {
#pragma unroll
        for(unsigned d=0;d<Depth;++d) {
          transfer<F,Kind==1>(base+d*16,v[d]);
          if(drain) {
            if constexpr(Kind==0) asm volatile("tcgen05.wait::ld.sync.aligned;":::"memory");
            else asm volatile("tcgen05.wait::st.sync.aligned;":::"memory");
          }
        }
        if constexpr(Kind==0) {
          if(!drain) asm volatile("tcgen05.wait::ld.sync.aligned;":::"memory");
          for(unsigned d=0;d<Depth;++d) sum+=v[d][0];
        } else if(!drain) asm volatile("tcgen05.wait::st.sync.aligned;":::"memory");
      }
      end=clock64();
    }
  } else {
    if(tid==0) {
      unsigned long long desc=(__cvta_generic_to_shared(input)>>4)|(1ull<<46);
      // Every copy shape uses the matrix descriptor strides, including broadcast.
      desc|=8ull<<32;
      if constexpr(F<2) desc|=128ull<<16;
      unsigned parity=0;
      begin=clock64();
#pragma unroll 1
      for(unsigned i=0;i<iterations;++i) {
#pragma unroll
        for(unsigned d=0;d<Depth;++d) {
          copy<F>(base+d*8,desc);
          if(drain) commit_wait(bar,parity);
        }
        if(!drain) commit_wait(bar,parity);
      }
      end=clock64();
    }
  }
  asm volatile("tcgen05.fence::before_thread_sync;":::"memory");
  __syncthreads();
  asm volatile("tcgen05.fence::after_thread_sync;":::"memory");
  if(tid%32==0) { clocks[tid/32*2]=begin;clocks[tid/32*2+1]=end; }
  out[8192+tid]=sum;
  // Check physical storage as well as every last loaded register.
  for(unsigned col=0;col<64;col+=4) {
    unsigned q[4];tload<4>(base+col,q);
    asm volatile("tcgen05.wait::ld.sync.aligned;":::"memory");
    for(unsigned j=0;j<4;++j) out[tid*64+col+j]=q[j];
  }
  for(unsigned d=0;d<Depth;++d) for(unsigned j=0;j<8;++j)
    out[8320+tid*32+d*8+j]=v[d][j];
  __syncthreads();
  if(tid<32) {
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;":::"memory");
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0,128;"::"r"(base):"memory");
  }
}
template<int K,int F,int D> int run(unsigned n,unsigned warps,bool drain) {
  std::vector<unsigned> out(12416);unsigned* dev;unsigned long long* time;
  CK(cudaMalloc(&dev,out.size()*4));CK(cudaMalloc(&time,64));
  probe<K,F,D><<<1,128>>>(dev,time,n,warps,drain);CK(cudaGetLastError());CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(out.data(),dev,out.size()*4,cudaMemcpyDeviceToHost));
  unsigned long long clocks[8];CK(cudaMemcpy(clocks,time,64,cudaMemcpyDeviceToHost));unsigned errors=0;
  for(unsigned row=0;row<128;++row) for(unsigned col=0;col<64;++col) {
    unsigned expected=K==0?value(row,col):~value(row,col);
    if constexpr(K==1) if(row/32<warps && row%32<16 && col/16<D && col%16<(F==3?16:8)) expected=value(row,col);
    if constexpr(K==2) if(col/8<D && col%8<(F==0?8:4)) {
      unsigned source=F==2?row%64:F==3?row/64*32+row%32:F==4?row%32:row;
      unsigned offset=source*4+col%8;
      if constexpr(F<2) offset=row*4+(col%8/4)*512+col%4;
      expected=17+offset;
    }
    if(out[row*64+col]!=expected) ++errors;
  }
  if constexpr(K==0) for(unsigned tid=0;tid<warps*32;++tid) {
    unsigned sum=0;
    for(unsigned d=0;d<D;++d) for(unsigned j=0;j<(F==3?8:4);++j) {
      unsigned expected=value(physical_row<F>(tid,j),d*16+physical_col<F>(tid,j));
      errors+=out[8320+tid*32+d*8+j]!=expected;
      if(j==0) sum+=expected*n;
    }
    errors+=out[8192+tid]!=sum;
  }
  for(unsigned w=0;w<4;++w) printf("CLOCK,%u,%llu,%llu\n",w,clocks[w*2],clocks[w*2+1]);
  printf("RESULT,kind=%d,form=%d,depth=%d,warps=%u,iterations=%u,drain=%u,verification=%s,errors=%u\n",K,F,D,warps,n,drain,errors?"FAIL":"PASS",errors);
  CK(cudaFree(dev));CK(cudaFree(time));return errors?1:0;
}
template<int K,int F> int depth(unsigned d,unsigned n,unsigned w,bool drain) {
  return d==1?run<K,F,1>(n,w,drain):run<K,F,4>(n,w,drain);
}
template<int K> int shape(unsigned f,unsigned d,unsigned n,unsigned w,bool drain) {
  switch(f) {
#define F(X) case X:return depth<K,X>(d,n,w,drain)
    F(0);F(1);F(2);F(3);case 4:if constexpr(K==2) return depth<K,4>(d,n,w,drain);
#undef F
  } return 2;
}
int main(int argc,char** argv) {
  if(argc!=7) return 2;
  unsigned k=atoi(argv[1]),f=atoi(argv[2]),d=atoi(argv[3]),w=atoi(argv[4]),n=atoi(argv[5]),drain=atoi(argv[6]);
  if(k>2||f>(k==2?4:3)||(d!=1&&d!=4)||(w!=1&&w!=4)||!n||n>4096||drain>1||(k==2&&w!=4))return 2;
  if(k==0)return shape<0>(f,d,n,w,drain);
  if(k==1)return shape<1>(f,d,n,w,drain);
  return shape<2>(f,d,n,w,drain);
}
