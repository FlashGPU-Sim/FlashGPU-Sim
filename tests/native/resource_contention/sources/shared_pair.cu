// Matched shared/TMA streams, using retained B200 transfer helpers.
#include "tma_helpers.cuh"
#include <string>
#include <stdexcept>

namespace pair_probe {
constexpr int Threads=512, Shared=224*1024, Meta=208*1024;
enum Kind { LDS=0, STS=1, TMA_LD=2 };
struct Config { int kind[2]={LDS,LDS}, addr[2]={4,16}, slots[2]={1,1}; int active=3,warps=8,mib=64; bool dry=false; };
struct Control { uint64_t bars[4],start,wall,done[Threads],sums[Threads]; unsigned errors[Threads]; };
struct Result { uint64_t cycles[2],wall,checksum[2]; unsigned base,errors; };
static_assert(sizeof(Control)<12*1024);
__device__ unsigned cvta(const void* p) { return smem_addr_u32(p); }

struct WorkerResult { uint64_t cycles=0, sum=0; };
template<int K> __device__ WorkerResult stream(int s,Config o,const CUtensorMap* map,
                                      unsigned char* mem,unsigned base,Control* c) {
  WorkerResult result;
  int local=threadIdx.x-s*256;
  unsigned address=o.addr[s]*1024;
  uint64_t bytes=uint64_t(o.mib)*1024*1024;
  if constexpr(K==TMA_LD) {
    if(local==0) {
      auto* dst=reinterpret_cast<float*>(mem+address-base);
      if(o.slots[s]==2)stream_tma_tiles<2>(map,dst,c->bars+2*s,0,bytes/kTileBytes,1,kLocalityBlockHot);
      else stream_tma_tiles<1>(map,dst,c->bars+2*s,0,bytes/kTileBytes,1,kLocalityBlockHot);
      result.cycles=clock64_now()-c->start;
    }
  } else if(local<o.warps*32) {
    unsigned a=address+local*16,v[8][4];
#pragma unroll
    for(int j=0;j<8;j++)for(int k=0;k<4;k++)v[j][k]=1;
    int loops=bytes/(o.warps*32*16*8);
#pragma unroll 1
    for(int i=0;i<loops;i++) {
#pragma unroll
      for(int j=0;j<8;j++) {
        if constexpr(K==LDS) asm volatile("ld.volatile.shared.v4.u32 {%0,%1,%2,%3},[%4];":"=r"(v[j][0]),"=r"(v[j][1]),"=r"(v[j][2]),"=r"(v[j][3]):"r"(a):"memory");
        else asm volatile("st.volatile.shared.v4.u32 [%0],{%1,%1,%1,%1};"::"r"(a),"r"(1u):"memory");
      }
    }
    uint64_t sum=0;
#pragma unroll
    for(int j=0;j<8;j++)for(int k=0;k<4;k++)sum+=v[j][k];
    result.sum=sum;
    __threadfence_block();
    result.cycles=clock64_now()-c->start;
  }
  return result;
}

template<int A,int B> __global__ void kernel(Config o,const CUtensorMap* map,Result* out) {
  extern __shared__ unsigned char mem[];
  unsigned base=cvta(mem),tid=threadIdx.x;
  if(base>4096 || base+Shared<Meta+sizeof(Control)) { if(!tid)out->errors=1; return; }
  auto* c=reinterpret_cast<Control*>(mem+Meta-base);
  c->done[tid]=c->sums[tid]=0;
  // Initialize all reservations, including inactive controls, outside timing.
  for(int s=0;s<2;s++) {
    unsigned count=o.kind[s]==TMA_LD?o.slots[s]*kTileBytes/4:o.warps*32*4;
    auto* ptr=reinterpret_cast<unsigned*>(mem+o.addr[s]*1024-base);
    for(unsigned i=tid;i<count;i+=Threads)ptr[i]=o.kind[s]==STS?~1u:1u;
  }
  __syncthreads();
  // Warm the common 64 KiB global source and both TMA reservations.
  for(int s=0;s<2;s++)if(o.kind[s]==TMA_LD && tid==unsigned(s*256)) {
    auto* dst=reinterpret_cast<float*>(mem+o.addr[s]*1024-base);
    if(o.slots[s]==2)stream_tma_tiles<2>(map,dst,c->bars+2*s,0,16,1,kLocalityBlockHot);
    else stream_tma_tiles<1>(map,dst,c->bars+2*s,0,16,1,kLocalityBlockHot);
  }
  __syncthreads();
  if(!tid)c->start=clock64_now();
  __syncthreads();
  WorkerResult worker;
  if(tid<256) { if(o.active&1)worker=stream<A>(0,o,map,mem,base,c); }
  else { if(o.active&2)worker=stream<B>(1,o,map,mem,base,c); }
  __syncthreads();
  c->done[tid]=worker.cycles; c->sums[tid]=worker.sum;
  __syncthreads();
  if(!tid)c->wall=clock64_now()-c->start;
  // Outside the timing interval: check the complete reservations in parallel.
  // A serial one-thread scan would dominate small numerical runs and pollute
  // full-kernel profiler statistics with verification LDS instructions.
  fence_proxy_async();
  unsigned errors=0;
  for(int s=0;s<2;s++) {
    unsigned count=o.kind[s]==TMA_LD?o.slots[s]*kTileBytes/4:o.warps*32*4;
    unsigned expected=o.kind[s]==TMA_LD?0x5a5a5a5au:
        o.kind[s]==STS && !(o.active&(1<<s))?~1u:1u;
    auto* ptr=reinterpret_cast<volatile unsigned*>(mem+o.addr[s]*1024-base);
    for(unsigned i=tid;i<count;i+=Threads)errors+=ptr[i]!=expected;
  }
  c->errors[tid]=errors;
  __syncthreads();
  if(!tid) {
    Result r{};r.wall=c->wall;r.base=base;
    for(int s=0;s<2;s++)for(int i=0;i<256;i++) {
      r.cycles[s]=max(r.cycles[s],c->done[s*256+i]);r.checksum[s]+=c->sums[s*256+i];
      r.errors+=c->errors[s*256+i];
    }
    for(int s=0;s<2;s++) {
      if((o.active&(1<<s)) && o.kind[s]!=TMA_LD)r.errors+=r.checksum[s]!=uint64_t(o.warps)*32*32;
    }
    *out=r;
  }
}

template<int A,int B> void launch(Config o,const CUtensorMap* map,Result* out) {
 CUDA_CHECK(cudaFuncSetAttribute(kernel<A,B>,cudaFuncAttributeMaxDynamicSharedMemorySize,Shared));
 CUDA_CHECK(cudaFuncSetAttribute(kernel<A,B>,cudaFuncAttributePreferredSharedMemoryCarveout,cudaSharedmemCarveoutMaxShared));
 kernel<A,B><<<1,Threads,Shared>>>(o,map,out);
}
void dispatch(Config o,const CUtensorMap* m,Result* r) {
#define CASE(A,B) case A*3+B:launch<A,B>(o,m,r);break
 switch(o.kind[0]*3+o.kind[1]) {CASE(0,0);CASE(0,1);CASE(0,2);CASE(1,0);CASE(1,1);CASE(1,2);CASE(2,0);CASE(2,1);CASE(2,2);}
#undef CASE
}
int parse_kind(std::string s) {if(s=="lds")return LDS;if(s=="sts")return STS;if(s=="tma")return TMA_LD;throw std::runtime_error("kind must be lds/sts/tma");}
}
int main(int argc,char** argv) {
 using namespace pair_probe;Config o;
 try {
  for(int i=1;i<argc;i++) {
   std::string a=argv[i];if(a=="--dry-run"){o.dry=true;continue;}
   auto eq=a.find('=');if(eq==std::string::npos)throw std::runtime_error("expected --option=value");
   auto k=a.substr(0,eq),v=a.substr(eq+1);
   if(k=="--a")o.kind[0]=parse_kind(v);else if(k=="--b")o.kind[1]=parse_kind(v);
   else if(k=="--addr-a")o.addr[0]=std::stoi(v);else if(k=="--addr-b")o.addr[1]=std::stoi(v);
   else if(k=="--slots-a")o.slots[0]=std::stoi(v);else if(k=="--slots-b")o.slots[1]=std::stoi(v);
   else if(k=="--active")o.active=std::stoi(v);else if(k=="--warps")o.warps=std::stoi(v);
   else if(k=="--mib")o.mib=std::stoi(v);else throw std::runtime_error("unknown option");
  }
  if((o.warps!=4&&o.warps!=8)||o.active<1||o.active>3||o.mib<1||o.mib>256)throw std::runtime_error("invalid warps/active/mib");
  int end[2];
  for(int s=0;s<2;s++) {
   if(o.slots[s]<1||o.slots[s]>2||o.addr[s]<4)throw std::runtime_error("invalid slots/address");
   end[s]=o.addr[s]*1024+(o.kind[s]==TMA_LD?o.slots[s]*kTileBytes:o.warps*32*16);
   if(end[s]>Shared || (o.addr[s]*1024<Meta+int(sizeof(Control)) && end[s]>Meta))throw std::runtime_error("payload overlaps metadata or exceeds allocation");
  }
  if(o.addr[0]*1024<end[1]&&o.addr[1]*1024<end[0])throw std::runtime_error("payloads overlap");
  printf("configuration=VALID kinds=%d,%d ranges=%d:%d,%d:%d active=%d warps=%d mib=%d\n",o.kind[0],o.kind[1],o.addr[0]*1024,end[0],o.addr[1]*1024,end[1],o.active,o.warps,o.mib);
  if(o.dry){puts("hardware_execution=NOT_RUN");return 0;}
  CUDA_CHECK(cudaSetDevice(0));CU_CHECK(cuInit(0));
  float* data;CUtensorMap* map;Result* result;
  CUDA_CHECK(cudaMalloc(&data,kTileBytes));CUDA_CHECK(cudaMemset(data,0x5a,kTileBytes));
  CUtensorMap host=encode_map(data,1);CUDA_CHECK(cudaMalloc(&map,sizeof(host)));CUDA_CHECK(cudaMemcpy(map,&host,sizeof(host),cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMalloc(&result,sizeof(Result)));CUDA_CHECK(cudaMemset(result,0,sizeof(Result)));
  dispatch(o,map,result);CUDA_CHECK(cudaGetLastError());CUDA_CHECK(cudaDeviceSynchronize());
  Result r;CUDA_CHECK(cudaMemcpy(&r,result,sizeof(r),cudaMemcpyDeviceToHost));
  uint64_t bytes=uint64_t(o.mib)*1024*1024,total=0;
  for(int s=0;s<2;s++) {
   bool active=o.active&(1<<s);if(active)total+=bytes;
   if(active&&!r.cycles[s])r.errors++;
   printf("stream=%d bytes=%llu cycles=%llu Bpc=%.6f\n",s,(unsigned long long)(active?bytes:0),(unsigned long long)r.cycles[s],active&&r.cycles[s]?double(bytes)/r.cycles[s]:0.);
  }
  printf("wall_cycles=%llu combined_Bpc=%.6f shared_base=%u errors=%u verification=%s\n",(unsigned long long)r.wall,r.wall?double(total)/r.wall:0.,r.base,r.errors,r.errors?"FAIL":"PASS");
  CUDA_CHECK(cudaFree(result));CUDA_CHECK(cudaFree(map));CUDA_CHECK(cudaFree(data));return r.errors?1:0;
 } catch(const std::exception& e) {fprintf(stderr,"%s\n",e.what());return 2;}
}
