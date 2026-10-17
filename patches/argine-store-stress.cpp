// Stress: budget stretto, sfratti durante letture concorrenti, contenuto
// verificato a ogni accesso. E' il caso che rompe le cache scritte in fretta.
#include "argine-store.h"
#include <cstdio>
#include <cstring>
#include <random>
#include <thread>
#include <vector>
#include <cstdlib>
int main(int argc, char** argv){
    if (argc<5) return 2;
    const uint32_t NL=(uint32_t)atoi(argv[3]), NE=(uint32_t)atoi(argv[4]);
    psx_store* s = psx_open(argv[1], 0);
    if (psx_error(s)) { printf("errore: %s\n", psx_error(s)); return 1; }
    const size_t slice = psx_slice_read_bytes(s,0,0,PSX_GATE);
    // riferimento completo
    std::vector<std::vector<unsigned char>> ref(NL*NE);
    for (uint32_t l=0;l<NL;l++) for (uint32_t x=0;x<NE;x++){
        ref[l*NE+x].assign(slice,0);
        psx_read_slice(s,l,x,PSX_GATE,ref[l*NE+x].data());
    }
    // budget per sole 3 slice su NL*NE: sfratti continui
    psx_cache* c = psx_cache_new((uint64_t)slice*(uint64_t)atoi(getenv("SLOTS")?getenv("SLOTS"):"3"));
    psx_region* r = psx_region_new(slice*NL*NE, slice, NL*NE);
    psx_debug_set_read_delay_us((unsigned)atoi(getenv("DLY")?getenv("DLY"):"50"));
    const int NT=atoi(getenv("NT")?getenv("NT"):"16");
    std::vector<std::thread> ts; std::vector<size_t> err(NT,0), ok(NT,0);
    for (int t=0;t<NT;t++) ts.emplace_back([&,t]{
        std::mt19937 rnd(1234+t);
        for (int i=0;i<300;i++){
            uint32_t l=rnd()%NL, x=rnd()%NE;
            size_t off=(size_t)(l*NE+x)*slice;
            if (!psx_cache_ensure(c,s,r,off,l,x,PSX_GATE)) { err[t]++; continue; }
            const unsigned char* p=(unsigned char*)psx_region_base(r)+off;
            if (memcmp(p, ref[l*NE+x].data(), psx_get(s,l,x)->size[PSX_GATE])!=0) err[t]++;
            else ok[t]++;
            psx_cache_unpin(c,r,off);
        }
    });
    for(auto&th:ts) th.join();
    psx_debug_set_read_delay_us(0);
    size_t E=0,O=0; for(int t=0;t<NT;t++){E+=err[t];O+=ok[t];}
    psx_cache_stats st; psx_cache_get_stats(c,&st);
    printf("accessi corretti %zu, errori %zu\n", O, E);
    printf("hit %llu miss %llu sfratti %llu\n",(unsigned long long)st.hits,
           (unsigned long long)st.misses,(unsigned long long)st.evictions);
    printf("residenti %llu budget %llu impegnati %zu\n",
           (unsigned long long)st.bytes_resident,(unsigned long long)st.budget,
           psx_region_committed(r));
    const bool bene = E==0 && st.bytes_resident<=st.budget
                      && psx_region_committed(r)<=st.budget && st.evictions>0;
    printf("%s\n", bene?"STRESS OK":"STRESS FALLITO");
    psx_cache_free(c); psx_region_free(r); psx_close(s);
    return bene?0:1;
}
