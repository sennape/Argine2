// Argine2 — deterministic driver of the REAL expert-store cache, for the
// runtime-vs-replay fidelity test (tests/test_fidelity.py).
//
// It drives argine-store.cpp exactly as argine-experts.cpp does, but on one
// thread and with a deterministic routing sequence, and writes a format-v2
// research trace. The Python StoreReplay must then reproduce every decision
// (hit/miss and every eviction victim) of this trace.
//
// Per step and layer: one R record, then three MoE nodes (gate, up, down).
// At each node: unpin the previous node's slices (as each thread does at node
// entry), then request all three kinds of the layer's experts in k order
// (kind-major, experts ascending), pinning each, then an N record.
//
// pools = 2 reproduces llama.cpp with --threads != --threads-batch: prompt
// steps are computed by one persistent thread, decode steps by another (run
// strictly one after the other, so the run stays deterministic). Each thread
// keeps its own pins until IT enters its next node, so the prompt thread's
// last slices stay pinned through every decode step and vice versa — the
// state that made the first Windows v2 validation diverge (EXP-021). The N
// record's thread count tells the pools apart (1 = decode, 2 = prompt), as
// nth does in the runtime.
//
//   g++ -O1 -std=c++17 argine-store.cpp argine-research.cpp \
//       argine-store-fidelity-test.cpp -o fidtest -lpthread
//   ./fidtest <store.psxs> <n_layers> <n_experts> <trace> <budget> <shards> <steps> <seed> [pools]

#include "argine-research.h"
#include "argine-store.h"

#include <algorithm>
#include <condition_variable>
#include <cstdio>
#include <functional>
#include <mutex>
#include <thread>
#include <cstdlib>
#include <cstring>
#include <set>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 9) {
        fprintf(stderr, "usage: %s <store> <n_layers> <n_experts> <trace> <budget> <shards> <steps> <seed>\n",
                argv[0]);
        return 2;
    }
    const char *   path     = argv[1];
    const uint32_t n_layers = (uint32_t) atoi(argv[2]);
    const uint32_t n_exp    = (uint32_t) atoi(argv[3]);
    const char *   trace    = argv[4];
    const uint64_t budget   = strtoull(argv[5], nullptr, 10);
    const unsigned shards   = (unsigned) atoi(argv[6]);
    const int      steps    = atoi(argv[7]);
    uint32_t       rng      = (uint32_t) atoi(argv[8]) * 2654435761u + 1;
    const int      n_pools  = argc > 9 ? atoi(argv[9]) : 1;

    psx_store * s = psx_open(path, 0);
    if (psx_error(s)) {
        fprintf(stderr, "store: %s\n", psx_error(s));
        return 1;
    }
    arx_debug_open(trace);
    char num[32];
    arx_config("instrumentation", "argine2-research-v2");
    snprintf(num, sizeof(num), "%llu", (unsigned long long) budget);
    arx_config("budget_bytes", num);
    snprintf(num, sizeof(num), "%u", shards);
    arx_config("cache_shards", num);
    arx_config("read_mode", "sincrono");

    // one region per (layer, kind), slices at e * slab
    std::vector<psx_region *> region(n_layers * PSX_N_KINDS);
    std::vector<size_t>       slab(n_layers * PSX_N_KINDS);
    for (uint32_t l = 0; l < n_layers; l++) {
        for (uint32_t k = 0; k < PSX_N_KINDS; k++) {
            const size_t sz = psx_slice_read_bytes(s, l, 0, (psx_kind) k);
            slab[l * PSX_N_KINDS + k]   = sz;
            region[l * PSX_N_KINDS + k] = psx_region_new(sz * n_exp, sz, n_exp);
        }
    }
    psx_cache * c = psx_cache_new_n(budget, shards);

    struct Pin { psx_region * r; size_t off; };
    std::vector<Pin> pool_pins[2];   // one list per thread, as thread_local in the runtime
    int failures = 0;

    // A persistent worker per pool; the main thread hands it one step at a
    // time and waits, so execution is serial and deterministic.
    struct Worker {
        std::mutex m; std::condition_variable cv;
        std::function<void()> job; bool quit = false; bool busy = false;
        std::thread th;
        void start() {
            th = std::thread([this] {
                for (;;) {
                    std::function<void()> j;
                    {
                        std::unique_lock<std::mutex> lk(m);
                        cv.wait(lk, [this] { return quit || (busy && job); });
                        if (quit) return;
                        j = job;
                    }
                    j();
                    arx_flush_thread();
                    { std::lock_guard<std::mutex> lk(m); job = nullptr; busy = false; }
                    cv.notify_all();
                }
            });
        }
        void run(std::function<void()> j) {
            { std::lock_guard<std::mutex> lk(m); job = std::move(j); busy = true; }
            cv.notify_all();
            std::unique_lock<std::mutex> lk(m);
            cv.wait(lk, [this] { return !busy; });
        }
        void stop() {
            { std::lock_guard<std::mutex> lk(m); quit = true; }
            cv.notify_all();
            th.join();
        }
    };
    Worker workers[2];
    if (n_pools == 2) {
        workers[0].start();
        workers[1].start();
    }

    for (int g = 0; g < steps; g++) {
      // every 25th step is a "prompt" step of 3 tokens, the rest decode
      const uint32_t n_tok = (g % 25 == 0) ? 3 : 1;
      const int      pool  = (n_pools == 2 && n_tok > 1) ? 1 : 0;
      std::vector<Pin> & pins = pool_pins[pool];
      auto step = [&, g, n_tok, pool]() {
        const uint32_t k_use = 2;
        for (uint32_t l = 0; l < n_layers; l++) {
            std::vector<int32_t> ids;
            for (uint32_t t = 0; t < n_tok; t++) {
                std::set<int32_t> chosen;
                while (chosen.size() < k_use) {
                    rng = rng * 1103515245u + 12345u;
                    // skewed: 60% from a small hot set
                    const uint32_t r = rng >> 8;
                    const int32_t e = (int32_t) ((r % 10) < 6 ? (r / 10) % 3 : (r / 10) % n_exp);
                    chosen.insert(e);
                }
                for (int32_t e : chosen) ids.push_back(e);
            }
            arx_route((uint64_t) g, l, n_tok, k_use, ids.data());
            std::vector<uint32_t> need(ids.begin(), ids.end());
            std::sort(need.begin(), need.end());
            need.erase(std::unique(need.begin(), need.end()), need.end());
            for (uint32_t node_kind = 0; node_kind < PSX_N_KINDS; node_kind++) {
                const uint64_t t0 = arx_now_ns();
                for (auto & p : pins) psx_cache_unpin(c, p.r, p.off);  // node entry
                pins.clear();
                for (uint32_t kind = 0; kind < PSX_N_KINDS; kind++) {
                    for (uint32_t e : need) {
                        psx_region * r = region[l * PSX_N_KINDS + kind];
                        const size_t off = (size_t) e * slab[l * PSX_N_KINDS + kind];
                        if (psx_cache_ensure(c, s, r, off, l, e, (psx_kind) kind)) {
                            pins.push_back(Pin{ r, off });
                        } else {
                            failures++;
                        }
                    }
                }
                arx_node((uint64_t) g, l, node_kind, t0, arx_now_ns(), (uint32_t) need.size(),
                         (uint32_t) (pool + 1));
                arx_flush_thread();
            }
        }
      };
      if (n_pools == 2) {
          workers[pool].run(step);
      } else {
          step();
      }
      {
        psx_cache_stats st;
        psx_cache_get_stats(c, &st);
        arx_snapshot((uint64_t) g, st.bytes_resident, st.budget, st.hits, st.misses, st.evictions,
                     st.bytes_read, st.violazioni);
      }
    }
    if (n_pools == 2) {
        workers[0].stop();
        workers[1].stop();
    }
    arx_flush_thread();
    arx_flush_file(0);
    arx_debug_open(nullptr);
    psx_cache_stats st;
    psx_cache_get_stats(c, &st);
    printf("hits %llu misses %llu evictions %llu violations %llu failures %d\n",
           (unsigned long long) st.hits, (unsigned long long) st.misses,
           (unsigned long long) st.evictions, (unsigned long long) st.violazioni, failures);
    return 0;
}
