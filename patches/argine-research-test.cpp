// Argine2 — test of the research instrumentation, independent of llama.cpp.
//
// What it proves:
//   1. tracing is pure observation: the same deterministic access sequence
//      produces identical cache statistics with and without the trace;
//   2. every miss that issued a read has exactly one I record with the same
//      request id, and every I record has a matching 'm' access;
//   3. timestamps are ordered: issue <= complete, wait_start <= wait_end;
//   4. every eviction counted by the cache appears as an E record;
//   5. the hard-budget invariant monitor reports zero violations, and resident
//      bytes never exceed the budget at any snapshot;
//   6. the above holds with several threads using the emit-and-continue path.
//
//   g++ -O1 -g -std=c++17 -fsanitize=thread argine-store.cpp argine-research.cpp
//       argine-research-test.cpp -o arxtest -lpthread
//   ./arxtest <store.psxs> <n_layers> <n_experts> <trace-file>

#include "argine-research.h"
#include "argine-store.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

static int failures = 0;

static void check(bool ok, const char * what) {
    printf("  [%s] %s\n", ok ? "ok" : "NO", what);
    if (!ok) {
        failures++;
    }
}

struct Run {
    psx_cache_stats st{};
};

// Deterministic single-thread sequence through the emit-and-continue API.
static Run run_sequence(psx_store * s, uint32_t n_layers, uint32_t n_experts, size_t slots) {
    const size_t slice = psx_slice_read_bytes(s, 0, 0, PSX_GATE);
    std::vector<psx_region *> regions(n_layers);
    for (uint32_t l = 0; l < n_layers; l++) {
        regions[l] = psx_region_new(slice * n_experts, slice, n_experts);
    }
    psx_cache * c = psx_cache_new_n((uint64_t) slice * slots, 1);
    uint32_t    x = 1;
    for (int tok = 0; tok < 200; tok++) {
        for (uint32_t l = 0; l < n_layers; l++) {
            std::vector<size_t> pinned;
            for (int k = 0; k < 2; k++) {
                x = x * 1103515245u + 12345u;
                // skewed: half of the accesses go to experts 0..1
                const uint32_t e = (x >> 16) % 4 == 0 ? (x >> 8) % n_experts : (x >> 8) % 2;
                if (std::find(pinned.begin(), pinned.end(), (size_t) e * slice) != pinned.end()) {
                    continue;
                }
                if (psx_cache_avvia(c, s, regions[l], e * slice, l, e, PSX_GATE) == 0) {
                    printf("  avvia failed: %s\n", psx_error(s) ? psx_error(s) : "?");
                    continue;
                }
                pinned.push_back((size_t) e * slice);
            }
            psx_cache_attendi(c, s);
            for (size_t off : pinned) {
                psx_cache_unpin(c, regions[l], off);
            }
            arx_flush_thread();
        }
    }
    Run r;
    psx_cache_get_stats(c, &r.st);
    psx_cache_free(c);
    for (auto * rg : regions) {
        psx_region_free(rg);
    }
    return r;
}

static std::vector<std::vector<std::string>> read_records(const char * path) {
    std::vector<std::vector<std::string>> out;
    FILE * f = fopen(path, "rb");
    if (!f) {
        return out;
    }
    char line[1 << 16];
    while (fgets(line, sizeof(line), f)) {
        std::vector<std::string> fields;
        std::string              cur;
        for (char * p = line; *p && *p != '\n'; p++) {
            if (*p == '\t') {
                fields.push_back(cur);
                cur.clear();
            } else {
                cur += *p;
            }
        }
        fields.push_back(cur);
        out.push_back(fields);
    }
    fclose(f);
    return out;
}

int main(int argc, char ** argv) {
    if (argc < 5) {
        fprintf(stderr, "usage: %s <store.psxs> <n_layers> <n_experts> <trace-file>\n", argv[0]);
        return 2;
    }
    const char *   store_path = argv[1];
    const uint32_t n_layers   = (uint32_t) atoi(argv[2]);
    const uint32_t n_experts  = (uint32_t) atoi(argv[3]);
    const char *   trace      = argv[4];

    psx_store * s = psx_open(store_path, 0);
    if (psx_error(s)) {
        printf("store error: %s\n", psx_error(s));
        return 1;
    }

    printf("tracing is observation only\n");
    arx_debug_open(nullptr);  // disabled
    check(!arx_enabled(), "trace disabled without a path");
    const Run off = run_sequence(s, n_layers, n_experts, 3);
    arx_debug_open(trace);
    check(arx_enabled() == 1, "trace enabled with a path");
    const Run on = run_sequence(s, n_layers, n_experts, 3);
    check(off.st.hits == on.st.hits && off.st.misses == on.st.misses &&
              off.st.evictions == on.st.evictions && off.st.bytes_read == on.st.bytes_read,
          "identical hits/misses/evictions/bytes with and without the trace");
    check(on.st.violazioni == 0, "no budget violations (single thread)");

    printf("multi-thread emit-and-continue\n");
    const size_t slice = psx_slice_read_bytes(s, 0, 0, PSX_GATE);
    psx_region * rr    = psx_region_new(slice * n_experts, slice, n_experts);
    psx_cache *  cc    = psx_cache_new_n((uint64_t) slice * 4, 2);
    std::vector<std::thread> ts;
    for (int t = 0; t < 6; t++) {
        ts.emplace_back([&, t] {
            for (int round = 0; round < 60; round++) {
                const uint32_t e = (uint32_t) ((t * 5 + round * 3) % n_experts);
                if (psx_cache_avvia(cc, s, rr, e * slice, 0, e, PSX_GATE) != 0) {
                    psx_cache_attendi(cc, s);
                    psx_cache_unpin(cc, rr, e * slice);
                } else {
                    psx_cache_attendi(cc, s);
                }
                arx_flush_thread();
            }
        });
    }
    for (auto & th : ts) {
        th.join();
    }
    psx_cache_stats mt{};
    psx_cache_get_stats(cc, &mt);
    check(mt.violazioni == 0, "no budget violations (6 threads)");
    check(mt.bytes_resident <= mt.budget, "resident <= budget at the end");
    arx_snapshot(1, mt.bytes_resident, mt.budget, mt.hits, mt.misses, mt.evictions, mt.bytes_read,
                 mt.violazioni);
    arx_flush_thread();
    arx_flush_file(0);
    arx_debug_open(nullptr);  // close the file

    printf("trace content\n");
    const auto recs = read_records(trace);
    check(!recs.empty() && recs[0][0] == "H" && recs[0].size() >= 6 && recs[0][1] == "argine-raw",
          "header record first");
    std::map<std::string, int> miss_req;  // req -> count of 'm'
    std::map<std::string, int> io_req;
    size_t n_access = 0, n_evict = 0, bad_order = 0, bad_flags = 0;
    // seq numbers per shard, from A, E and U records; one run of the cache at
    // a time, so they are collected per cache instance via the C/shard key below
    std::map<std::string, std::vector<unsigned long long>> shard_seqs;
    for (const auto & r : recs) {
        if (r[0] == "A" && r.size() == 11) {
            shard_seqs[r[9]].push_back(strtoull(r[10].c_str(), nullptr, 10));
            n_access++;
            if (r[6] == "m") {
                miss_req[r[8]]++;
            }
        } else if (r[0] == "I" && r.size() == 13) {
            io_req[r[1]]++;
            const unsigned long long issue = strtoull(r[7].c_str(), nullptr, 10);
            const unsigned long long done  = strtoull(r[8].c_str(), nullptr, 10);
            const unsigned long long ws    = strtoull(r[9].c_str(), nullptr, 10);
            const unsigned long long we    = strtoull(r[10].c_str(), nullptr, 10);
            if (issue > done || ws > we) {
                bad_order++;
            }
            const unsigned flags = (unsigned) atoi(r[12].c_str());
            // exactly one of EXACT / BOUND must be set
            if (((flags & ARX_FLAG_COMPLETE_EXACT) != 0) == ((flags & ARX_FLAG_COMPLETE_BOUND) != 0)) {
                bad_flags++;
            }
        } else if (r[0] == "E" && r.size() == 12) {
            n_evict++;
            shard_seqs[r[9]].push_back(strtoull(r[10].c_str(), nullptr, 10));
        } else if (r[0] == "U" && r.size() == 8) {
            shard_seqs[r[6]].push_back(strtoull(r[7].c_str(), nullptr, 10));
        }
    }
    check(n_access > 0, "access records present");
    bool one_to_one = miss_req.size() == io_req.size();
    for (const auto & kv : miss_req) {
        one_to_one = one_to_one && kv.second == 1 && io_req.count(kv.first) && io_req[kv.first] == 1;
    }
    check(one_to_one, "every miss has exactly one I record with the same request id");
    check(bad_order == 0, "issue <= complete and wait_start <= wait_end for every read");
    // Every shard's state changes carry a gapless, duplicate-free sequence
    // 0..N-1, per cache instance. The single-thread cache (1 shard) and the
    // multi-thread cache (2 shards) restart their counters, so shard "0"
    // holds two sequences: check that each value appears at most twice and
    // that the union covers 0..max for every shard.
    {
        bool gapless = !shard_seqs.empty();
        for (auto & kv : shard_seqs) {
            std::map<unsigned long long, int> cnt;
            for (auto v : kv.second) cnt[v]++;
            const unsigned long long mx = cnt.rbegin()->first;
            for (unsigned long long v = 0; v <= mx; v++) {
                if (!cnt.count(v)) gapless = false;
            }
            for (auto & c2 : cnt) {
                if (c2.second > 2) gapless = false;
            }
        }
        check(gapless, "per-shard seq numbers are gapless and unique per cache instance");
        check(shard_seqs.size() == 2, "both shards of the multi-thread cache appear in the trace");
    }
    check(bad_flags == 0, "every completion is flagged either exact or upper bound");
    check(n_evict == on.st.evictions + mt.evictions,
          "every eviction counted by the cache appears as an E record");
    printf("  (%zu accesses, %zu reads, %zu evictions)\n", n_access, io_req.size(), n_evict);

    psx_cache_free(cc);
    psx_region_free(rr);
    psx_close(s);
    printf(failures ? "FAILED: %d\n" : "all ok\n", failures);
    return failures ? 1 : 0;
}
