// Argine2 — research instrumentation for the expert store.
//
// Phase 1 of Argine2 is OBSERVATION ONLY: before any new residency policy is
// designed, the runtime must expose what actually happens on every token. This
// module writes a raw, append-only event log that the Python side
// (`argine research ...`) turns into router traces, I/O timelines and cache
// statistics.
//
// Activation: ARGINE_RESEARCH_TRACE=<path>. Without the variable every hook is
// a single relaxed atomic load and returns. Nothing in this module changes a
// residency decision, a read, or a byte of model weights: it only observes.
//
// Self-contained on purpose (no ggml, no llama.cpp includes), like
// argine-store.{h,cpp}, so it can be compiled and tested on its own.
//
// ---------------------------------------------------------------------------
// Raw log format ("argine-raw v1"), one tab-separated record per line.
// All timestamps are std::chrono::steady_clock nanoseconds (QueryPerformance-
// Counter on Windows). The H record pairs the steady clock with the system
// (UNIX epoch) clock once, so external events can be aligned.
//
//   H  format version steady_ns system_ns pid
//   C  key value                                     (configuration)
//   R  graph t_ns layer n_tokens n_used id0 id1 ...  (router decision; ids are
//                                                     grouped per token, exactly
//                                                     as the router produced them)
//   A  t_ns tid layer expert kind outcome bytes req shard seq
//                                                    (one slice access;
//                                                     outcome h=hit, m=miss that
//                                                     issued a read, w=waited for
//                                                     a read issued by another
//                                                     thread, x=error)
//   I  req tid layer expert kind bytes issue_ns complete_ns wait_start_ns
//      wait_end_ns ok flags                          (one storage read)
//   E  t_ns layer expert kind bytes loaded_ns last_use_ns uses shard seq skipped
//                                                    (eviction)
//   U  t_ns layer expert kind bytes shard seq       (a slot inserted for a miss
//                                                     and removed again because
//                                                     its read failed)
//   N  graph layer kind tid0_start_ns ready_ns n_serve nth       (MoE node:
//                                                     residency phase of one
//                                                     MUL_MAT_ID, measured by
//                                                     thread 0 from entry to the
//                                                     barrier after all reads)
//   S  t_ns graph resident budget hits misses evictions bytes_read violations
//                                                    (cache snapshot, once per
//                                                     graph)
//
// Format version 2 (instrumentation "argine2-research-v2") adds `shard` and
// `seq` to A and E records, `skipped` to E records, and U records. They exist
// so that an offline replay can be checked against the runtime EVENT BY EVENT:
//   shard    index of the cache shard that owns the slice. The shard is chosen
//            by hashing the slice's virtual address, which is not reproducible
//            offline, so it must be recorded.
//   seq      per-shard sequence number of the state change, assigned UNDER the
//            shard mutex at the moment of the decision (hit -> move to front,
//            miss -> insert at front, eviction, undo). Within one shard, seq
//            order IS the order in which the runtime changed that shard's LRU
//            list; timestamps from several threads are not.
//   skipped  slots passed over (pinned, or still being read) while scanning
//            from the LRU tail before the victim was found.
// Version-1 logs (no shard/seq) remain readable.
//
// I-record flags (bit field). They exist so that no proxy is ever presented
// as an exact value:
//   1  COMPLETE_EXACT   completion was observed by a wait that returned for
//                       this request: complete_ns is accurate to the wake-up
//                       latency of the waiting thread.
//   2  COMPLETE_BOUND   the read was found already complete when polled:
//                       complete_ns is an UPPER BOUND (the true completion is
//                       between the previous poll and complete_ns).
//   4  SYNCHRONOUS      blocking read (sync/block modes, or POSIX): issue and
//                       wait coincide, nothing is hidden.
//   8  FALLBACK         asynchronous read degraded to synchronous (no handle
//                       or no event available).
// ---------------------------------------------------------------------------

#ifndef ARGINE_RESEARCH_H
#define ARGINE_RESEARCH_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ARX_FLAG_COMPLETE_EXACT 1u
#define ARX_FLAG_COMPLETE_BOUND 2u
#define ARX_FLAG_SYNCHRONOUS    4u
#define ARX_FLAG_FALLBACK       8u

// 1 if ARGINE_RESEARCH_TRACE is set and the log file could be opened.
// The first call reads the environment; later calls are one atomic load.
int arx_enabled(void);

// Force a path (tests only; production uses the environment variable).
// Passing NULL closes the log.
void arx_debug_open(const char * path);

uint64_t arx_now_ns(void);
uint64_t arx_next_request_id(void);
// Small dense id of the calling thread (0, 1, 2, ...), stable for its lifetime.
uint32_t arx_thread_id(void);

void arx_config(const char * key, const char * value);

void arx_route(uint64_t graph, uint32_t layer, uint32_t n_tokens, uint32_t n_used,
               const int32_t * ids);

void arx_access(uint32_t layer, uint32_t expert, uint32_t kind, char outcome,
                uint64_t bytes, uint64_t request_id, uint32_t shard, uint64_t seq);

void arx_io(uint64_t request_id, uint32_t layer, uint32_t expert, uint32_t kind,
            uint64_t bytes, uint64_t issue_ns, uint64_t complete_ns, uint64_t wait_start_ns,
            uint64_t wait_end_ns, int ok, uint32_t flags);

void arx_evict(uint32_t layer, uint32_t expert, uint32_t kind, uint64_t bytes,
               uint64_t loaded_ns, uint64_t last_use_ns, uint64_t uses, uint32_t shard,
               uint64_t seq, uint32_t skipped);

void arx_undo(uint32_t layer, uint32_t expert, uint32_t kind, uint64_t bytes, uint32_t shard,
              uint64_t seq);

void arx_node(uint64_t graph, uint32_t layer, uint32_t kind, uint64_t start_ns,
              uint64_t ready_ns, uint32_t n_serve, uint32_t nth);

void arx_snapshot(uint64_t graph, uint64_t resident, uint64_t budget, uint64_t hits,
                  uint64_t misses, uint64_t evictions, uint64_t bytes_read,
                  uint64_t violations);

// Records are accumulated in a per-thread buffer and appended to the file in
// one locked write. Call at the end of every node from every thread.
void arx_flush_thread(void);
// Flush the stdio buffer to the OS if at least `min_interval_ms` passed since
// the last flush. llama-server is terminated, not exited, by Argine on Windows,
// so destructors never run: the log must reach the OS during execution.
void arx_flush_file(unsigned min_interval_ms);

#ifdef __cplusplus
}
#endif

#endif  // ARGINE_RESEARCH_H
