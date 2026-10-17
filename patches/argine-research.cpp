// Argine2 — research instrumentation. See argine-research.h for the format.

#include "argine-research.h"

#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>

#if defined(_WIN32)
#    ifndef WIN32_LEAN_AND_MEAN
#        define WIN32_LEAN_AND_MEAN
#    endif
#    include <windows.h>
#    define ARX_GETPID() ((unsigned long) GetCurrentProcessId())
#else
#    include <unistd.h>
#    define ARX_GETPID() ((unsigned long) getpid())
#endif

namespace {

// 0 = not decided yet, 1 = enabled, 2 = disabled
std::atomic<int>      g_state{ 0 };
std::mutex            g_mtx;          // guards g_file and the decision
FILE *                g_file = nullptr;
std::atomic<uint64_t> g_next_req{ 1 };
std::atomic<uint32_t> g_next_tid{ 0 };
std::atomic<uint64_t> g_last_flush_ns{ 0 };

constexpr size_t kThreadBuffer = 64 * 1024;

uint64_t steady_ns() {
    return (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch()).count();
}

uint64_t system_ns() {
    return (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::system_clock::now().time_since_epoch()).count();
}

// Expand "%p" to the process id, so several processes can share a template.
std::string expand_path(const char * p) {
    std::string out;
    for (const char * c = p; *c; c++) {
        if (c[0] == '%' && c[1] == 'p') {
            out += std::to_string(ARX_GETPID());
            c++;
        } else {
            out += *c;
        }
    }
    return out;
}

void write_header_locked() {
    fprintf(g_file, "H\targine-raw\t2\t%" PRIu64 "\t%" PRIu64 "\t%lu\n", steady_ns(), system_ns(),
            ARX_GETPID());
    fflush(g_file);
}

bool open_locked(const char * path) {
    if (g_file) {
        fclose(g_file);
        g_file = nullptr;
    }
    if (!path || !*path) {
        return false;
    }
    const std::string full = expand_path(path);
    g_file = fopen(full.c_str(), "wb");
    if (!g_file) {
        fprintf(stderr, "argine: research trace: cannot open %s\n", full.c_str());
        return false;
    }
    // 1 MiB stdio buffer: records arrive in 64 KiB chunks from the threads.
    setvbuf(g_file, nullptr, _IOFBF, 1u << 20);
    write_header_locked();
    // Printed on stderr so the Python harness can confirm the instrumentation
    // was active instead of assuming it from the variable it set.
    fprintf(stderr, "argine: research trace attiva: %s\n", full.c_str());
    return true;
}

struct ThreadBuf {
    std::string buf;
    uint32_t    tid = UINT32_MAX;
    ThreadBuf() { buf.reserve(kThreadBuffer + 4096); }
    ~ThreadBuf();
};

// Appends one thread's buffer to the file under the global lock.
void flush_buf(ThreadBuf & t) {
    if (t.buf.empty() || g_state.load(std::memory_order_acquire) != 1) {
        return;
    }
    {
        std::lock_guard<std::mutex> lk(g_mtx);
        if (g_file) {
            fwrite(t.buf.data(), 1, t.buf.size(), g_file);
        }
    }
    t.buf.clear();
}

ThreadBuf::~ThreadBuf() {
    flush_buf(*this);
}

ThreadBuf & tb() {
    static thread_local ThreadBuf t;
    return t;
}

void append_u64(std::string & s, uint64_t v) {
    char tmp[24];
    int  n = snprintf(tmp, sizeof(tmp), "%" PRIu64, v);
    s.append(tmp, (size_t) n);
}

void maybe_spill(ThreadBuf & t) {
    if (t.buf.size() >= kThreadBuffer) {
        flush_buf(t);
    }
}

}  // namespace

extern "C" {

int arx_enabled(void) {
    int st = g_state.load(std::memory_order_acquire);
    if (st != 0) {
        return st == 1;
    }
    std::lock_guard<std::mutex> lk(g_mtx);
    st = g_state.load(std::memory_order_relaxed);
    if (st == 0) {
        const char * p = getenv("ARGINE_RESEARCH_TRACE");
        st = open_locked(p) ? 1 : 2;
        g_state.store(st, std::memory_order_release);
    }
    return st == 1;
}

void arx_debug_open(const char * path) {
    std::lock_guard<std::mutex> lk(g_mtx);
    const bool ok = path ? open_locked(path) : (open_locked(nullptr), false);
    g_state.store(ok ? 1 : 2, std::memory_order_release);
}

uint64_t arx_now_ns(void) {
    return steady_ns();
}

uint64_t arx_next_request_id(void) {
    return g_next_req.fetch_add(1, std::memory_order_relaxed);
}

uint32_t arx_thread_id(void) {
    ThreadBuf & t = tb();
    if (t.tid == UINT32_MAX) {
        t.tid = g_next_tid.fetch_add(1, std::memory_order_relaxed);
    }
    return t.tid;
}

void arx_config(const char * key, const char * value) {
    if (!arx_enabled()) {
        return;
    }
    ThreadBuf & t = tb();
    t.buf += "C\t";
    t.buf += key ? key : "";
    t.buf += '\t';
    t.buf += value ? value : "";
    t.buf += '\n';
    maybe_spill(t);
}

void arx_route(uint64_t graph, uint32_t layer, uint32_t n_tokens, uint32_t n_used,
               const int32_t * ids) {
    if (!arx_enabled()) {
        return;
    }
    ThreadBuf & t = tb();
    std::string & s = t.buf;
    s += "R\t";
    append_u64(s, graph);
    s += '\t';
    append_u64(s, steady_ns());
    s += '\t';
    append_u64(s, layer);
    s += '\t';
    append_u64(s, n_tokens);
    s += '\t';
    append_u64(s, n_used);
    const uint64_t n = (uint64_t) n_tokens * n_used;
    for (uint64_t i = 0; i < n; i++) {
        s += '\t';
        char tmp[16];
        int  k = snprintf(tmp, sizeof(tmp), "%d", (int) ids[i]);
        s.append(tmp, (size_t) k);
    }
    s += '\n';
    maybe_spill(t);
}

void arx_access(uint32_t layer, uint32_t expert, uint32_t kind, char outcome, uint64_t bytes,
                uint64_t request_id, uint32_t shard, uint64_t seq) {
    if (!arx_enabled()) {
        return;
    }
    ThreadBuf & t = tb();
    std::string & s = t.buf;
    s += "A\t";
    append_u64(s, steady_ns());
    s += '\t';
    append_u64(s, arx_thread_id());
    s += '\t';
    append_u64(s, layer);
    s += '\t';
    append_u64(s, expert);
    s += '\t';
    append_u64(s, kind);
    s += '\t';
    s += outcome;
    s += '\t';
    append_u64(s, bytes);
    s += '\t';
    append_u64(s, request_id);
    s += '\t';
    append_u64(s, shard);
    s += '\t';
    append_u64(s, seq);
    s += '\n';
    maybe_spill(t);
}

void arx_io(uint64_t request_id, uint32_t layer, uint32_t expert, uint32_t kind, uint64_t bytes,
            uint64_t issue_ns, uint64_t complete_ns, uint64_t wait_start_ns, uint64_t wait_end_ns,
            int ok, uint32_t flags) {
    if (!arx_enabled()) {
        return;
    }
    ThreadBuf & t = tb();
    std::string & s = t.buf;
    s += "I\t";
    append_u64(s, request_id);
    s += '\t';
    append_u64(s, arx_thread_id());
    s += '\t';
    append_u64(s, layer);
    s += '\t';
    append_u64(s, expert);
    s += '\t';
    append_u64(s, kind);
    s += '\t';
    append_u64(s, bytes);
    s += '\t';
    append_u64(s, issue_ns);
    s += '\t';
    append_u64(s, complete_ns);
    s += '\t';
    append_u64(s, wait_start_ns);
    s += '\t';
    append_u64(s, wait_end_ns);
    s += '\t';
    s += ok ? '1' : '0';
    s += '\t';
    append_u64(s, flags);
    s += '\n';
    maybe_spill(t);
}

void arx_evict(uint32_t layer, uint32_t expert, uint32_t kind, uint64_t bytes, uint64_t loaded_ns,
               uint64_t last_use_ns, uint64_t uses, uint32_t shard, uint64_t seq,
               uint32_t skipped) {
    if (!arx_enabled()) {
        return;
    }
    ThreadBuf & t = tb();
    std::string & s = t.buf;
    s += "E\t";
    append_u64(s, steady_ns());
    s += '\t';
    append_u64(s, layer);
    s += '\t';
    append_u64(s, expert);
    s += '\t';
    append_u64(s, kind);
    s += '\t';
    append_u64(s, bytes);
    s += '\t';
    append_u64(s, loaded_ns);
    s += '\t';
    append_u64(s, last_use_ns);
    s += '\t';
    append_u64(s, uses);
    s += '\t';
    append_u64(s, shard);
    s += '\t';
    append_u64(s, seq);
    s += '\t';
    append_u64(s, skipped);
    s += '\n';
    maybe_spill(t);
}

void arx_undo(uint32_t layer, uint32_t expert, uint32_t kind, uint64_t bytes, uint32_t shard,
              uint64_t seq) {
    if (!arx_enabled()) {
        return;
    }
    ThreadBuf & t = tb();
    std::string & s = t.buf;
    s += "U\t";
    append_u64(s, steady_ns());
    s += '\t';
    append_u64(s, layer);
    s += '\t';
    append_u64(s, expert);
    s += '\t';
    append_u64(s, kind);
    s += '\t';
    append_u64(s, bytes);
    s += '\t';
    append_u64(s, shard);
    s += '\t';
    append_u64(s, seq);
    s += '\n';
    maybe_spill(t);
}

void arx_node(uint64_t graph, uint32_t layer, uint32_t kind, uint64_t start_ns, uint64_t ready_ns,
              uint32_t n_serve, uint32_t nth) {
    if (!arx_enabled()) {
        return;
    }
    ThreadBuf & t = tb();
    std::string & s = t.buf;
    s += "N\t";
    append_u64(s, graph);
    s += '\t';
    append_u64(s, layer);
    s += '\t';
    append_u64(s, kind);
    s += '\t';
    append_u64(s, start_ns);
    s += '\t';
    append_u64(s, ready_ns);
    s += '\t';
    append_u64(s, n_serve);
    s += '\t';
    append_u64(s, nth);
    s += '\n';
    maybe_spill(t);
}

void arx_snapshot(uint64_t graph, uint64_t resident, uint64_t budget, uint64_t hits,
                  uint64_t misses, uint64_t evictions, uint64_t bytes_read, uint64_t violations) {
    if (!arx_enabled()) {
        return;
    }
    ThreadBuf & t = tb();
    std::string & s = t.buf;
    s += "S\t";
    append_u64(s, steady_ns());
    s += '\t';
    append_u64(s, graph);
    s += '\t';
    append_u64(s, resident);
    s += '\t';
    append_u64(s, budget);
    s += '\t';
    append_u64(s, hits);
    s += '\t';
    append_u64(s, misses);
    s += '\t';
    append_u64(s, evictions);
    s += '\t';
    append_u64(s, bytes_read);
    s += '\t';
    append_u64(s, violations);
    s += '\n';
    maybe_spill(t);
}

void arx_flush_thread(void) {
    if (g_state.load(std::memory_order_acquire) != 1) {
        return;
    }
    flush_buf(tb());
}

void arx_flush_file(unsigned min_interval_ms) {
    if (g_state.load(std::memory_order_acquire) != 1) {
        return;
    }
    const uint64_t now  = steady_ns();
    const uint64_t last = g_last_flush_ns.load(std::memory_order_relaxed);
    if (now - last < (uint64_t) min_interval_ms * 1000000ull) {
        return;
    }
    g_last_flush_ns.store(now, std::memory_order_relaxed);
    std::lock_guard<std::mutex> lk(g_mtx);
    if (g_file) {
        fflush(g_file);
    }
}

}  // extern "C"
