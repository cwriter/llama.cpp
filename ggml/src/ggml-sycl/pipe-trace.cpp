#include "pipe-trace.hpp"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <mutex>
#include <string>
#include <thread>

#include <dlfcn.h>
#include <execinfo.h>

namespace {

struct pending_mark {
    sycl::event ev;
    int         device;
    const char * kind;
    int         seq;
    double      t_submit;
};

struct pipe_trace {
    FILE *                                f = nullptr;
    std::chrono::steady_clock::time_point t_start;
    std::mutex                            mtx;
    std::deque<pending_mark>              marks[16];
    std::thread                           poller;
    std::atomic<bool>                     stop{ false };

    pipe_trace() {
        const char * path = getenv("GGML_SYCL_PIPE_TRACE");
        if (path == nullptr || path[0] == '\0') {
            return;
        }
        f = fopen(path, "w");
        if (f == nullptr) {
            return;
        }
        t_start = std::chrono::steady_clock::now();
        fprintf(f, "# kind dev t0_us t1_us detail bytes\n");
        poller = std::thread([this] { poll(); });
    }

    ~pipe_trace() {
        if (f == nullptr) {
            return;
        }
        stop = true;
        if (poller.joinable()) {
            poller.join();
        }
        fclose(f);
    }

    double now() const {
        return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t_start).count();
    }

    // queues are in order, so only the oldest mark of each device can be the next to finish
    void poll() {
        while (!stop) {
            bool any = false;
            for (int d = 0; d < 16; ++d) {
                for (;;) {
                    pending_mark m;
                    {
                        std::lock_guard<std::mutex> lock(mtx);
                        if (marks[d].empty()) {
                            break;
                        }
                        m = marks[d].front();
                    }
                    any = true;
                    bool done = false;
                    try {
                        done = m.ev.get_info<sycl::info::event::command_execution_status>() ==
                               sycl::info::event_command_status::complete;
                    } catch (...) {
                        done = true;
                    }
                    if (!done) {
                        break;
                    }
                    const double t = now();
                    std::lock_guard<std::mutex> lock(mtx);
                    marks[d].pop_front();
                    fprintf(f, "DEV %d %.1f %.1f %s#%d 0\n", d, m.t_submit, t, m.kind, m.seq);
                }
            }
            std::this_thread::sleep_for(std::chrono::microseconds(any ? 20 : 200));
        }
    }
};

pipe_trace & trace() {
    static pipe_trace t;
    return t;
}

}  // namespace

bool ggml_sycl_pipe_trace_on() {
    return trace().f != nullptr;
}

double ggml_sycl_pipe_trace_now() {
    return trace().now();
}

void ggml_sycl_pipe_trace_host(const char * kind, int device, double t0, const char * detail, size_t bytes) {
    pipe_trace & t = trace();
    if (t.f == nullptr) {
        return;
    }
    const double                t1 = t.now();
    std::lock_guard<std::mutex> lock(t.mtx);
    fprintf(t.f, "%s %d %.1f %.1f %s %zu\n", kind, device, t0, t1, detail && detail[0] ? detail : "-", bytes);
}

void ggml_sycl_pipe_trace_mark(sycl::queue & q, int device, const char * kind, int seq) {
    pipe_trace & t = trace();
    if (t.f == nullptr || device < 0 || device >= 16) {
        return;
    }
    pending_mark m{ q.ext_oneapi_submit_barrier(), device, kind, seq, t.now() };
    std::lock_guard<std::mutex> lock(t.mtx);
    t.marks[device].push_back(m);
}

// "lib+0xoff|..." for the callers, resolved later with addr2line
std::string ggml_sycl_pipe_trace_bt(int skip) {
    void * frames[24];
    const int n = backtrace(frames, 24);
    std::string out;
    for (int i = skip; i < n; ++i) {
        Dl_info info;
        char    buf[256];
        if (dladdr(frames[i], &info) && info.dli_fname) {
            const char * base = strrchr(info.dli_fname, '/');
            snprintf(buf, sizeof(buf), "%s+0x%lx|", base ? base + 1 : info.dli_fname,
                     (unsigned long) ((char *) frames[i] - (char *) info.dli_fbase));
        } else {
            snprintf(buf, sizeof(buf), "%p|", frames[i]);
        }
        out += buf;
    }
    return out;
}
