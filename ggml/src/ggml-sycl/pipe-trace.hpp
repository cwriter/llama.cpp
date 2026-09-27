#ifndef GGML_SYCL_PIPE_TRACE_HPP
#define GGML_SYCL_PIPE_TRACE_HPP

#include <sycl/sycl.hpp>

// Timeline of host calls and queue progress, for finding what serializes pipeline mode.
// GGML_SYCL_PIPE_TRACE=<file> turns it on. Off, every call is one branch.

bool ggml_sycl_pipe_trace_on();

// microseconds since the trace started
double ggml_sycl_pipe_trace_now();

// a host span [t0, now) of the given kind on a device
void ggml_sycl_pipe_trace_host(const char * kind, int device, double t0, const char * detail = "", size_t bytes = 0);

// a barrier on q; its completion time is logged as the device reaching this point
void ggml_sycl_pipe_trace_mark(sycl::queue & q, int device, const char * kind, int seq);

#include <string>
std::string ggml_sycl_pipe_trace_bt(int skip);

struct ggml_sycl_pipe_trace_scope {
    const char * kind;
    int          device;
    const char * detail;
    size_t       bytes;
    double       t0;

    ggml_sycl_pipe_trace_scope(const char * kind, int device, const char * detail = "", size_t bytes = 0) :
        kind(kind), device(device), detail(detail), bytes(bytes),
        t0(ggml_sycl_pipe_trace_on() ? ggml_sycl_pipe_trace_now() : -1.0) {}

    ~ggml_sycl_pipe_trace_scope() {
        if (t0 >= 0.0) {
            ggml_sycl_pipe_trace_host(kind, device, t0, detail, bytes);
        }
    }
};

#endif  // GGML_SYCL_PIPE_TRACE_HPP
