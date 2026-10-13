// How memory access patterns decide GPU speed.
//  1. Copy kernel with strided reads (stride 1 is coalesced, larger strides are not).
//  2. The preprocessing kernel with two thread mappings: neighbors along a row
//     (coalesced) or neighbors down a column (not coalesced). Output must match.
//  3. Block-size sweep for the coalesced version.
//
// Build: nvcc -O2 src/cuda/memory_experiment.cu -o memory_experiment
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cuda_runtime.h>

#define CHECK(call)                                                          \
    do {                                                                     \
        cudaError_t err = (call);                                            \
        if (err != cudaSuccess) {                                            \
            fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__,               \
                    cudaGetErrorString(err));                                \
            exit(1);                                                         \
        }                                                                    \
    } while (0)

// Average milliseconds per call of launch(), after a short warm-up.
template <typename F>
float time_ms(F launch, int runs) {
    cudaEvent_t s, e;
    CHECK(cudaEventCreate(&s));
    CHECK(cudaEventCreate(&e));
    for (int i = 0; i < 5; i++) launch();
    CHECK(cudaGetLastError());
    CHECK(cudaEventRecord(s));
    for (int i = 0; i < runs; i++) launch();
    CHECK(cudaEventRecord(e));
    CHECK(cudaEventSynchronize(e));
    float ms = 0.0f;
    CHECK(cudaEventElapsedTime(&ms, s, e));
    CHECK(cudaGetLastError());
    cudaEventDestroy(s);
    cudaEventDestroy(e);
    return ms / runs;
}

// ---------------- Experiment 1: strided reads ----------------
// Thread i reads element (i * stride) and writes element i. n is a power of two.
__global__ void strided_copy(const float* in, float* out, size_t n, size_t stride, size_t mask) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = in[(i * stride) & mask];
}

// ---------------- Experiment 2: preprocessing with two thread mappings ----------------
// SWAP = false: threadIdx.x walks along a row of the output (neighbors write neighbors).
// SWAP = true:  threadIdx.x walks down a column (neighbors write far apart).
template <bool SWAP>
__global__ void preprocess(const unsigned char* in, float* out,
                           int in_h, int in_w, int out_h, int out_w) {
    int a = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.y * blockDim.y + threadIdx.y;
    int x = SWAP ? b : a;
    int y = SWAP ? a : b;
    if (x >= out_w || y >= out_h) return;

    float sx = (x + 0.5f) * in_w / out_w - 0.5f;
    float sy = (y + 0.5f) * in_h / out_h - 0.5f;
    sx = fminf(fmaxf(sx, 0.0f), in_w - 1.0f);
    sy = fminf(fmaxf(sy, 0.0f), in_h - 1.0f);
    int x0 = (int)floorf(sx), y0 = (int)floorf(sy);
    int x1 = min(x0 + 1, in_w - 1), y1 = min(y0 + 1, in_h - 1);
    float fx = sx - x0, fy = sy - y0;

    for (int c = 0; c < 3; c++) {
        float p00 = in[((size_t)y0 * in_w + x0) * 3 + c];
        float p01 = in[((size_t)y0 * in_w + x1) * 3 + c];
        float p10 = in[((size_t)y1 * in_w + x0) * 3 + c];
        float p11 = in[((size_t)y1 * in_w + x1) * 3 + c];
        float top = p00 + (p01 - p00) * fx;
        float bot = p10 + (p11 - p10) * fx;
        float v = top + (bot - top) * fy;
        out[(size_t)(2 - c) * out_h * out_w + (size_t)y * out_w + x] = v / 255.0f;
    }
}

struct Case {
    const char* name;
    int in_h, in_w, out_h, out_w;
    int runs;
};

void run_case(const Case& cs, int bx, int by) {
    size_t in_bytes = (size_t)cs.in_h * cs.in_w * 3;
    size_t out_count = (size_t)3 * cs.out_h * cs.out_w;
    double out_gb = out_count * sizeof(float) / 1e9;

    std::vector<unsigned char> frame(in_bytes);
    srand(0);
    for (auto& v : frame) v = rand() % 256;
    unsigned char* d_in;
    float *d_a, *d_b;
    CHECK(cudaMalloc(&d_in, in_bytes));
    CHECK(cudaMalloc(&d_a, out_count * sizeof(float)));
    CHECK(cudaMalloc(&d_b, out_count * sizeof(float)));
    CHECK(cudaMemcpy(d_in, frame.data(), in_bytes, cudaMemcpyHostToDevice));

    dim3 block(bx, by);
    dim3 grid_row((cs.out_w + bx - 1) / bx, (cs.out_h + by - 1) / by);
    dim3 grid_col((cs.out_h + bx - 1) / bx, (cs.out_w + by - 1) / by);

    float ms_row = time_ms([&] {
        preprocess<false><<<grid_row, block>>>(d_in, d_a, cs.in_h, cs.in_w, cs.out_h, cs.out_w);
    }, cs.runs);
    float ms_col = time_ms([&] {
        preprocess<true><<<grid_col, block>>>(d_in, d_b, cs.in_h, cs.in_w, cs.out_h, cs.out_w);
    }, cs.runs);

    std::vector<float> ha(out_count), hb(out_count);
    CHECK(cudaMemcpy(ha.data(), d_a, out_count * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(hb.data(), d_b, out_count * sizeof(float), cudaMemcpyDeviceToHost));
    float max_diff = 0.0f;
    for (size_t i = 0; i < out_count; i++) max_diff = fmaxf(max_diff, fabsf(ha[i] - hb[i]));

    printf("%-34s row-wise %8.4f ms (%6.1f GB/s out)   column-wise %8.4f ms (%6.1f GB/s out)   slowdown %.2fx   diff %g\n",
           cs.name, ms_row, out_gb / (ms_row * 1e-3), ms_col, out_gb / (ms_col * 1e-3),
           ms_col / ms_row, max_diff);

    cudaFree(d_in);
    cudaFree(d_a);
    cudaFree(d_b);
}

void block_sweep(const Case& cs) {
    size_t in_bytes = (size_t)cs.in_h * cs.in_w * 3;
    size_t out_count = (size_t)3 * cs.out_h * cs.out_w;
    std::vector<unsigned char> frame(in_bytes);
    srand(0);
    for (auto& v : frame) v = rand() % 256;
    unsigned char* d_in;
    float* d_out;
    CHECK(cudaMalloc(&d_in, in_bytes));
    CHECK(cudaMalloc(&d_out, out_count * sizeof(float)));
    CHECK(cudaMemcpy(d_in, frame.data(), in_bytes, cudaMemcpyHostToDevice));

    const int shapes[][2] = {{8, 8}, {16, 16}, {32, 8}, {32, 32}, {8, 32}};
    for (auto& sh : shapes) {
        dim3 block(sh[0], sh[1]);
        dim3 grid((cs.out_w + sh[0] - 1) / sh[0], (cs.out_h + sh[1] - 1) / sh[1]);
        float ms = time_ms([&] {
            preprocess<false><<<grid, block>>>(d_in, d_out, cs.in_h, cs.in_w, cs.out_h, cs.out_w);
        }, cs.runs);
        printf("  block %2d x %2d (%4d threads): %8.4f ms\n", sh[0], sh[1], sh[0] * sh[1], ms);
    }
    cudaFree(d_in);
    cudaFree(d_out);
}

int main() {
    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, 0));
    int clk_khz = 0, bus_bits = 0;
    CHECK(cudaDeviceGetAttribute(&clk_khz, cudaDevAttrMemoryClockRate, 0));
    CHECK(cudaDeviceGetAttribute(&bus_bits, cudaDevAttrGlobalMemoryBusWidth, 0));
    double peak = 2.0 * clk_khz * 1e3 * (bus_bits / 8.0) / 1e9;  // DDR: two transfers per clock
    printf("GPU: %s, memory bus %d bits, theoretical peak ~%.0f GB/s\n\n", prop.name, bus_bits, peak);

    // ---- Experiment 1 ----
    printf("Experiment 1: copy kernel, thread i reads element i*stride (128 MB array)\n");
    const size_t n = (size_t)1 << 25;
    float *d_in, *d_out;
    CHECK(cudaMalloc(&d_in, n * sizeof(float)));
    CHECK(cudaMalloc(&d_out, n * sizeof(float)));
    CHECK(cudaMemset(d_in, 0, n * sizeof(float)));
    CHECK(cudaMemset(d_out, 0, n * sizeof(float)));
    dim3 block(256), grid((unsigned)((n + 255) / 256));
    float base_ms = 0.0f;
    for (size_t stride : {1, 2, 4, 8, 16, 32}) {
        float ms = time_ms([&] {
            strided_copy<<<grid, block>>>(d_in, d_out, n, stride, n - 1);
        }, 50);
        if (stride == 1) base_ms = ms;
        double gbs = 2.0 * n * sizeof(float) / (ms * 1e-3) / 1e9;  // 4 B read + 4 B written per element
        printf("  stride %2zu: %8.4f ms  %6.1f GB/s useful  (%.1f%% of peak)  %.2fx slower than stride 1\n",
               stride, ms, gbs, 100.0 * gbs / peak, ms / base_ms);
    }
    cudaFree(d_in);
    cudaFree(d_out);

    // ---- Experiment 2 ----
    printf("\nExperiment 2: preprocessing kernel, 16x16 blocks, two thread mappings\n");
    run_case({"webcam 640x480 -> 224x224", 480, 640, 224, 224, 500}, 16, 16);
    run_case({"large 3840x2160 -> 1920x1080", 2160, 3840, 1080, 1920, 50}, 16, 16);

    // ---- Experiment 3 ----
    printf("\nExperiment 3: block-size sweep, row-wise mapping, large frame\n");
    block_sweep({"large", 2160, 3840, 1080, 1920, 50});
    return 0;
}
