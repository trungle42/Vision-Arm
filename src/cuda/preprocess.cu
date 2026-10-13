// Fused webcam preprocessing on the GPU.
// Input:  one BGR frame, uint8, H x W x 3 (what OpenCV gives us)
// Output: RGB float32, 3 x 224 x 224, values 0..1 (what the model expects)
// One kernel does BGR->RGB, bilinear resize, scale by 1/255, and HWC->CHW.
//
// Build: nvcc -O2 src/cuda/preprocess.cu -o preprocess
#include <algorithm>
#include <chrono>
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

// One thread computes one output pixel (all 3 channels).
__global__ void preprocess_kernel(const unsigned char* in, float* out,
                                  int in_h, int in_w, int out_h, int out_w) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= out_w || y >= out_h) return;

    // Where this output pixel lands in the input image (pixel centers aligned).
    float sx = (x + 0.5f) * in_w / out_w - 0.5f;
    float sy = (y + 0.5f) * in_h / out_h - 0.5f;
    sx = fminf(fmaxf(sx, 0.0f), in_w - 1.0f);
    sy = fminf(fmaxf(sy, 0.0f), in_h - 1.0f);

    int x0 = (int)floorf(sx), y0 = (int)floorf(sy);
    int x1 = min(x0 + 1, in_w - 1), y1 = min(y0 + 1, in_h - 1);
    float fx = sx - x0, fy = sy - y0;

    for (int c = 0; c < 3; c++) {
        float p00 = in[(y0 * in_w + x0) * 3 + c];
        float p01 = in[(y0 * in_w + x1) * 3 + c];
        float p10 = in[(y1 * in_w + x0) * 3 + c];
        float p11 = in[(y1 * in_w + x1) * 3 + c];
        float top = p00 + (p01 - p00) * fx;
        float bot = p10 + (p11 - p10) * fx;
        float v = top + (bot - top) * fy;
        int out_c = 2 - c;  // input channel 0 is blue, output channel 0 is red
        out[out_c * out_h * out_w + y * out_w + x] = v / 255.0f;
    }
}

// Same math on the CPU, used to check the GPU result.
void preprocess_cpu(const unsigned char* in, float* out,
                    int in_h, int in_w, int out_h, int out_w) {
    for (int y = 0; y < out_h; y++) {
        for (int x = 0; x < out_w; x++) {
            float sx = (x + 0.5f) * in_w / out_w - 0.5f;
            float sy = (y + 0.5f) * in_h / out_h - 0.5f;
            sx = fminf(fmaxf(sx, 0.0f), in_w - 1.0f);
            sy = fminf(fmaxf(sy, 0.0f), in_h - 1.0f);
            int x0 = (int)floorf(sx), y0 = (int)floorf(sy);
            int x1 = std::min(x0 + 1, in_w - 1), y1 = std::min(y0 + 1, in_h - 1);
            float fx = sx - x0, fy = sy - y0;
            for (int c = 0; c < 3; c++) {
                float p00 = in[(y0 * in_w + x0) * 3 + c];
                float p01 = in[(y0 * in_w + x1) * 3 + c];
                float p10 = in[(y1 * in_w + x0) * 3 + c];
                float p11 = in[(y1 * in_w + x1) * 3 + c];
                float top = p00 + (p01 - p00) * fx;
                float bot = p10 + (p11 - p10) * fx;
                float v = top + (bot - top) * fy;
                out[(2 - c) * out_h * out_w + y * out_w + x] = v / 255.0f;
            }
        }
    }
}

int main() {
    const int in_h = 480, in_w = 640, out_h = 224, out_w = 224;
    const size_t in_bytes = (size_t)in_h * in_w * 3;
    const size_t out_count = (size_t)3 * out_h * out_w;

    // Fake frame. Timing does not depend on the content.
    std::vector<unsigned char> frame(in_bytes);
    srand(0);
    for (auto& b : frame) b = rand() % 256;

    unsigned char* d_in;
    float* d_out;
    CHECK(cudaMalloc(&d_in, in_bytes));
    CHECK(cudaMalloc(&d_out, out_count * sizeof(float)));

    dim3 block(16, 16);
    dim3 grid((out_w + block.x - 1) / block.x, (out_h + block.y - 1) / block.y);

    // Correctness: GPU result vs CPU result.
    CHECK(cudaMemcpy(d_in, frame.data(), in_bytes, cudaMemcpyHostToDevice));
    preprocess_kernel<<<grid, block>>>(d_in, d_out, in_h, in_w, out_h, out_w);
    CHECK(cudaGetLastError());
    std::vector<float> gpu_out(out_count), cpu_out(out_count);
    CHECK(cudaMemcpy(gpu_out.data(), d_out, out_count * sizeof(float),
                     cudaMemcpyDeviceToHost));
    preprocess_cpu(frame.data(), cpu_out.data(), in_h, in_w, out_h, out_w);
    float max_diff = 0.0f;
    for (size_t i = 0; i < out_count; i++)
        max_diff = fmaxf(max_diff, fabsf(gpu_out[i] - cpu_out[i]));
    printf("max abs difference GPU vs CPU: %g\n", max_diff);

    // Timing with CUDA events. Warm up first, then average over many runs.
    const int runs = 200;
    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));
    for (int i = 0; i < 20; i++)
        preprocess_kernel<<<grid, block>>>(d_in, d_out, in_h, in_w, out_h, out_w);

    CHECK(cudaEventRecord(start));
    for (int i = 0; i < runs; i++)
        preprocess_kernel<<<grid, block>>>(d_in, d_out, in_h, in_w, out_h, out_w);
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));
    float ms = 0.0f;
    CHECK(cudaEventElapsedTime(&ms, start, stop));
    printf("GPU kernel only:         %.4f ms per frame\n", ms / runs);

    // Same, but including the host-to-device copy of the frame, as in a real loop.
    CHECK(cudaEventRecord(start));
    for (int i = 0; i < runs; i++) {
        CHECK(cudaMemcpy(d_in, frame.data(), in_bytes, cudaMemcpyHostToDevice));
        preprocess_kernel<<<grid, block>>>(d_in, d_out, in_h, in_w, out_h, out_w);
    }
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));
    CHECK(cudaEventElapsedTime(&ms, start, stop));
    printf("GPU copy + kernel:       %.4f ms per frame\n", ms / runs);

    // CPU baseline (single thread, same math).
    const int cpu_runs = 20;
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < cpu_runs; i++)
        preprocess_cpu(frame.data(), cpu_out.data(), in_h, in_w, out_h, out_w);
    auto t1 = std::chrono::steady_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    printf("CPU single thread:       %.4f ms per frame\n", cpu_ms / cpu_runs);

    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}
