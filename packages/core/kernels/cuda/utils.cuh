/**
 * CUDA Utility Functions
 * =======================
 * 
 * Helper functions for CUDA operations.
 */

#ifndef FORGE_CUDA_UTILS_H
#define FORGE_CUDA_UTILS_H

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>

// Error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

#define CUBLAS_CHECK(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            fprintf(stderr, "cuBLAS error at %s:%d: %d\n", \
                    __FILE__, __LINE__, status); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Device information
typedef struct {
    int device_id;
    char name[256];
    size_t total_memory;
    size_t free_memory;
    int compute_capability_major;
    int compute_capability_minor;
    int multiprocessor_count;
    int max_threads_per_block;
    int warp_size;
} DeviceInfo;

// Get device information
inline DeviceInfo get_device_info(int device_id = 0) {
    DeviceInfo info;
    info.device_id = device_id;
    
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));
    
    strncpy(info.name, prop.name, 256);
    info.total_memory = prop.totalGlobalMem;
    info.compute_capability_major = prop.major;
    info.compute_capability_minor = prop.minor;
    info.multiprocessor_count = prop.multiProcessorCount;
    info.max_threads_per_block = prop.maxThreadsPerBlock;
    info.warp_size = prop.warpSize;
    
    size_t free_mem, total_mem;
    CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
    info.free_memory = free_mem;
    
    return info;
}

// Print device info
inline void print_device_info(const DeviceInfo& info) {
    printf("GPU Device %d: %s\n", info.device_id, info.name);
    printf("  - Total Memory: %.2f GB\n", info.total_memory / 1e9);
    printf("  - Free Memory: %.2f GB\n", info.free_memory / 1e9);
    printf("  - Compute Capability: %d.%d\n", 
           info.compute_capability_major, info.compute_capability_minor);
    printf("  - Multiprocessors: %d\n", info.multiprocessor_count);
    printf("  - Max Threads/Block: %d\n", info.max_threads_per_block);
}

// Memory allocation helpers
template<typename T>
inline T* cuda_malloc(size_t count) {
    T* ptr;
    CUDA_CHECK(cudaMalloc(&ptr, count * sizeof(T)));
    return ptr;
}

template<typename T>
inline void cuda_free(T* ptr) {
    CUDA_CHECK(cudaFree(ptr));
}

template<typename T>
inline void cuda_memcpy_to_device(T* dst, const T* src, size_t count) {
    CUDA_CHECK(cudaMemcpy(dst, src, count * sizeof(T), cudaMemcpyHostToDevice));
}

template<typename T>
inline void cuda_memcpy_to_host(T* dst, const T* src, size_t count) {
    CUDA_CHECK(cudaMemcpy(dst, src, count * sizeof(T), cudaMemcpyDeviceToHost));
}

template<typename T>
inline void cuda_memset(T* ptr, int value, size_t count) {
    CUDA_CHECK(cudaMemset(ptr, value, count * sizeof(T)));
}

// cuBLAS handle management
class CublasHandle {
public:
    CublasHandle() {
        CUBLAS_CHECK(cublasCreate(&handle_));
    }
    
    ~CublasHandle() {
        cublasDestroy(handle_);
    }
    
    cublasHandle_t get() const { return handle_; }
    
    // GEMM wrapper
    void gemm(
        bool transA, bool transB,
        int m, int n, int k,
        float alpha,
        const float* A, int lda,
        const float* B, int ldb,
        float beta,
        float* C, int ldc
    ) {
        CUBLAS_CHECK(cublasSgemm(
            handle_,
            transB ? CUBLAS_OP_T : CUBLAS_OP_N,
            transA ? CUBLAS_OP_T : CUBLAS_OP_N,
            n, m, k,
            &alpha,
            B, ldb,
            A, lda,
            &beta,
            C, ldc
        ));
    }
    
private:
    cublasHandle_t handle_;
};

// Timing utilities
class CudaTimer {
public:
    CudaTimer() {
        CUDA_CHECK(cudaEventCreate(&start_));
        CUDA_CHECK(cudaEventCreate(&stop_));
    }
    
    ~CudaTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }
    
    void start() {
        CUDA_CHECK(cudaEventRecord(start_));
    }
    
    float stop() {
        CUDA_CHECK(cudaEventRecord(stop_));
        CUDA_CHECK(cudaEventSynchronize(stop_));
        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
        return ms;
    }
    
private:
    cudaEvent_t start_, stop_;
};

// Kernel launch helpers
inline int ceil_div(int a, int b) {
    return (a + b - 1) / b;
}

#define LAUNCH_KERNEL(kernel, grid, block, ...) \
    kernel<<<grid, block>>>(__VA_ARGS__); \
    CUDA_CHECK(cudaGetLastError())

#define LAUNCH_KERNEL_SHARED(kernel, grid, block, shared_mem, ...) \
    kernel<<<grid, block, shared_mem>>>(__VA_ARGS__); \
    CUDA_CHECK(cudaGetLastError())

#endif // FORGE_CUDA_UTILS_H
