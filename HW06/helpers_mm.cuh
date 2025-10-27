#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>
#include <cmath>

// Common constants
#define TILE_SIZE 16
#define MAT_DIM 1024

// CUDA error checking
#define checkCuda(val) check((val), #val, __FILE__, __LINE__)

inline void check(cudaError_t err,
				  const char* const func,
				  const char* const file,
				  int line) {
	if (err != cudaSuccess) {
		std::cerr << "CUDA Runtime Error at: " << file << ":" << line
				  << std::endl;
		std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
		std::exit(EXIT_FAILURE);
	}
}

// Random initialization
template <typename T>
inline std::vector<T> create_rand_vector(size_t n,
										 T min_val = 0,
										 T max_val = 10) {
	std::random_device r;
	std::default_random_engine e(r());
	std::uniform_real_distribution<double> dist(min_val, max_val);
	std::vector<T> vec(n);
	for (size_t i = 0; i < n; ++i)
		vec[i] = static_cast<T>(dist(e));
	return vec;
}

// CPU reference implementation (double accumulator)
template <typename T>
inline void mm_host(const T* A, const T* B, T* C, int N) {
	for (int i = 0; i < N; ++i) {
		for (int j = 0; j < N; ++j) {
			double acc = 0.0;
			for (int k = 0; k < N; ++k)
				acc += static_cast<double>(A[i * N + k]) *
					   static_cast<double>(B[k * N + j]);
			C[i * N + j] = static_cast<T>(acc);
		}
	}
}

// Validation (Relative Error)
template <typename T>
inline bool validate_results(const std::vector<T>& ref,
							 const std::vector<T>& gpu,
							 int N,
							 T rel_tol = static_cast<T>(1e-2)) {
	for (int i = 0; i < N; ++i) {
		for (int j = 0; j < N; ++j) {
			size_t idx = i * N + j;
			T diff = std::abs(ref[idx] - gpu[idx]);
			T denom = std::max(static_cast<T>(1.0), std::abs(ref[idx]));
			if (diff / denom > rel_tol) {
				std::cerr << "Mismatch at (" << i << ", " << j << "): "
						  << "CPU=" << ref[idx] << ", GPU=" << gpu[idx]
						  << ", rel_err=" << diff / denom << std::endl;
				return false;
			}
		}
	}
	return true;
}

// Generic CUDA event timing for any kernel
template <typename T>
inline float measure_kernel_latency(
	void (*kernel_launcher)(const T*, const T*, T*, int),
	int N,
	int num_tests = 1,
	int num_warmups = 1) {
	cudaEvent_t startEvent, stopEvent;
	float time_ms{0.0f};

	checkCuda(cudaEventCreate(&startEvent));
	checkCuda(cudaEventCreate(&stopEvent));

	T *d_A, *d_B, *d_C;
	checkCuda(cudaMalloc(&d_A, sizeof(T) * N * N));
	checkCuda(cudaMalloc(&d_B, sizeof(T) * N * N));
	checkCuda(cudaMalloc(&d_C, sizeof(T) * N * N));

	for (int i = 0; i < num_warmups; ++i)
		kernel_launcher(d_A, d_B, d_C, N);

	checkCuda(cudaEventRecord(startEvent, 0));
	for (int i = 0; i < num_tests; ++i)
		kernel_launcher(d_A, d_B, d_C, N);
	checkCuda(cudaEventRecord(stopEvent, 0));
	checkCuda(cudaEventSynchronize(stopEvent));
	checkCuda(cudaEventElapsedTime(&time_ms, startEvent, stopEvent));

	checkCuda(cudaFree(d_A));
	checkCuda(cudaFree(d_B));
	checkCuda(cudaFree(d_C));

	return time_ms / num_tests;
}