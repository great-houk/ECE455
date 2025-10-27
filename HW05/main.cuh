#include <iostream>
#include <cuda_runtime.h>
#include <cassert>
#include "helpers.cuh"

extern size_t MAT_DIM;

int main() {
	const size_t num_tests = 2;	 // Correctness trials
	assert(random_multiple_test_mm_cuda<int32_t>(num_tests));
	assert(random_multiple_test_mm_cuda<float>(num_tests));
	assert(random_multiple_test_mm_cuda<double>(num_tests));
	std::cout << "All tests passed!\n";

	// --- Performance measurement ---
	const size_t num_measurement_tests = 100;
	const size_t num_measurement_warmups = 10;
	size_t m = MAT_DIM, n = MAT_DIM, p = MAT_DIM;

	// Measure average latency across data types
	float mm_cuda_int32_latency = measure_latency_mm_cuda<int32_t>(
		m, n, p, num_measurement_tests, num_measurement_warmups);
	float mm_cuda_float_latency = measure_latency_mm_cuda<float>(
		m, n, p, num_measurement_tests, num_measurement_warmups);
	float mm_cuda_double_latency = measure_latency_mm_cuda<double>(
		m, n, p, num_measurement_tests, num_measurement_warmups);

	// Print results
	std::cout << "Matrix Multiplication Runtime\n";
	std::cout << "m: " << m << " n: " << n << " p: " << p << "\n";
	std::cout << "INT32: " << mm_cuda_int32_latency << " ms\n";
	std::cout << "FLOAT: " << mm_cuda_float_latency << " ms\n";
	std::cout << "DOUBLE: " << mm_cuda_double_latency << " ms\n";

	return 0;
}