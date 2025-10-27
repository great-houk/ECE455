#ifndef HELPERS
#define HELPERS

#include <iostream>
#include <cuda_runtime.h>
#include <vector>
#include <random>

// Macro wrapper: converts the CUDA call `val` into a string (#val)
// and passes it with file and line info to `check`
#define checkCuda(val) check((val), #val, __FILE__, __LINE__)

// Checks the return code of CUDA API calls
void check(cudaError_t err,
		   const char* const func,
		   const char* const file,
		   const int line) {
	if (err != cudaSuccess) {  // If CUDA call failed
		std::cerr << "CUDA Runtime Error at: " << file << " : " << line
				  << std::endl;
		std::cerr << cudaGetErrorString(err) << " " << func
				  << std::endl;	  // Print readable error
		std::exit(EXIT_FAILURE);  // Abort program
	}
}

// Random Initialization
// Create a vector filled with random numbers in [-256, 256]
template <typename T>
std::vector<T> create_rand_vector(size_t n, T min_val = -256, T max_val = 256) {
	std::random_device r;				// Non-deterministic seed
	std::default_random_engine e(r());	// Random engine
	std::uniform_real_distribution<double> uniform_dist(min_val, max_val);
	std::vector<T> vec(n);
	for (size_t i = 0; i < n; ++i) {
		vec[i] = static_cast<T>(uniform_dist(e));  // Fill each element
	}
	return vec;
}

// Validation (allclose)
// Compare two vectors elementwise within a tolerance
template <typename T>
bool allclose(std::vector<T> const& vec_1,
			  std::vector<T> const& vec_2,
			  T const& abs_tol) {
	if (vec_1.size() != vec_2.size())
		return false;  // Size mismatch
	for (size_t i = 0; i < vec_1.size(); ++i) {
		// Check absolute difference
		if (std::abs(vec_1.at(i) - vec_2.at(i)) > abs_tol) {
			std::cout << vec_1.at(i) << " " << vec_2.at(i) << std::endl;
			return false;  // First mismatch
		}
	}
	return true;  // All elements close
}

#endif