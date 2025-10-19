template <typename T>
__global__ void mm_kernel(T const* mat_1,
						  T const* mat_2,
						  T* mat_3,
						  size_t m,
						  size_t n,
						  size_t p) {
	// Compute (i, j) coordinates from 2D grid
	size_t j = blockIdx.x * blockDim.x + threadIdx.x;
	size_t i = blockIdx.y * blockDim.y + threadIdx.y;

	// Boundary check
	if ((i >= m) || (j >= p))
		return;

	// Compute dot product of row i (A) and column j (B)
	T acc_sum = 0;
	for (size_t k = 0; k + 3 < n; k += 4) {
		acc_sum += mat_1[i * n + (k + 0)] * mat_2[(k + 0) * p + j];
		acc_sum += mat_1[i * n + (k + 1)] * mat_2[(k + 1) * p + j];
		acc_sum += mat_1[i * n + (k + 2)] * mat_2[(k + 2) * p + j];
		acc_sum += mat_1[i * n + (k + 3)] * mat_2[(k + 3) * p + j];
	}

	for (; k < n; k++)
		acc_sum += mat_1[i * n + k] * mat_2[k * p + j];

	mat_3[i * p + j] = acc_sum;	 // Write result
}

#include "helpers.cu"

#include "main.cu"