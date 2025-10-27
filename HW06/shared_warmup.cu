#include <iostream>
#include <cuda_runtime.h>
#include "helpers.cuh"

const size_t BLOCK_DIM = 128;
const size_t N = (1 << 16);

template <typename T>
__global__ void square_shared_kernel(const T* in, T* out, size_t N) {
	__shared__ T tile[BLOCK_DIM];

	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= N)
		return;

	// Load
	tile[threadIdx.x] = in[idx];
	// __syncthreads();

	// Compute
	tile[threadIdx.x] = tile[threadIdx.x] * tile[threadIdx.x];
	// __syncthreads();

	// Write back
	out[idx] = tile[threadIdx.x];
}

int main() {
	using T = float;
	std::vector<T> h_in = create_rand_vector<T>(N);
	std::vector<T> h_out(N);

	T *d_in, *d_out;
	checkCuda(cudaMalloc(&d_in, sizeof(T) * N));
	checkCuda(cudaMalloc(&d_out, sizeof(T) * N));
	checkCuda(
		cudaMemcpy(d_in, h_in.data(), sizeof(T) * N, cudaMemcpyHostToDevice));

	dim3 threadsPerBlock(BLOCK_DIM);
	dim3 blocksPerGrid((N + BLOCK_DIM - 1) / BLOCK_DIM);
	square_shared_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_in, d_out, N);

	checkCuda(
		cudaMemcpy(h_out.data(), d_out, sizeof(T) * N, cudaMemcpyDeviceToHost));
	checkCuda(cudaFree(d_in));
	checkCuda(cudaFree(d_out));

	// Verify output
	for (size_t i = 0; i < 5; ++i)
		std::cout << "in[" << i << "] = " << h_in[i]
				  << " ^ 2 == " << (h_in[i] * h_in[i]) << ", out[" << i
				  << "] = " << h_out[i] << std::endl;

	std::cout << "\nSuccess! Computed squares using shared memory.\n";
	return 0;
}