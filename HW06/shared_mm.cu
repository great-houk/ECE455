#include "helpers_mm.cuh"

// ---------- GPU Shared Memory Kernel (Tiled) ----------
template <typename T>
__global__ void mm_tiled(const T* A, const T* B, T* C, int N) {
	// Shared-memory tiles for A and B
	__shared__ T tile_A[TILE_SIZE][TILE_SIZE];
	__shared__ T tile_B[TILE_SIZE][TILE_SIZE];

	// Global row/col this thread computes
	int row = blockIdx.y * TILE_SIZE + threadIdx.y;
	int col = blockIdx.x * TILE_SIZE + threadIdx.x;

	T val = 0;

	// Loop over all sub-tiles of A and B needed to compute this C element
	for (int t = 0; t < (N + TILE_SIZE - 1) / TILE_SIZE; ++t) {
		// Load a tile of A into shared memory
		if (row < N && (t * TILE_SIZE + threadIdx.x) < N)
			tile_A[threadIdx.y][threadIdx.x] =
				A[row * N + t * TILE_SIZE + threadIdx.x];
		else
			tile_A[threadIdx.y][threadIdx.x] = 0;

		// Load a tile of B into shared memory
		if (col < N && (t * TILE_SIZE + threadIdx.y) < N)
			tile_B[threadIdx.y][threadIdx.x] =
				B[(t * TILE_SIZE + threadIdx.y) * N + col];
		else
			tile_B[threadIdx.y][threadIdx.x] = 0;

		__syncthreads();  // Wait until all data is loaded

		// Compute partial products
		for (int k = 0; k < TILE_SIZE; ++k)
			val += tile_A[threadIdx.y][k] * tile_B[k][threadIdx.x];

		__syncthreads();  // Wait before loading next tiles
	}

	// Store the result
	if (row < N && col < N)
		C[row * N + col] = val;
}

// ---------- GPU Launcher ----------
template <typename T>
void mm_cuda(const T* d_A, const T* d_B, T* d_C, int N) {
	dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
	dim3 blocksPerGrid((N + TILE_SIZE - 1) / TILE_SIZE,
					   (N + TILE_SIZE - 1) / TILE_SIZE);
	mm_tiled<T><<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
}

// ---------- CUDA Event Timing Wrapper ----------
template <typename T>
float measure_latency_mm_cuda(int N, int num_tests, int num_warmups) {
	return measure_kernel_latency<T>(mm_cuda<T>, N, num_tests, num_warmups);
}

// ---------- Main ----------
int main() {
	const int N = MAT_DIM;
	std::vector<float> h_A = create_rand_vector<float>(N * N);
	std::vector<float> h_B = create_rand_vector<float>(N * N);
	std::vector<float> h_C_ref(N * N, 0);
	std::vector<float> h_C_gpu(N * N, 0);

	std::cout << "Running CPU reference..." << std::endl;
	mm_host(h_A.data(), h_B.data(), h_C_ref.data(), N);

	float *d_A, *d_B, *d_C;
	checkCuda(cudaMalloc(&d_A, sizeof(float) * N * N));
	checkCuda(cudaMalloc(&d_B, sizeof(float) * N * N));
	checkCuda(cudaMalloc(&d_C, sizeof(float) * N * N));

	checkCuda(cudaMemcpy(d_A, h_A.data(), sizeof(float) * N * N,
						 cudaMemcpyHostToDevice));
	checkCuda(cudaMemcpy(d_B, h_B.data(), sizeof(float) * N * N,
						 cudaMemcpyHostToDevice));

	// ---- Run tiled kernel ----
	mm_cuda<float>(d_A, d_B, d_C, N);
	checkCuda(cudaDeviceSynchronize());
	checkCuda(cudaMemcpy(h_C_gpu.data(), d_C, sizeof(float) * N * N,
						 cudaMemcpyDeviceToHost));

	// ---- Validate ----
	std::cout << "Validating results..." << std::endl;
	bool ok = validate_results(h_C_ref, h_C_gpu, N);
	if (!ok) {
		std::cerr << "Validation FAILED." << std::endl;
		return 1;
	}
	std::cout << "Validation PASSED." << std::endl;

	// ---- Measure average runtime ----
	const int num_measurement_tests = 3;
	const int num_measurement_warmups = 1;
	float avg_ms = measure_latency_mm_cuda<float>(N, num_measurement_tests,
												  num_measurement_warmups);
	std::cout << "Average kernel runtime: " << std::fixed
			  << std::setprecision(4) << avg_ms << " ms" << std::endl;

	checkCuda(cudaFree(d_A));
	checkCuda(cudaFree(d_B));
	checkCuda(cudaFree(d_C));
	return 0;
}