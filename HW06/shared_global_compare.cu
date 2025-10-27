#include "helpers_mm.cuh"

// ---------------- Naive Global Memory Kernel (1D threads) ----------------
template <typename T>
__global__ void mm_naive(const T* A, const T* B, T* C, int N) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int total_elems = N * N;
	if (tid >= total_elems)
		return;

	int row = tid / N;
	int col = tid % N;

	T val = 0;
	for (int k = 0; k < N; ++k)
		val += A[row * N + k] * B[k * N + col];

	C[tid] = val;
}

// ---------------- Tiled Shared Memory Kernel ----------------
template <typename T>
__global__ void mm_tiled(const T* A, const T* B, T* C, int N) {
	__shared__ T tile_A[TILE_SIZE][TILE_SIZE];
	__shared__ T tile_B[TILE_SIZE][TILE_SIZE];

	int row = blockIdx.y * TILE_SIZE + threadIdx.y;
	int col = blockIdx.x * TILE_SIZE + threadIdx.x;

	T val = 0;

	for (int t = 0; t < (N + TILE_SIZE - 1) / TILE_SIZE; ++t) {
		if (row < N && (t * TILE_SIZE + threadIdx.x) < N)
			tile_A[threadIdx.y][threadIdx.x] =
				A[row * N + t * TILE_SIZE + threadIdx.x];
		else
			tile_A[threadIdx.y][threadIdx.x] = 0;

		if (col < N && (t * TILE_SIZE + threadIdx.y) < N)
			tile_B[threadIdx.y][threadIdx.x] =
				B[(t * TILE_SIZE + threadIdx.y) * N + col];
		else
			tile_B[threadIdx.y][threadIdx.x] = 0;

		__syncthreads();

		for (int k = 0; k < TILE_SIZE; ++k)
			val += tile_A[threadIdx.y][k] * tile_B[k][threadIdx.x];

		__syncthreads();
	}

	if (row < N && col < N)
		C[row * N + col] = val;
}

// ---------------- GPU Launchers ----------------
template <typename T>
void launch_naive(const T* d_A, const T* d_B, T* d_C, int N) {
	int threadsPerBlock = 256;
	int totalThreads = N * N;
	dim3 blocks((totalThreads + threadsPerBlock - 1) / threadsPerBlock);
	dim3 threads(threadsPerBlock);
	mm_naive<T><<<blocks, threads>>>(d_A, d_B, d_C, N);
}

template <typename T>
void launch_tiled(const T* d_A, const T* d_B, T* d_C, int N) {
	dim3 threads(TILE_SIZE, TILE_SIZE);
	dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE,
				(N + TILE_SIZE - 1) / TILE_SIZE);
	mm_tiled<T><<<blocks, threads>>>(d_A, d_B, d_C, N);
}

// ---------------- Main ----------------
int main() {
	const int N = MAT_DIM;
	std::vector<float> h_A = create_rand_vector<float>(N * N);
	std::vector<float> h_B = create_rand_vector<float>(N * N);
	std::vector<float> h_C_ref(N * N, 0);
	std::vector<float> h_C_gpu_tiled(N * N, 0);
	std::vector<float> h_C_gpu_naive(N * N, 0);

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

	// ---- Run Naive Kernel ----
	{
		int threadsPerBlock = 256;
		int totalThreads = N * N;
		dim3 blocks((totalThreads + threadsPerBlock - 1) / threadsPerBlock);
		dim3 threads(threadsPerBlock);
		mm_naive<float><<<blocks, threads>>>(d_A, d_B, d_C, N);
		checkCuda(cudaDeviceSynchronize());
		checkCuda(cudaMemcpy(h_C_gpu_naive.data(), d_C, sizeof(float) * N * N,
							 cudaMemcpyDeviceToHost));
	}

	// ---- Run Tiled Kernel ----
	{
		dim3 threads(TILE_SIZE, TILE_SIZE);
		dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE,
					(N + TILE_SIZE - 1) / TILE_SIZE);
		mm_tiled<float><<<blocks, threads>>>(d_A, d_B, d_C, N);
		checkCuda(cudaDeviceSynchronize());
		checkCuda(cudaMemcpy(h_C_gpu_tiled.data(), d_C, sizeof(float) * N * N,
							 cudaMemcpyDeviceToHost));
	}

	// ---- Validate ----
	std::cout << "Validating tiled version..." << std::endl;
	bool ok_tiled = validate_results(h_C_ref, h_C_gpu_tiled, N);
	bool ok_naive = validate_results(h_C_ref, h_C_gpu_naive, N);

	if (!ok_tiled || !ok_naive) {
		std::cerr << "Validation FAILED." << std::endl;
		return 1;
	}
	std::cout << "Validation PASSED for both." << std::endl;

	// ---- Measure Runtime ----
	const int num_tests = 100;
	const int num_warmups = 10;
	float time_naive = measure_kernel_latency<float>(launch_naive<float>, N,
													 num_tests, num_warmups);
	float time_tiled = measure_kernel_latency<float>(launch_tiled<float>, N,
													 num_tests, num_warmups);

	std::cout << std::fixed << std::setprecision(4);
	std::cout << "Naive (global memory): " << time_naive << " ms" << std::endl;
	std::cout << "Tiled (shared memory): " << time_tiled << " ms" << std::endl;
	std::cout << "Speedup: " << time_naive / time_tiled << "x" << std::endl;

	checkCuda(cudaFree(d_A));
	checkCuda(cudaFree(d_B));
	checkCuda(cudaFree(d_C));
	return 0;
}