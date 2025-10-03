#include <iostream>
#include <vector>
#include <omp.h>

int main() {
	const int N = 10'000'000;
	std::vector<double> data(N, 1.0);

	for (int threads = 1; threads <= 8; threads *= 2) {
		double sum = 0;
		double start_time = omp_get_wtime();

#pragma omp parallel for reduction(+ : sum) num_threads(threads)
		for (int i = 0; i < N; ++i) {
			sum += data[i];
		}

		double end_time = omp_get_wtime();
		std::cout << "Threads: " << threads
				  << ", Time: " << (end_time - start_time)
				  << " seconds, Sum: " << sum << std::endl;
	}
	return 0;
}