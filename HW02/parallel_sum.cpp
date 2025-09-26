#include <iostream>
#include <numeric>
#include <vector>
#include <thread>
#include <chrono>
#include <algorithm>
#include <random>

void sum(const std::vector<int>& data,
		 size_t start,
		 size_t end,
		 long long& out) {
	out = std::accumulate(data.begin() + start, data.begin() + end, 0LL);
}

int main() {
	const size_t AMT = 10'000'000;
	unsigned int SEGS = std::thread::hardware_concurrency();
	if (!SEGS)
		SEGS = 4;
	std::vector<int> data(AMT);
	std::mt19937 mt(42);
	std::uniform_int_distribution dist(1, 100);
	std::generate(data.begin(), data.end(), [&] { return dist(mt); });

	// Single threaded
	auto s0 = std::chrono::high_resolution_clock::now();
	auto baseline = std::accumulate(data.begin(), data.end(), 0LL);
	auto s1 = std::chrono::high_resolution_clock::now();

	// Parallel
	std::vector<std::thread> threads;
	threads.reserve(SEGS);
	std::vector<long long> results(SEGS, 0);
	size_t start = 0;

	auto p0 = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < SEGS - 1; i++) {
		size_t end = start + AMT / SEGS;
		threads.emplace_back(sum, std::cref(data), start, end,
							 std::ref(results[i]));
		start = end;
	}
	threads.emplace_back(sum, std::cref(data), start, AMT,
						 std::ref(results[SEGS - 1]));
	std::for_each(threads.begin(), threads.end(), [](auto& t) { t.join(); });
	auto total = std::accumulate(results.begin(), results.end(), 0LL);
	auto p1 = std::chrono::high_resolution_clock::now();

	// Printout
	std::chrono::duration<double> t_base = s1 - s0;
	std::chrono::duration<double> t_par = p1 - p0;
	std::cout << "Baseline sum: " << baseline << " Time: " << t_base.count()
			  << " s" << std::endl;
	std::cout << "Parallel sum: " << total << " Time: " << t_par.count() << " s"
			  << std::endl;
	return 0;
}