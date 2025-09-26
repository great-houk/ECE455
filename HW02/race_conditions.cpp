#include <iostream>
#include <vector>
#include <thread>
#include <functional>
#include <mutex>
#include <atomic>
#include <chrono>

constexpr int NUM_INCREMENTS = 100'000;

void no_lock(int& counter) {
	for (int i = 0; i < NUM_INCREMENTS; i++)
		counter++;
}

void with_mutex(int& counter, std::mutex& mtx) {
	for (int i = 0; i < NUM_INCREMENTS; i++) {
		std::lock_guard<std::mutex> lock(mtx);
		++counter;
	}
}

void with_atomic(std::atomic<int>& counter) {
	for (int i = 0; i < NUM_INCREMENTS; i++)
		++counter;
}

template <typename Func, typename... Args>
double run_and_time(int num_threads, Func func, Args... args) {
	std::vector<std::thread> threads;
	threads.reserve(num_threads);

	auto start = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < num_threads; i++)
		threads.emplace_back(func, args...);
	for (auto& t : threads)
		t.join();
	auto end = std::chrono::high_resolution_clock::now();

	std::chrono::duration<double> duration = end - start;
	return duration.count();
}

int main() {
	const int T = std::thread::hardware_concurrency()
					  ? std::thread::hardware_concurrency()
					  : 4;
	const int EXPECTED = T * NUM_INCREMENTS;

	// No Lock
	int counter = 0;
	auto duration = run_and_time(T, no_lock, std::ref(counter));
	std::cout << "[No Lock] Final Counter: " << counter
			  << " (Expected: " << EXPECTED << "), Time: " << duration
			  << " seconds\n";

	// With Mutex
	counter = 0;
	std::mutex mtx;
	duration = run_and_time(T, with_mutex, std::ref(counter), std::ref(mtx));
	std::cout << "[With Mutex] Final Counter: " << counter
			  << " (Expected: " << EXPECTED << "), Time: " << duration
			  << " seconds\n";

	// With Atomic
	std::atomic<int> atomic_counter(0);
	duration = run_and_time(T, with_atomic, std::ref(atomic_counter));
	std::cout << "[With Atomic] Final Counter: " << atomic_counter.load()
			  << " (Expected: " << EXPECTED << "), Time: " << duration
			  << " seconds\n";

	return 0;
}