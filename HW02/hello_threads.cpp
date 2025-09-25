#include <iostream>
#include <thread>
#include <vector>
#include <algorithm>

void hello(int id, int N) {
	printf("Hello from thread %i of %i total\n", id, N);
}

int main() {
	const int N = 5;
	std::vector<std::thread> threads;

	for (int i = 0; i < N; i++)
		threads.emplace_back(hello, i, N);

	std::for_each(threads.begin(), threads.end(), std::thread::join);
	return 0;
}