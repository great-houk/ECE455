#include <iostream>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>

constexpr int QUEUE_SIZE = 10;
std::queue<int> queue;
std::mutex mtx;
std::condition_variable cv;
bool done = false;

void producer() {
	// Send 0-99
	for (int i = 0; i < 100; ++i) {
		std::unique_lock<std::mutex> lock(mtx);
		cv.wait(lock, [] { return queue.size() < QUEUE_SIZE; });
		queue.push(i);
		std::cout << "Produced: " << i << std::endl;
		lock.unlock();
		cv.notify_all();
	}
	{
		std::lock_guard<std::mutex> lock(mtx);
		done = true;
	}
	cv.notify_all();
}

void consumer() {
	while (!done) {
		std::unique_lock<std::mutex> lock(mtx);
		cv.wait(lock, [] { return !queue.empty() || done; });
		if (!queue.empty()) {
			int value = queue.front();
			queue.pop();
			std::cout << "Consumed: " << value << std::endl;
			lock.unlock();
			cv.notify_all();
		}
	}
}

int main() {
	std::thread prodThread(producer);
	std::thread consThread(consumer);

	prodThread.join();
	consThread.join();

	return 0;
}