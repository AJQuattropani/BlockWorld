#pragma once

#include "Log.hpp"

#include <forward_list>
#include <future>
#include <algorithm>
#include <queue>
#include <functional>
#include <mutex>

namespace utils
{
	class ThreadList {
	public:
		ThreadList(size_t size)
		{
			thread_list.reserve(size);
			for (size_t id = 0; id < size; id++)
				thread_list.emplace_back(std::bind(&ThreadList::executionLoop, this, id));
		}

		void pushTask(std::function<void(void)> task)
		{
			std::unique_lock<std::mutex> pushLock(lock);

			work_load.emplace(std::move(task));
			conditional.notify_one();
		}

		~ThreadList()
		{
			{
				std::unique_lock<std::mutex> deleteLock(lock);

				terminating = true;
				conditional.notify_all();
			}

			for (std::thread& thread : thread_list) thread.join();
		}
	private:
		bool terminating = false;
		std::vector<std::thread> thread_list;
		std::mutex lock;
		std::condition_variable conditional;
		std::queue<std::function<void(void)>> work_load;
	private:
		void executionLoop(int id)
		{
			std::function <void(void)> current_task;

			while (true)
			{
				{
					std::unique_lock<std::mutex> collectTaskLock(lock);

					// wait to be delegated a task
					conditional.wait(collectTaskLock, [this] {
						return !work_load.empty() || terminating;
						});

					// will not exit unless destruction is called and all work has completed.
					if (work_load.empty() && terminating)
					{
						BW_DEBUG("<Thread %d closed.>", id);
						return;
					}

					current_task = std::move(work_load.front());
					work_load.pop();
				}

				current_task();

			}

		}

	};

}