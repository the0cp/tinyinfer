#pragma once

#include <condition_variable>
#include <cstddef>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>
#include <exception>

namespace tinyinfer{

class ThreadPool{
public:
    explicit ThreadPool(size_t num_threads);
    ~ThreadPool();

    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    void enqueue(std::function<void()> task);
    void wait();

    size_t size() const;
private:
    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> tasks_;

    mutable std::mutex mutex_;
    std::condition_variable task_cv_;
    std::condition_variable done_cv_;

    bool stop_ = false;
    size_t active_tasks_ = 0;

    std::exception_ptr first_failure_;
    static thread_local const ThreadPool* current_worker_pool_;

    void wait_until_idle();

    void worker_loop();
};

}