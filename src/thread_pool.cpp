#include "thread_pool.h"

#include <stdexcept>

namespace tinyinfer{

ThreadPool::ThreadPool(size_t num_threads){
    if(num_threads == 0){
        throw std::runtime_error("ThreadPool requires at least one thread.");
    }

    workers_.reserve(num_threads);

    for(size_t i = 0; i < num_threads; i++){
        workers_.emplace_back([this](){
            worker_loop();
        });
    }
}

ThreadPool::~ThreadPool(){
    wait();

    {
        std::lock_guard<std::mutex> lock(mutex_);
        stop_ = true;
    }

    task_cv_.notify_all();

    for(auto& worker : workers_){
        if(worker.joinable()){
            worker.join();
        }
    }
}

void ThreadPool::enqueue(std::function<void()> task){
    {
        std::lock_guard<std::mutex> lock(mutex_);

        if(stop_){
            throw std::runtime_error("Cannot enqueue task after ThreadPool stopped.");
        }

        tasks_.push(std::move(task));
    }

    task_cv_.notify_one();
}

void ThreadPool::wait(){
    std::unique_lock<std::mutex> lock(mutex_);

    done_cv_.wait(lock, [this](){
        return tasks_.empty() && active_tasks_ == 0;
    });
}

size_t ThreadPool::size() const{
    return workers_.size();
}

void ThreadPool::worker_loop(){
    while(true){
        std::function<void()> task;

        {
            std::unique_lock<std::mutex> lock(mutex_);

            task_cv_.wait(lock, [this](){
                return stop_ || !tasks_.empty();
            });

            if(stop_ && tasks_.empty()){
                return;
            }

            task = std::move(tasks_.front());
            tasks_.pop();
            active_tasks_++;
        }

        task();

        {
            std::lock_guard<std::mutex> lock(mutex_);

            active_tasks_--;

            if(tasks_.empty() && active_tasks_ == 0){
                done_cv_.notify_all();
            }
        }
    }
}

}