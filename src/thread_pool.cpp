#include "thread_pool.h"

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <utility>

namespace tinyinfer{

thread_local const ThreadPool* ThreadPool::current_worker_pool_ = nullptr;

ThreadPool::ThreadPool(size_t num_threads){
    if(num_threads == 0){
        throw std::runtime_error("ThreadPool requires at least one thread.");
    }

    workers_.reserve(num_threads);

    for(size_t i = 0; i < num_threads; i++){
        workers_.emplace_back([this](){ worker_loop(); });
    }
}

ThreadPool::~ThreadPool(){
    try{
        wait_until_idle();
    }catch(...){}

    {
        std::lock_guard<std::mutex> lock(mutex_);
        stop_ = true;
        first_failure_ = nullptr;
    }

    task_cv_.notify_all();

    for(auto& worker : workers_){
        if(worker.joinable()){
            worker.join();
        }
    }
}

void ThreadPool::enqueue(std::function<void()> task){
    if(!task){
        throw std::invalid_argument("Cannot enqueue an empty task.");
    }

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
    if(current_worker_pool_ == this){
        throw std::logic_error("A worker cannot wait on its own ThreadPool.");
    }

    std::exception_ptr failure;

    {
        std::unique_lock<std::mutex> lock(mutex_);
        done_cv_.wait(lock, [this](){ return tasks_.empty() && active_tasks_ == 0; });
        failure = std::exchange(first_failure_, nullptr);
    }

    if(failure){
        std::rethrow_exception(failure);
    }
}

void ThreadPool::parallel_for(
    size_t work_items,
    size_t task_count,
    std::function<void(size_t begin, size_t end)> task
){
    if(!task){
        throw std::invalid_argument("ThreadPool::parallel_for requires a task.");
    }
    if(work_items == 0){
        return;
    }
    if(is_current_worker_thread()){
        throw std::logic_error("A worker cannot wait on its own ThreadPool task group.");
    }

    if(task_count == 0){
        task_count = size();
    }
    task_count = std::max<size_t>(1, std::min(task_count, work_items));

    struct TaskGroupState{
        std::mutex mutex;
        std::condition_variable cv;
        size_t remaining = 0;
        std::exception_ptr first_failure;
    };

    auto state = std::make_shared<TaskGroupState>();
    state->remaining = task_count;
    const size_t base_items = work_items / task_count;
    const size_t extra_items = work_items % task_count;
    size_t enqueued = 0;

    try{
        for(size_t task_index = 0; task_index < task_count; task_index++){
            // Give the first `extra_items` tasks one additional item. Unlike a
            // ceil-sized split, this guarantees that every scheduled range is
            // non-empty when task_count <= work_items.
            const size_t begin =
                task_index * base_items + std::min(task_index, extra_items);
            const size_t count = base_items + (task_index < extra_items ? 1 : 0);
            const size_t end = begin + count;

            enqueue([state, task, begin, end](){
                std::exception_ptr failure;
                try{
                    task(begin, end);
                }catch(...){
                    failure = std::current_exception();
                }

                std::lock_guard<std::mutex> lock(state->mutex);
                if(failure && !state->first_failure){
                    state->first_failure = failure;
                }
                state->remaining--;
                if(state->remaining == 0){
                    state->cv.notify_all();
                }
            });
            enqueued++;
        }
    }catch(...){
        std::lock_guard<std::mutex> lock(state->mutex);
        if(!state->first_failure){
            state->first_failure = std::current_exception();
        }
        state->remaining -= task_count - enqueued;
        if(state->remaining == 0){
            state->cv.notify_all();
        }
    }

    std::exception_ptr failure;
    {
        std::unique_lock<std::mutex> lock(state->mutex);
        state->cv.wait(lock, [&](){ return state->remaining == 0; });
        failure = state->first_failure;
    }
    if(failure){
        std::rethrow_exception(failure);
    }
}

void ThreadPool::wait_until_idle(){
    std::unique_lock<std::mutex> lock(mutex_);
    done_cv_.wait(lock, [this](){ return tasks_.empty() && active_tasks_ == 0; });
}

bool ThreadPool::is_current_worker_thread() const noexcept{
    return current_worker_pool_ == this;
}

size_t ThreadPool::size() const{
    return workers_.size();
}

void ThreadPool::worker_loop(){
    current_worker_pool_ = this;

    while(true){
        std::function<void()> task;

        {
            std::unique_lock<std::mutex> lock(mutex_);
            task_cv_.wait(lock, [this](){ return stop_ || !tasks_.empty(); });

            if(stop_ && tasks_.empty()){
                return;
            }

            task = std::move(tasks_.front());
            tasks_.pop();
            active_tasks_++;
        }

        std::exception_ptr failure;

        try{
            task();
        }catch(...){
            failure = std::current_exception();
        }

        {
            std::lock_guard<std::mutex> lock(mutex_);

            if(failure && !first_failure_){
                first_failure_ = failure;
            }

            active_tasks_--;

            if(tasks_.empty() && active_tasks_ == 0){
                done_cv_.notify_all();
            }
        }
    }
}

}
