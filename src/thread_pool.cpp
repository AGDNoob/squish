#include "thread_pool.hpp"

namespace squish {

ThreadPool::ThreadPool(size_t num_threads) {
    // hardware_concurrency als default, 4 wenn das fehlschlägt
    if (num_threads == 0) {
        num_threads = std::thread::hardware_concurrency();
        if (num_threads == 0) num_threads = 4;  // fallback
    }

    workers_.reserve(num_threads);
    for (size_t i = 0; i < num_threads; ++i) {
        workers_.emplace_back([this] {
            while (true) {
                std::function<void()> task;
                
                {
                    std::unique_lock<std::mutex> lock(queue_mutex_);
                    condition_.wait(lock, [this] {
                        return stop_ || !tasks_.empty();
                    });
                    
                    if (stop_ && tasks_.empty()) {
                        return;
                    }
                    
                    task = std::move(tasks_.front());
                    tasks_.pop();
                }
                
                task();
                
                pending_tasks_--;
                done_condition_.notify_all();
            }
        });
    }
}

ThreadPool::~ThreadPool() {
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        stop_ = true;
    }
    condition_.notify_all();
    
    for (std::thread& worker : workers_) {
        if (worker.joinable()) {
            worker.join();
        }
    }
}

void ThreadPool::wait_all() {
    std::unique_lock<std::mutex> lock(queue_mutex_);
    done_condition_.wait(lock, [this] {
        return pending_tasks_ == 0;
    });
}

} // namespace squish
