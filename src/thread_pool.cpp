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
                
                // FIX: Decrement pending_tasks_ inside lock to prevent race with wait_all()
                {
                    std::lock_guard<std::mutex> lock(queue_mutex_);
                    pending_tasks_--;
                }
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
    // DEADLOCK FIX: Add 5-minute timeout to prevent infinite hang on OOM/slow I/O
    auto timeout = std::chrono::minutes(5);
    bool completed = done_condition_.wait_for(lock, timeout, [this] {
        return pending_tasks_ == 0;
    });
    
    if (!completed) {
        // Timeout occurred - likely deadlock from memory pressure or slow disk I/O
        throw std::runtime_error(
            "ThreadPool timeout after 5 minutes. Possible causes:\n"
            "  - Out of memory (OOM) causing threads to hang\n"
            "  - Slow disk I/O blocking file reads\n"
            "  - Deadlock in worker task\n"
            "  Pending tasks: " + std::to_string(pending_tasks_)
        );
    }
}

} // namespace squish
