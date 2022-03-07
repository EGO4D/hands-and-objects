#ifndef CLUE_CONCURRENT_QUEUE__
#define CLUE_CONCURRENT_QUEUE__

#include <clue/common.hpp>
#include <mutex>
#include <condition_variable>
#include <queue>

namespace clue {

template<class T, class Container=std::deque<T>>
class concurrent_queue final {
private:
    using mutex_type = std::mutex;
    std::queue<T, Container> queue_;
    mutex_type mut_;
    std::condition_variable cv1_; // notify when the queue becomes non-empty

public:
    ~concurrent_queue() {
        synchronize();
    }

    size_t size() const {
        return queue_.size();
    }

    bool empty() const {
        return queue_.empty();
    }

    void synchronize() {
        std::lock_guard<mutex_type> lk(mut_);
    }

    void clear() {
        std::lock_guard<mutex_type> lk(mut_);
        while (!empty()) {
            queue_.pop();
        }
    }

    void push(const T& x) {
        {
            std::lock_guard<mutex_type> lk(mut_);
            queue_.push(x);
        }
        if (size() == 1) cv1_.notify_one();
    }

    void push(T&& x) {
        {
            std::lock_guard<mutex_type> lk(mut_);
            queue_.push(std::move(x));
        }
        if (size() == 1) cv1_.notify_one();
    }

    template<class... Args>
    void push(Args&&... args) {
        {
            std::lock_guard<mutex_type> lk(mut_);
            queue_.emplace(std::forward<Args>(args)...);
        }
        if (size() == 1) cv1_.notify_one();
    }

    // If it is non empty, pop and write the front element to dst,
    // and return true, otherwise, it returns false immediately.
    bool try_pop(T& dst) {
        std::lock_guard<mutex_type> lk(mut_);
        if (empty()) return false;
        dst = std::move(queue_.front());
        queue_.pop();
        return true;
    }

    // Wait until non-empty and then pop
    T wait_pop() {
        std::unique_lock<mutex_type> lk(mut_);
        cv1_.wait(lk, [this](){ return !empty(); });
        T x = std::move(queue_.front());
        queue_.pop();
        return std::move(x);
    }
};


} // end namespace clue

#endif
