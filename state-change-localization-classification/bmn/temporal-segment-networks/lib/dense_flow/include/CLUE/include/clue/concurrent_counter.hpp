#ifndef CLUE_CONCURRENT_COUNTER__
#define CLUE_CONCURRENT_COUNTER__

#include <clue/predicates.hpp>
#include <mutex>
#include <chrono>
#include <condition_variable>

namespace clue {

class concurrent_counter {
public:
    typedef long value_type;
    typedef std::mutex mutex_type;

private:
    value_type cnt_;
    mutable mutex_type mut_;
    std::condition_variable cv_;

public:
    concurrent_counter()
        : cnt_(0) {}

    explicit concurrent_counter(long v0)
        : cnt_(v0) {}

    long get() const {
        std::lock_guard<mutex_type> lk(mut_);
        return cnt_;
    }

    void set(long x) {
        if (x != cnt_) {
            {
                std::lock_guard<mutex_type> lk(mut_);
                cnt_ = x;
            }
            cv_.notify_all();
        }
    }

    void inc(long x = 1) {
        if (x != 0) {
            {
                std::lock_guard<mutex_type> lk(mut_);
                cnt_ += x;
            }
            cv_.notify_all();
        }
    }

    void dec(long x = 1) {
        if (x != 0) {
            {
                std::lock_guard<mutex_type> lk(mut_);
                cnt_ -= x;
            }
            cv_.notify_all();
        }
    }

    void reset() { set(0); }

    template<class Pred>
    void wait(Pred&& pred) {
        std::unique_lock<mutex_type> lk(mut_);
        cv_.wait(lk, [this,&pred](){ return pred(cnt_); });
    }

    template<class Pred, class Rep, class Period>
    bool wait_for(Pred&& pred, const std::chrono::duration<Rep, Period>& dur) {
        std::unique_lock<mutex_type> lk(mut_);
        return cv_.wait_for(lk, dur, [this,&pred](){ return pred(cnt_); });
    }

    template<class Pred, class Clk, class Dur>
    bool wait_until(Pred&& pred, const std::chrono::time_point<Clk, Dur>& t) {
        std::unique_lock<mutex_type> lk(mut_);
        return cv_.wait_until(lk, t, [this,&pred](){ return pred(cnt_); });
    }

    void wait(long v) {
        wait(eq(v));
    }

    template<class Rep, class Period>
    bool wait_for(long v, const std::chrono::duration<Rep, Period>& dur) {
        return wait_for(eq(v), dur);
    }

    template<class Clk, class Dur>
    bool wait_until(long v, const std::chrono::time_point<Clk, Dur>& t) {
        return wait_until(eq(v), t);
    }
};

}

#endif
