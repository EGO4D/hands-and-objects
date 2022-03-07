/**
 * @file shared_mutex.hpp
 *
 ** @note
 *
 *   The implementation of shared_mutex_impl is adapted from
 *   Howard Hinnant's implementation, which was released with
 *   the Boost License.
 *
 *   The facet classes, including shared_mutex,
 *   shared_timed_mutex, and shared_lock is adapted from libc++,
 *   in order to conform to the C++14/C++17 standard.
 */

#ifndef CLUE_SHARED_MUTEX__
#define CLUE_SHARED_MUTEX__

#include <clue/common.hpp>
#include <mutex>
#include <condition_variable>
#include <chrono>

namespace clue {

using std::unique_lock;

namespace details {

class shared_mutex_impl {
    typedef ::std::mutex mutex_t;
    typedef unsigned int count_t;

    mutex_t mut_;
    ::std::condition_variable gate1_;
    ::std::condition_variable gate2_;
    count_t state_;

    static constexpr count_t write_entered_ = 1U << (sizeof(count_t)*CHAR_BIT - 1);
    static constexpr count_t n_readers_ = ~write_entered_;

public:
    // constructs the mutex
    shared_mutex_impl() :
        state_(0) {}

    // destroys the mutex
    ~shared_mutex_impl() {
        std::lock_guard<mutex_t> _(mut_);
    }

    shared_mutex_impl(const shared_mutex_impl&) = delete;

    shared_mutex_impl& operator=(const shared_mutex_impl&) = delete;

    // exclusive ownership

    // locks the mutex, blocks if the mutex is not available
    void lock() {
        std::unique_lock<mutex_t> lk(mut_);

        while (state_ & write_entered_) gate1_.wait(lk);
        state_ |= write_entered_;
        while (state_ & n_readers_) gate2_.wait(lk);
    }

    // tries to lock the mutex, returns if the mutex is not available
    bool try_lock() {
        std::unique_lock<mutex_t> lk(mut_);
        if (state_ == 0) {
            state_ = write_entered_;
            return true;
        }
        return false;
    }

    // tries to lock the mutex, returns if the mutex has been
    // unavailable until specified time point has been reached
    template <class Clock, class Duration>
    bool try_lock_until(const std::chrono::time_point<Clock, Duration>& due_time) {
        std::unique_lock<mutex_t> lk(mut_);

        if (state_ & write_entered_) {
            while (true) {
                std::cv_status status = gate1_.wait_until(lk, due_time);
                if ((state_ & write_entered_) == 0)
                    break;
                if (status == std::cv_status::timeout)
                    return false;
            }
        }

        state_ |= write_entered_;
        if (state_ & n_readers_) {
            while (true) {
                std::cv_status status = gate2_.wait_until(lk, due_time);
                if ((state_ & n_readers_) == 0)
                    break;
                if (status == std::cv_status::timeout) {
                    state_ &= ~write_entered_;
                    return false;
                }
            }
        }

        return true;
    }

    // unlocks the mutex
    void unlock() {
        std::lock_guard<mutex_t> _(mut_);
        state_ = 0;
        gate1_.notify_all();
    }

    // shared ownership

    // locks the mutex for shared ownership, blocks if the mutex is not available
    void lock_shared() {
        std::unique_lock<mutex_t> lk(mut_);

        while ((state_ & write_entered_) || (state_ & n_readers_) == n_readers_)
            gate1_.wait(lk);
        count_t num_readers = (state_ & n_readers_) + 1;
        state_ &= ~n_readers_;
        state_ |= num_readers;
    }

    // tries to lock the mutex for shared ownership, returns if the mutex is not available
    bool try_lock_shared() {
        std::unique_lock<mutex_t> lk(mut_);
        count_t num_readers = state_ & n_readers_;
        if (!(state_ & write_entered_) && num_readers != n_readers_)
        {
            ++num_readers;
            state_ &= ~n_readers_;
            state_ |= num_readers;
            return true;
        }
        return false;
    }

    // tries to lock the mutex for shared ownership, returns if the mutex has been
    // unavailable until specified time point has been reached
    template <class Clock, class Duration>
    bool try_lock_shared_until(const ::std::chrono::time_point<Clock, Duration>& due_time) {
        std::unique_lock<mutex_t> lk(mut_);
        if ((state_ & write_entered_) || (state_ & n_readers_) == n_readers_) {
            while (true)
            {
                std::cv_status status = gate1_.wait_until(lk, due_time);
                if ((state_ & write_entered_) == 0 && (state_ & n_readers_) < n_readers_)
                    break;
                if (status == std::cv_status::timeout)
                    return false;
            }
        }
        count_t num_readers = (state_ & n_readers_) + 1;
        state_ &= ~n_readers_;
        state_ |= num_readers;
        return true;
    }

    // unlocks the mutex (shared ownership)
    void unlock_shared() {
        std::lock_guard<mutex_t> _(mut_);
        count_t num_readers = (state_ & n_readers_) - 1;
        state_ &= ~n_readers_;
        state_ |= num_readers;
        if (state_ & write_entered_) {
            if (num_readers == 0)
                gate2_.notify_one();
        }
        else {
            if (num_readers == n_readers_ - 1)
                gate1_.notify_one();
        }
    }

}; // end class shared_mutex

} // end namespace details


class shared_mutex {
private:
    details::shared_mutex_impl impl_;

public:
    // Constructors and destructor

    shared_mutex() : impl_() {}
    ~shared_mutex() = default;

    shared_mutex(const shared_mutex&) = delete;
    shared_mutex& operator=(const shared_mutex&) = delete;

    // Exclusive ownership

    void lock()     { return impl_.lock(); }
    bool try_lock() { return impl_.try_lock(); }
    void unlock()   { return impl_.unlock(); }

    // Shared ownership

    void lock_shared()     { return impl_.lock_shared(); }
    bool try_lock_shared() { return impl_.try_lock_shared(); }
    void unlock_shared()   { return impl_.unlock_shared(); }

}; // end class shared_mutex


class shared_timed_mutex {
private:
    details::shared_mutex_impl impl_;

public:
    shared_timed_mutex();
    ~shared_timed_mutex() = default;

    shared_timed_mutex(const shared_timed_mutex&) = delete;
    shared_timed_mutex& operator=(const shared_timed_mutex&) = delete;

    // Exclusive ownership

    void lock()     { impl_.lock(); }
    void unlock()   { impl_.unlock(); }
    bool try_lock() { return impl_.try_lock(); }

    template <class Rep, class Period>
    bool try_lock_for(const ::std::chrono::duration<Rep, Period>& duration) {
        return try_lock_until(::std::chrono::steady_clock::now() + duration);
    }

    template <class Clock, class Duration>
    bool try_lock_until(const ::std::chrono::time_point<Clock, Duration>& due_time) {
        return impl_.try_lock_until(due_time);
    }

    // Shared ownership

    void lock_shared()     { impl_.lock_shared(); }
    void unlock_shared()   { impl_.unlock_shared(); }
    bool try_lock_shared() { return impl_.try_lock_shared(); }

    template <class Rep, class Period>
    bool try_lock_shared_for(const ::std::chrono::duration<Rep, Period>& duration) {
        return try_lock_shared_until(::std::chrono::steady_clock::now() + duration);
    }

    template <class Clock, class Duration>
    bool try_lock_shared_until(const ::std::chrono::time_point<Clock, Duration>& due_time) {
        return impl_.try_lock_shared_until(due_time);
    }

}; // end class shared_timed_mutex


template <class Mutex>
class shared_lock {
public:
    typedef Mutex mutex_type;

private:
    mutex_type* mut_;
    bool owns_;

public:
    // Constructors

    shared_lock() noexcept :
        mut_(nullptr), owns_(false) {}

    explicit shared_lock(mutex_type& m) :
        mut_(&m), owns_(true) {
        mut_->lock_shared();
    }

    shared_lock(mutex_type& m, ::std::defer_lock_t) noexcept :
        mut_(&m), owns_(false) {}

    shared_lock(mutex_type& m, ::std::try_to_lock_t) :
        mut_(&m), owns_(m.try_lock_shared()) {}

    shared_lock(mutex_type& m, ::std::adopt_lock_t) :
        mut_(&m), owns_(true) {}

    template <class Clock, class Duration>
    shared_lock(mutex_type& m, const ::std::chrono::time_point<Clock, Duration>& due_time) :
        mut_(&m), owns_(m.try_lock_shared_until(due_time)) {}

    template <class Rep, class Period>
    shared_lock(mutex_type& m, const ::std::chrono::duration<Rep, Period>& duration) :
        mut_(&m), owns_(m.try_lock_shared_for(duration)) {}

    // Destructor

    ~shared_lock() {
        if (owns_)
            mut_->unlock_shared();
    }

    // Disable copying

    shared_lock(shared_lock const&) = delete;
    shared_lock& operator=(shared_lock const&) = delete;

    // Move

    shared_lock(shared_lock&& u) noexcept :
        mut_(u.mut_), owns_(u.owns_) {
        u.mut_ = nullptr;
        u.owns_ = false;
    }

    shared_lock& operator=(shared_lock&& u) noexcept {
        if (owns_)
            mut_->unlock_shared();
        mut_ = nullptr;
        owns_ = false;
        mut_ = u.mut_;
        owns_ = u.owns_;
        u.mut_ = nullptr;
        u.owns_ = false;
        return *this;
    }

}; // end class shared_lock

} // end namespace clue

#endif
