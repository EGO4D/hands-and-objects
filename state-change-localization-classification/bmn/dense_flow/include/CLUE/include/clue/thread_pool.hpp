#ifndef CLUE_THREAD_POOL__
#define CLUE_THREAD_POOL__

#include <clue/common.hpp>
#include <memory>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <future>
#include <vector>
#include <queue>
#include <stdexcept>
#include <cstdio>

namespace clue {

class thread_pool {
private:
    typedef std::mutex mutex_type;
    typedef std::function<void(size_t)> task_func_t;
    typedef std::queue<task_func_t> task_queue_t;

    struct th_entry_t {
        size_t idx;
        std::thread th;
        bool stopped;

        template<class F>
        th_entry_t(size_t i, F&& f)
            : idx(i)
            , th(f)
            , stopped(false) {}

        void join() {
            if (th.joinable()) th.join();
        }

        void stop() {
            stopped = true;
        }
    };

    std::vector<std::unique_ptr<th_entry_t>> entries_;
    task_queue_t tsk_queue_;

    struct state_t {
        size_t n_pushed = 0;
        size_t n_completed = 0;
        size_t sync_count = 0;
        bool closed = false;
        bool done = false;
        bool stopped = false;

        void revive() {
            closed = false;
            done = false;
            stopped = false;
        }

        bool alive() {
            return !(stopped || done);
        }
    };
    state_t st_;

    mutable mutex_type mut_;
    std::condition_variable cv_; // general notification
    std::condition_variable cv_c_; // notified upon completion of a task

public:
    thread_pool() = default;

    explicit thread_pool(size_t nthreads) {
        resize(nthreads);
    }

    bool empty() const {
        std::lock_guard<mutex_type> lk(mut_);
        return entries_.empty();
    }

    size_t size() const {
        std::lock_guard<mutex_type> lk(mut_);
        return entries_.size();
    }

    const std::thread& get_thread(size_t idx) const {
        std::lock_guard<mutex_type> lk(mut_);
        return entries_.at(idx)->th;
    }

    std::thread& get_thread(size_t idx) {
        std::lock_guard<mutex_type> lk(mut_);
        return entries_.at(idx)->th;
    }

    size_t num_scheduled_tasks() const {
        std::lock_guard<mutex_type> lk(mut_);
        return st_.n_pushed;
    }

    size_t num_completed_tasks() const {
        std::lock_guard<mutex_type> lk(mut_);
        return st_.n_completed;
    }

    // "closed" means no new task can be scheduled
    bool closed() const {
        std::lock_guard<mutex_type> lk(mut_);
        return st_.closed;
    }

    // "done" means all scheduled tasks have been done
    bool done() const {
        std::lock_guard<mutex_type> lk(mut_);
        return st_.done;
    }

    // "stopped" means stopped manually by calling "stop()"
    bool stopped() const {
        std::lock_guard<mutex_type> lk(mut_);
        return st_.stopped;
    }

public:
    void resize(size_t nthreads) {
        if (nthreads == entries_.size())
            return;
        {
            std::lock_guard<mutex_type> lk(mut_);
            resize_(nthreads);
        }
        cv_.notify_all();
    }

    template<class F>
    auto schedule(F&& f) -> std::future<decltype(f((size_t)0))> {
        if (st_.closed) {
            throw std::runtime_error(
                "thread_pool::schedule: "
                "Cannot schedule while the thread_pool is closed.");
        }
        if (st_.sync_count > 0) {
            throw std::runtime_error(
                "thread_pool::schedule: "
                "Cannot schedule while other threads are synchronizing the pool.");
        }
        
        using pck_task_t = std::packaged_task<decltype(f((size_t)0))(size_t)>;
        auto sp = std::make_shared<pck_task_t>(std::forward<F>(f));
        {
            std::lock_guard<mutex_type> lk(mut_);
            tsk_queue_.emplace([sp](size_t idx){
                (*sp)(idx);
            });
            st_.n_pushed ++;
        }
        cv_.notify_one();
        return sp->get_future();
    }

    // synchronize:
    // block until all current tasks have been finished
    // but it does not close the quque
    void synchronize() {
        std::unique_lock<mutex_type> lk(mut_);
        if (st_.n_completed < st_.n_pushed) {
            st_.sync_count ++;
            cv_c_.wait(lk, [this](){
                return st_.n_completed == st_.n_pushed;
            });
            st_.sync_count --;
        }
    }

    // close the queue, so no new tasks can be added
    void close(bool stop_cmd=false) {
        if (st_.closed) return;
        {
            std::lock_guard<mutex_type> lk(mut_);
            st_.closed = true;
            if (stop_cmd) {
                st_.stopped = true;
                for (auto& pe: entries_) pe->stopped = true;
            }
        }
        cv_.notify_all();
    }

    void close_and_stop() {
        close(true);
    }

    // wait until all threads finish their jobs
    // and then clear them
    void join() {
        if (!st_.closed) {
            throw std::runtime_error(
                "thread_pool::join: "
                "The thread pool cannot be joined while it is not closed.");
        }
        for (auto& pe: entries_) {
            pe->join();
        }

        std::lock_guard<mutex_type> lk2(mut_);
        st_.done = tsk_queue_.empty();
        entries_.clear();
    }

    // block until all tasks finish
    void wait_done() {
        close();
        join();
    }

    // block until all current tasks finish
    // remaining tasks are all cleared
    void stop_and_wait() {
        close_and_stop();
        join();
    }

    void clear_tasks() {
        bool to_notify = false;
        {
            std::lock_guard<mutex_type> lk(mut_);
            to_notify = !tsk_queue_.empty();
            while (!tsk_queue_.empty()) {
                tsk_queue_.pop();
            }
        }
        if (to_notify)
            cv_.notify_all();
    }

private:
    bool can_thread_exit(const th_entry_t& e) {
        return e.stopped ||
            (tsk_queue_.empty() && st_.closed);
    }

    bool try_pop_task(size_t th_idx, task_func_t& f) {
        std::lock_guard<mutex_type> lk(mut_);
        const th_entry_t& e = *(entries_.at(th_idx));
        if (can_thread_exit(e)) return false;

        if (!tsk_queue_.empty()) {
            f = std::move(tsk_queue_.front());
            tsk_queue_.pop();
            return true;
        } else {
            return false;
        }
    }

    // wait until:
    // - a task is available: move task to f, and return true, or
    // - the thread (th_idx) should stop: return false
    //
    bool wait_next_task(size_t th_idx, task_func_t& f) {
        std::unique_lock<mutex_type> lk(mut_);
        const th_entry_t& e = *(entries_.at(th_idx));
        cv_.wait(lk, [this,&e](){
            return can_thread_exit(e) || !tsk_queue_.empty();
        });
        if (!e.stopped && !tsk_queue_.empty()) {
            f = std::move(tsk_queue_.front());
            tsk_queue_.pop();
            return true;
        } else {
            return false;
        }
    }

    void on_completed() {
        std::lock_guard<mutex_type> lk(mut_);
        st_.n_completed ++;
        cv_c_.notify_all();
    }

    void resize_(size_t nthreads) {
        size_t n0 = entries_.size();
        if (nthreads > n0) {
            // grow the thread pool
            size_t na = nthreads - n0;
            entries_.reserve(nthreads);
            for (size_t i = 0; i < na; ++i)
                add_thread();
            st_.revive();

        } else if (nthreads < n0) {
            if (st_.alive()) {
                throw std::runtime_error(
                    "thread_pool::resize: "
                    "The pool can be shrinked only when stopped or done.");
            }
            size_t nr = n0 - nthreads;
            for (size_t i = 0; i < nr; ++i) {
                entries_.pop_back();
            }
        }
        CLUE_ASSERT(entries_.size() == nthreads);
    }

    void add_thread() {
        size_t th_idx = entries_.size();
        entries_.emplace_back(new th_entry_t(th_idx, [this, th_idx](){
            task_func_t tfun;
            bool got_tsk = this->try_pop_task(th_idx, tfun);
            for(;;) {
                // execute current task and whatever
                // remain in the task queue
                while (got_tsk) {
                    tfun(th_idx);
                    this->on_completed();
                    got_tsk = this->try_pop_task(th_idx, tfun);
                }
                // wait for new task or a signal to stop
                if (wait_next_task(th_idx, tfun)) {
                    got_tsk = true;
                } else {
                    return;
                }
            }
        }));
    }

}; // end class thread_pool


}

#endif
