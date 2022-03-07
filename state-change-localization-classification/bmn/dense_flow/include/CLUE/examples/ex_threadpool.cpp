// An example adapted from CTPL

#include <clue/thread_pool.hpp>
#include <cstdio>
#include <chrono>

void proc(size_t tidx, size_t id, size_t t) {
    std::printf("thread %zu: proc %zu starting ...\n", tidx, id);
    std::this_thread::sleep_for(std::chrono::milliseconds(t));
    std::printf("thread %zu: proc %zu stopped\n", tidx, id);
}

void test(size_t nth, size_t interval, size_t proc_time) {
    std::printf("Testing with #th = %zu, interval = %zu ms, proc_time = %zu ms\n",
        nth, interval, proc_time);

    clue::thread_pool tpool(nth);

    size_t ntsks = 20;
    for (size_t i = 0; i < ntsks; ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(interval));
        tpool.schedule([i, proc_time](size_t tidx){ proc(tidx, i+1, proc_time); });
    }

    tpool.wait_done();
    std::printf("\n");
}


int main() {
    test(4, 50, 20);
    test(4, 20, 100);
    test(4, 50, 100);
}
