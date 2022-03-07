#include <clue/thread_pool.hpp>
#include <cstdio>

void test_construction_and_resize() {
    std::printf("TEST thread_pool: construction + resize\n");
    clue::thread_pool P;

    assert(P.empty());
    assert(0 == P.size());

    P.resize(4);
    assert(!P.empty());
    assert(4 == P.size());

    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    assert(!P.stopped());
    assert(!P.done());

    // verify that get_thread is ok
    for (size_t i = 0; i < 4; ++i) P.get_thread(i);

    P.wait_done();

    assert(0 == P.num_scheduled_tasks());
    assert(0 == P.num_completed_tasks());
    assert(P.closed());
    assert(!P.stopped());
    assert(P.done());
    assert(P.empty());
}

void task(size_t idx, size_t ms) {
    std::this_thread::sleep_for(std::chrono::milliseconds(ms));
}

void test_schedule_and_wait() {
    std::printf("TEST thread_pool: schedule + wait\n");
    clue::thread_pool P(4);

    assert(!P.empty());
    assert(4 == P.size());

    for (size_t i = 0; i < 20; ++i) {
        P.schedule([](size_t tid){ task(tid, 5); });
    }

    P.wait_done();

    assert(20 == P.num_scheduled_tasks());
    assert(20 == P.num_completed_tasks());
    assert(P.closed());
    assert(!P.stopped());
    assert(P.done());
    assert(P.empty());
}

void test_synchronize() {
    std::printf("TEST thread_pool: synchronize\n");
    clue::thread_pool P(4);

    assert(!P.empty());
    assert(4 == P.size());

    for (size_t i = 0; i < 20; ++i) {
        P.schedule([](size_t tid){ task(tid, 10); });
    }
    P.synchronize();

    assert(20 == P.num_completed_tasks());
    assert(20 == P.num_scheduled_tasks());
    assert(!P.closed());

    for (size_t i = 0; i < 20; ++i) {
        P.schedule([](size_t tid){ task(tid, 10); });
    }
    P.synchronize();

    assert(40 == P.num_completed_tasks());
    assert(40 == P.num_scheduled_tasks());
    assert(!P.closed());

    P.wait_done();

    assert(40 == P.num_scheduled_tasks());
    assert(40 == P.num_completed_tasks());
    assert(P.closed());
    assert(!P.stopped());
    assert(P.done());
    assert(P.empty());
}


void test_early_stop_and_revive() {
    std::printf("TEST thread_pool: early stop + revive\n");
    clue::thread_pool P(2);

    assert(!P.empty());
    assert(2 == P.size());

    for (size_t i = 0; i < 10; ++i) {
        P.schedule([](size_t tid){ task(tid, 50); });
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(125));
    P.stop_and_wait();  // will wait for active tasks to finish

    assert(10 == P.num_scheduled_tasks());
    assert(6 == P.num_completed_tasks());
    assert(P.closed());
    assert(P.stopped());
    assert(!P.done());
    assert(P.empty());

    P.resize(4);
    P.wait_done();

    assert(10 == P.num_scheduled_tasks());
    assert(10 == P.num_completed_tasks());
    assert(P.closed());
    assert(!P.stopped());
    assert(P.done());
    assert(P.empty());
}


int main() {
    test_construction_and_resize();
    test_schedule_and_wait();
    test_synchronize();
    test_early_stop_and_revive();
    return 0;
}
