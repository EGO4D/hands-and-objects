// The implementation is imported from reliable sources,
// this unit tests basically to ensure that it is imported correctly.

#include <clue/shared_mutex.hpp>
#include <thread>
#include <iostream>

using namespace clue;

class Gate {
    bool fired;
    std::condition_variable cv_;
    std::mutex mut_;

public:
    Gate() : fired(false) {}

    void wait() {
        unique_lock<std::mutex> lk(mut_);
        if (!fired) {
            cv_.wait(lk);
        }
    }

    void fire() {
        unique_lock<std::mutex> lk(mut_);
        fired = true;
        cv_.notify_all();
    }
};

void test_exclusive_lock() {
    std::printf("Testing exclusive locking ...\n");

    shared_mutex smut;
    bool correct = true;

    Gate t1_locked;
    Gate t2_testdone;

    std::thread t1([&](){
        std::printf("  t1: enter\n");
        unique_lock<shared_mutex> lk(smut);
        std::printf("  t1: locked\n");
        t1_locked.fire();
        std::printf("  t1: wait for exit\n");
        t2_testdone.wait();
        std::printf("  t1: exit\n");
    });

    std::thread t2([&](){
        std::printf("  t2: enter\n");
        t1_locked.wait();
        std::printf("  t2: try lock\n");
        bool b = smut.try_lock();
        if (b) smut.unlock();
        std::printf("  t2: test done\n");
        t2_testdone.fire();

        // b is supposed to be false
        if (b) correct = false;
        std::printf("  t2: exit\n");
    });

    t1.join();
    t2.join();
    assert(correct);

    std::thread t3([&](){
        std::printf("  t3: enter\n");
        std::printf("  t3: try lock\n");
        bool b = smut.try_lock();
        if (b) smut.unlock();
        std::printf("  t3: test done\n");

        // b is supposed to be true
        if (!b) correct = false;
        std::printf("  t3: exit\n");
    });

    t3.join();
    assert(correct);
}


void test_shared_lock() {
    std::printf("Testing shared locking ...\n");

    shared_mutex smut;
    bool correct = true;

    Gate t1_locked;
    Gate t2_testdone;

    std::thread t1([&](){
        std::printf("  t1: enter\n");
        unique_lock<shared_mutex> lk(smut);
        std::printf("  t1: locked\n");
        t1_locked.fire();
        std::printf("  t1: wait for exit\n");
        t2_testdone.wait();
        std::printf("  t1: exit\n");
    });

    std::thread t2([&](){
        std::printf("  t2: enter\n");
        t1_locked.wait();
        std::printf("  t2: try lock shared\n");
        bool b = smut.try_lock_shared();
        if (b) smut.unlock();
        std::printf("  t2: test done\n");
        t2_testdone.fire();

        // b is supposed to be false
        if (b) correct = false;
        std::printf("  t2: exit\n");
    });

    t1.join();
    t2.join();
    assert(correct);

    Gate t3_locked;
    Gate t4_locked;
    Gate t5_testdone;

    std::thread t3([&](){
        std::printf("  t3: enter\n");
        shared_lock<shared_mutex> lk(smut);
        std::printf("  t3: locked (shared)\n");
        t3_locked.fire();
        t4_locked.wait();
        t5_testdone.wait();
        std::printf("  t3: exit\n");
    });

    std::thread t4([&](){
        std::printf("  t4: enter\n");
        shared_lock<shared_mutex> lk(smut);
        std::printf("  t4: locked (shared)\n");
        t4_locked.fire();
        t3_locked.wait();
        std::printf("  t4: exit\n");
    });

    std::thread t5([&](){
        std::printf("  t5: enter\n");
        t3_locked.wait();
        std::printf("  t5: try lock\n");
        bool b = smut.try_lock();
        if (b) smut.unlock();
        if (b) correct = false;
        std::printf("  t5: test done\n");
        t5_testdone.fire();
        std::printf("  t5: exit\n");
    });

    t3.join();
    t4.join();
    t5.join();
    assert(correct);
}

int main() {
    test_exclusive_lock();
    test_shared_lock();
    return 0;
}

