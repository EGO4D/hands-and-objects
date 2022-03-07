#include <clue/concurrent_counter.hpp>
#include <thread>

void test_basics() {
    clue::concurrent_counter cc_n;
    clue::concurrent_counter cc_a;

    assert(0 == cc_n.get());
    assert(0 == cc_a.get());

    bool stop = false;
    std::thread worker([&](){
        long i = 0;
        while (!stop) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            cc_n.inc();
            cc_a.inc(static_cast<long>(++i));
        }
    });

    long a = 0;
    long n = 0;
    std::thread listener([&](){
        cc_a.wait( clue::gt(50) );
        stop = true;
        n = cc_n.get();
        a = cc_a.get();
    });

    worker.join();
    listener.join();

    assert(n == 10);
    assert(a == 55);
}

int main() {
    test_basics();
    return 0;
}
