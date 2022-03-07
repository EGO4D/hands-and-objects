// Use concurrent_queue to implement producer/consumer pattern

#include <clue/concurrent_queue.hpp>
#include <vector>
#include <thread>
#include <chrono>
#include <cstdio>

inline void sleep_for(size_t ms) {
    std::this_thread::sleep_for(std::chrono::microseconds(ms));
}

int main() {
    const size_t M = 2;  // # producers
    const size_t k = 10;  // # items per producer
    size_t produce_time = 100; // 100 ms / per item
    size_t consume_time = 30;  // 30 ms / per item
    size_t remain_nitems = M * k;

    clue::concurrent_queue<double> Q;
    std::vector<std::thread> producers;

    // producers
    for (size_t t = 0; t < M; ++t) {
        producers.emplace_back([&Q,t,k,produce_time](){
            for (size_t i = 0; i < k; ++i) {
                sleep_for(produce_time);
                double v = i + 1;
                std::printf("producer[%zu] >> %g\n", t, v);
                Q.push(v);
            }
        });
    }

    // consumers
    std::thread consumer([&](){
        while (remain_nitems > 0) {
            sleep_for(consume_time);
            double v = Q.wait_pop();
            std::printf("consumer[*] << %g\n", v);
            -- remain_nitems;
        }
    });

    // wait for all threads to complete
    for (auto& th: producers) th.join();
    consumer.join();
}
