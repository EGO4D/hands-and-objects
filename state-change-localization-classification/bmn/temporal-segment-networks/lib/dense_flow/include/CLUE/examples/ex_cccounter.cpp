#include <clue/concurrent_counter.hpp>
#include <thread>
#include <chrono>
#include <cstdio>

using namespace clue;

inline void sleep_for(size_t ms) {
    std::this_thread::sleep_for(std::chrono::microseconds(ms));
}

int main() {
    concurrent_counter cnt_orders;  // number of items to produce (-1 indicates termination)
    concurrent_counter cnt_products;  // number of items produced (but not yet consumed)
    double res = 0.0;

    std::thread producer([&](){
        for(;;) {
            cnt_orders.wait( ne(0) );
            long n = cnt_orders.get();
            if (n < 0) break; // use num < 0 to indicate the end

            cnt_orders.dec(n);
            res = 0.0;
            for (long i = 0; i < n; ++i) {
                sleep_for(10);
                res += double(i+1);
            }
            cnt_products.inc(n);
        }
        std::printf("producer exits.\n");
    });

    // response every 10 increments
    std::thread consumer([&](){
        for (long n = 1; n <= 5; ++n) {
            // notify the producer to process n items
            cnt_orders.inc(n);

            // wait until the current batch of production is done
            cnt_products.wait(n);
            cnt_products.reset();

            // process the result
            std::printf("n = %ld ==> res = %g\n", n, res);
        }
        cnt_orders.set(-1);  // notify the producer to terminate
        std::printf("consumer exits.\n");
    });

    producer.join();
    consumer.join();

    return 0;
}
