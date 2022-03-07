// An example to illustrate the use of stop_watch class

#include <clue/value_range.hpp>
#include <clue/timing.hpp>
#include <cstdio>
#include <thread>

using std::size_t;
using namespace clue;

inline void sleep_ms(size_t ms) {
    std::this_thread::sleep_for(std::chrono::milliseconds(ms));
}

int main() {
    stop_watch sw;

    const size_t n = 10;
    for (size_t i: vrange(n)) {
        sw.start();
        sleep_ms(100);
        sw.stop();

        std::printf("sleep %2zu: total elapsed = %.4f secs\n",
            (i+1), sw.elapsed().secs());
    }

    return 0;
}
