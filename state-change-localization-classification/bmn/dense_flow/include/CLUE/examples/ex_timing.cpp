// a simple example to illustrate the use of timing facilities

#include <clue/timing.hpp>
#include <cstdio>
#include <cstring>

using namespace clue;

static char src[1000000];
static char dst[1000000];

void unused(char c) {}

// copy 1 million bytes
void copy1M() {
    std::memcpy(dst, src, sizeof(src));

    // ensure the copy actually happens in optimized code
    volatile char v = dst[0];
    unused(v);   // suppress warning
}

int main() {
    std::memset(src, 0, sizeof(src));

    auto r = calibrated_time(copy1M);

    std::printf("Result:\n");
    std::printf("    runs    = %zu\n", r.count_runs);
    std::printf("    elapsed = %.4f secs\n", r.elapsed_secs);

    double gps = r.count_runs * 1.0e-3 / r.elapsed_secs;    
    std::printf("    speed   = %.4f Gbytes/sec\n", gps);

    return 0;
}
