Timing
=======

*Timing*, namely to measure the run-time of a piece of code, is a common practice in development, especially in contexts where performance is critical (*e.g.* numerical computation). *CLUE++* provides timing facilities to facilitate this practice. All these facilities are in the namespace ``clue``.


Representation of duration
---------------------------

A class ``duration`` is introduced to represent time durations.

.. cpp:class:: duration

    A wrapper of ``std::chrono::high_resolution_clock::duration`` that exposes more user friendly interface to work with duration.

The ``stop_watch`` class use a ``duration`` object to represent the elapsed time. The ``duration`` class has several member functions to retrieve the duration in different units.

.. cpp:function:: constexpr explicit duration() noexcept

    Construct a zero duration.

.. cpp:function:: constexpr duration(const value_type& val) noexcept

    Construct a duration with an object of class ``value_type``, namely ``std::chrono::high_resolution_clock::duration``.

.. cpp:function:: constexpr double get<U>() const noexcept

    Get the duration in unit ``U``. Here, ``U`` should be an instantiation of the class template ``std::ratio``.

    The following table lists the correspondence between ``U`` and physical time units.

    ===================== ===================
     **type** ``U``        **physical unit**
    --------------------- -------------------
     ``std::ratio<1>``     seconds
     ``std::milli``        milliseconds
     ``std::micro``        microseconds
     ``std::nano``         nanoseconds
     ``std::ratio<60>``    minutes
     ``std::ratio<3600>``  hours
    ===================== ===================

A set of convenient member functions are also provided to make this a bit easier:

.. cpp:function:: constexpr double secs() const noexcept

    Get the duration in seconds.

.. cpp:function:: constexpr double msecs() const noexcept

    Get the duration in milliseconds.

.. cpp:function:: constexpr double usecs() const noexcept

    Get the duration in microseconds.

.. cpp:function:: constexpr double nsecs() const noexcept

    Get the duration in nanoseconds.

.. cpp:function:: constexpr double mins() const noexcept

    Get the duration in minutes.

.. cpp:function:: constexpr double hours() const noexcept

    Get the duration in minutes.

Stopwatch
----------

A ``stop_watch`` class is introduced to measure running time.

.. cpp:class:: stop_watch

    Stop watch class for measuring run-time, in wall-clock sense.

    :note: Internally, it relies on the class ``std::chrono::high_resolution_clock`` introduced in C++11 for timing, and hence it is highly portable.

The class ``stop_watch`` has the following members:

.. cpp:function:: explicit stop_watch(bool st=false) noexcept

    Construct a stop watch. By default, it is not started. One can set ``st`` to ``true`` to let the stop watch starts upon construction.

.. cpp:function:: void reset() noexcept

    Reset the watch: stop it and clear the accumulated elapsed duration.

.. cpp:function:: void start() noexcept

    Start or resume the watch.

.. cpp:function:: void stop() noexcept

    Stop the watch and accumulates the duration of last run to the total elapsed duration.

.. cpp:function:: duration elapsed() const noexcept

    Get the total elapsed time.

Here is an example to illustrate the use of the ``stop_watch`` class.

.. code-block:: cpp

    #include <clue/timing.hpp>

    using namespace clue;

    // simple use

    stop_watch sw(true);  // starts upon construction
    run_my_code();
    std::cout << sw.elapsed().secs() << std::endl;

    // multiple laps

    stop_watch sw1;
    for (size_t i = 0; i < 10; ++i) {
        sw1.start();
        run_my_code();
        sw1.stop();
        std::cout << "cumulative elapsed = "
                  << sw1.elapsed().secs() << std::endl;
    }


Timing functions
------------------

We also provide convenient functions to help people time a certain function.

.. cpp:function:: duration simple_time(F&& f, size_t n, size_t n0 = 0)

    Run the function ``f()`` for ``n`` times and return the total elapsed duration.

    :param f:  The function to be timed.
    :param n:  The number of times ``f`` is to be executed.
    :param n0:  The number of pre-running times. If ``n0 > 0``, it will *pre-run* ``f`` for ``n0`` times to *warm up* the function (for certain functions, the first run or first several runs may take substantially longer time).

.. cpp:function:: calibrated_timing_result calibrated_time(F&& f, double measure_secs = 1.0, double calib_secs = 1.0e-4)

    Calibrated timing.

    This function may spend a little bit time (around ``calib_secs`` seconds) to roughly measure the average running time of ``f()`` (*i.e.* calibaration), and then run ``f()`` for more times for actual measurement such that the entire duration of measurement is around ``measure_secs`` seconds.

    :param f:  The function to be timed.
    :param measure_secs: The time to be spent on actual measurement (in seconds).
    :param calib_secs:   The time to be spent on calibration (in seconds).

    :return: the timing result of class ``calibrated_timing_result``.

.. cpp:class:: calibrated_timing_result

    A struct to represent the result of calibrated timing, which has two fields:

    - ``count_runs``:  the number of runs in actual timing.
    - ``elapsed_secs``: elapsed duration of the actual timing process, in seconds.

**Examples:**

.. code-block:: cpp

    // source file: examples/ex_timing.hpp

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
        unused(v);   // suppress unused warning
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
