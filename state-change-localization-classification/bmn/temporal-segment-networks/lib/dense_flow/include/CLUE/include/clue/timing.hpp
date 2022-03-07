/**
 * @file timing.hpp
 *
 * A class for timing.
 */

#ifndef CLUE_TIMING__
#define CLUE_TIMING__

#include <clue/common.hpp>
#include <chrono>

namespace clue {

class stop_watch;

class duration {
	friend class stop_watch;

private:
	using value_type = std::chrono::high_resolution_clock::duration;
	value_type value;

public:
	// construction

	constexpr explicit duration() noexcept :
		value(value_type::zero()) {}

	constexpr duration(const value_type& val) noexcept :
		value(val) {}

	// get values

	template<typename U>
	constexpr double get() const noexcept {
		using ut = std::chrono::duration<double, U>;
		return std::chrono::duration_cast<ut>(value).count();
	}

	constexpr double secs()  const noexcept { return get<std::ratio<1>>();    }
	constexpr double msecs() const noexcept { return get<std::milli>();       }
	constexpr double usecs() const noexcept { return get<std::micro>();       }
	constexpr double nsecs() const noexcept { return get<std::nano>();        }
	constexpr double mins()  const noexcept { return get<std::ratio<60>>();   }
	constexpr double hours() const noexcept { return get<std::ratio<3600>>(); }
};


class stop_watch {
private:
	using clock_t = std::chrono::high_resolution_clock;
	using time_point = clock_t::time_point;
	bool started_;       // whether the stop-watch is started
	duration elapsed_;   // total elapsed time until last stop
	time_point anchor_;  // the time when the stop-watch is last resumed

public:
	explicit stop_watch(bool st=false) noexcept :
		started_(false),
		elapsed_() {
		if (st) {
			start();
		}
	}

	void reset() noexcept {
		started_ = false;
		elapsed_ = duration();
	}

	void start() noexcept {
		if (!started_) {
			anchor_ = clock_t::now();
			started_ = true;
		}
	}

	void stop() noexcept {
		if (started_) {
			elapsed_.value += (clock_t::now() - anchor_);
			started_ = false;
		}
	}

	duration elapsed() const noexcept {
		return started_ ?
			elapsed_.value + (clock_t::now() - anchor_) : elapsed_.value;
	}
};


// timing functions

template<typename F>
inline duration simple_time(F&& f, ::std::size_t n, ::std::size_t n0 = 0) {
	using ::std::size_t;

	// warming
	for (size_t i = 0; i < n0; ++i) f();

	// measuring
	stop_watch sw(true);
	for (size_t i = 0; i < n; ++i) f();
	return sw.elapsed();
}


// calibrated timing

struct calibrated_timing_result {
	const ::std::size_t count_runs;
	const double elapsed_secs;
};

template<typename F>
inline calibrated_timing_result calibrated_time(F&& f,
	                                            double measure_secs = 1.0,
												double calib_secs = 1.0e-4) {
	// warming stage
	stop_watch sw0(true);
	f();
	double avg_et = sw0.elapsed().secs();

	// more accurate calibration of the average elapsed time
	size_t nc = static_cast<size_t>(calib_secs / avg_et);
	if (nc == 0) ++nc;
	stop_watch swc(true);
	for (size_t i = 0; i < nc; ++i) f();
	avg_et = swc.elapsed().secs() / nc;

	// actual measurements
	size_t n = static_cast<size_t>(measure_secs / avg_et);
	if (n == 0) ++n;
	stop_watch sw(true);
	for (size_t i = 0; i < n; ++i) f();
	double et = sw.elapsed().secs();

	// return
	return calibrated_timing_result{ n, et };
}

}

#endif
