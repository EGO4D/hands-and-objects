// to ensure <clue/clue.hpp> is correct

#include <clue/clue.hpp>
#include <clue/clue.hpp>   // ensure duplicated inclusion is ok

// ensure that all headers are actually included

// misc
using clue::_;
using clue::pass;

// predicates
using clue::eq;
using clue::chars::is_digit;
using clue::chars::is_space;
using clue::floats::is_inf;
using clue::floats::is_nan;

// memory
using clue::aligned_alloc;
using clue::aligned_free;

// array_view
using clue::array_view;

// formatting
using clue::cfmt;
using clue::sstr;

// meta
using clue::meta::type_;

// meta_seq
using clue::meta::seq_;

// optional
using clue::optional;

// reindexed_view
using clue::reindexed_view;

// string_view
using clue::string_view;

// stringex
using clue::trim;
using clue::foreach_token_of;

// timing
using clue::stop_watch;
using clue::calibrated_time;

// type_traits
using clue::enable_if_t;

// value_range
using clue::value_range;

// string_view
using clue::string_view;

// textio
using clue::read_file_content;
using clue::line_stream;

// type_name
using clue::demangle;
using clue::type_name;

// shared_mutex
using clue::shared_mutex;
using clue::shared_timed_mutex;
using clue::shared_lock;

// concurrent_queue
using clue::concurrent_queue;

// concurrent_counter
using clue::concurrent_counter;

// thread_pool
using clue::thread_pool;

int main() {
    return 0;
}
