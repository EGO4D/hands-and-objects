#ifndef CLUE_MISC__
#define CLUE_MISC__

// Miscellaneous utilities

#include <clue/common.hpp>
#include <memory>
#include <sstream>

namespace clue {

struct place_holder_t {};
constexpr place_holder_t _{};

template<typename... Args>
inline void pass(Args&&... args) {}

template<class T, class... Args>
std::unique_ptr<T> make_unique(Args&&... args) {
    return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}


template<typename T>
class temporary_buffer final {
private:
    std::pair<T*, std::ptrdiff_t> ret_;

public:
    temporary_buffer(size_t n)
        : ret_(std::get_temporary_buffer<T>(static_cast<ptrdiff_t>(n))) {}

    ~temporary_buffer() {
        std::return_temporary_buffer(ret_.first);
    }

    temporary_buffer(const temporary_buffer&) = delete;
    temporary_buffer& operator= (const temporary_buffer&) = delete;

    size_t capacity() const noexcept {
        return static_cast<size_t>(ret_.second);
    }

    T* data() noexcept {
        return ret_.first;
    }
};

}

#endif
