#ifndef CLUE_SFORMAT__
#define CLUE_SFORMAT__

#include <clue/misc.hpp>
#include <string>
#include <ostream>
#include <cstdio>

namespace clue {

// Generic string concatenation

namespace details {

template<class A>
inline void insert_to_stream(std::ostream& os, A&& x) {
    os << x;
}

template<class A, class... Rest>
inline void insert_to_stream(std::ostream& os, A&& x, Rest&&... rest) {
    os << x;
    insert_to_stream(os, std::forward<Rest>(rest)...);
}

} // end namespace details

template<class... Args>
inline std::string sstr(Args&&... args) {
    std::ostringstream ss;
    details::insert_to_stream(ss, std::forward<Args>(args)...);
    return ss.str();
}

inline std::string sstr() {
    return std::string();
}


// Delimited output

template<class Seq>
struct Delimits {
    const Seq& seq;
    const char *delimiter;

    Delimits(const Seq& s, const char *delim)
        : seq(s), delimiter(delim) {}
};

template<class Seq>
inline Delimits<Seq> delimits(const Seq& seq, const char *delim) {
    return Delimits<Seq>(seq, delim);
}

template<class Seq>
inline std::ostream& operator << (std::ostream& out, const Delimits<Seq>& a) {
    auto it = a.seq.begin();
    auto it_end = a.seq.end();
    if (it != it_end) {
        out << *it;
        ++it;
        for(;it != it_end; ++it)
            out << a.delimiter << *it;
    }
    return out;
}

// C formatting

template<typename T>
struct cfmt_t {
    const char *format;
    T value;
};

template<typename T>
inline cfmt_t<T> cfmt(const char *f, T x) {
    return cfmt_t<T>{f, x};
}

template<typename T>
inline std::ostream& operator << (std::ostream& out, const cfmt_t<T>& a) {
    constexpr size_t bufSize = 64;
    char buf[bufSize];
    int n = std::snprintf(buf, bufSize, a.format, a.value);
    if (n < 0)
        throw std::invalid_argument("Failed cfmt caused by invalid argument.");
    if (static_cast<size_t>(n) < bufSize) {
        out << buf;
    } else {
        size_t bufSize2 = static_cast<size_t>(n+1);
        temporary_buffer<char> tbuf(bufSize2);
        int n2 = std::snprintf(tbuf.data(), tbuf.capacity(), a.format, a.value);
        CLUE_ASSERT(n2 == n);
        out << tbuf.data();
    }
    return out;
}

template<typename... Ts>
inline std::string cfmt_s(const char *f, const Ts&... xs) {
    constexpr size_t bufSize = 128;
    char buf[bufSize];
    int n = std::snprintf(buf, bufSize, f, xs...);
    if (n < 0)
        throw std::invalid_argument("Failed cfmt caused by invalid argument.");

    if (static_cast<size_t>(n) < bufSize) {
        return std::string(buf, static_cast<size_t>(n));
    } else {
        size_t bufSize2 = static_cast<size_t>(n+1);
        temporary_buffer<char> tbuf(bufSize2);
        int n2 = std::snprintf(tbuf.data(), tbuf.capacity(), f, xs...);
        CLUE_ASSERT(n2 == n);
        return std::string(tbuf.data(), static_cast<size_t>(n));
    }
}

template<typename T>
inline std::string sstr(const cfmt_t<T>& a) {
    return cfmt_s(a.format, a.value);
}

} // end namespace clue


#endif
