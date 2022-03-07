/**
 * @file stringex.hpp
 *
 * Extensions of string facilities.
 */

#ifndef CLUE_STRINGEX__
#define CLUE_STRINGEX__

#include <clue/type_traits.hpp>
#include <clue/string_view.hpp>
#include <clue/predicates.hpp>
#include <vector>
#include <sstream>
#include <cctype>

namespace clue {

namespace details {

template<typename T>
struct is_char {
    static constexpr bool value =
        ::std::is_same<T, char>::value ||
        ::std::is_same<T, wchar_t>::value ||
        ::std::is_same<T, char16_t>::value ||
        ::std::is_same<T, char32_t>::value;
};

template<typename T>
struct is_cchar {
    static constexpr bool value =
        ::std::is_same<T, char>::value ||
        ::std::is_same<T, wchar_t>::value;
};

template<size_t N>
inline bool _icmp(const char *s, const char *r) {
    for (size_t i = 0; i < N; ++i) {
        if (static_cast<char>(std::tolower(s[i])) != r[i])
            return false;
    }
    return true;
}

}

//===============================================
//
//   Make string view
//
//===============================================

template<typename charT, typename Traits, typename Allocator>
constexpr basic_string_view<charT, Traits> view(const ::std::basic_string<charT, Traits, Allocator>& s) {
    return basic_string_view<charT, Traits>(s);
}


//===============================================
//
//   Prefix & suffix
//
//===============================================

// prefix

template<typename charT, typename Traits>
constexpr basic_string_view<charT, Traits>
prefix(basic_string_view<charT, Traits> str, ::std::size_t n) noexcept {
    return str.substr(0, n);
}

template<typename charT, typename Traits, typename Allocator>
inline ::std::basic_string<charT, Traits, Allocator>
prefix(const ::std::basic_string<charT, Traits, Allocator>& str, ::std::size_t n) {
    return str.substr(0, n);
}

// suffix

template<typename charT, typename Traits>
constexpr basic_string_view<charT, Traits>
suffix(basic_string_view<charT, Traits> str, ::std::size_t n) noexcept {
    return n > str.size() ? str : str.substr(str.size() - n, n);
}

template<typename charT, typename Traits, typename Allocator>
inline ::std::basic_string<charT, Traits, Allocator>
suffix(const ::std::basic_string<charT, Traits, Allocator>& str, ::std::size_t n) {
    return n > str.size() ? str : str.substr(str.size() - n, n);
}

// starts_with (char)

template<typename charT>
inline typename ::std::enable_if<details::is_char<charT>::value, bool>::type
starts_with(const charT* str, charT c) noexcept {
    using Traits = ::std::char_traits<charT>;
    return *str && Traits::eq(*str, c);
}

template<typename charT, typename Traits>
inline bool starts_with(basic_string_view<charT, Traits> str, charT c) noexcept {
    return !str.empty() && Traits::eq(str.front(), c);
}

template<typename charT, typename Traits, typename Allocator>
inline bool starts_with(const ::std::basic_string<charT, Traits, Allocator>& str, charT c) noexcept {
    return !str.empty() && Traits::eq(str.front(), c);
}


// starts_with (string)

template<typename charT>
inline typename ::std::enable_if<details::is_char<charT>::value, bool>::type
starts_with(const charT* str, const charT* sub) noexcept {
    using Traits = ::std::char_traits<charT>;
    for (;(*str) && (*sub) && Traits::eq(*sub, *str); str++, sub++);
    return !(*sub);
}

template<typename charT, typename Traits>
inline bool starts_with(const charT* str, basic_string_view<charT, Traits> sub) noexcept {
    auto send = sub.cend();
    auto sp = sub.cbegin();
    for(;(*str) && (sp != send) && Traits::eq(*str, *sp); str++, sp++);
    return sp == send;
}

template<typename charT, typename Traits, typename Allocator>
inline bool starts_with(const charT* str,
                        const std::basic_string<charT, Traits, Allocator>& sub) noexcept {
    return starts_with(str, view(sub));
}

template<typename charT, typename Traits>
inline bool starts_with(basic_string_view<charT, Traits> str, const charT *sub) {
    auto end = str.cend();
    auto p = str.cbegin();
    for(;(p != end) && (*sub) && Traits::eq(*p, *sub); p++, sub++);
    return !(*sub);
}

template<typename charT, typename Traits>
inline bool starts_with(basic_string_view<charT, Traits> str,
                        basic_string_view<charT, Traits> sub) noexcept {
    return str.size() >= sub.size() && str.substr(0, sub.size()) == sub;
}

template<typename charT, typename Traits, typename Allocator>
inline bool starts_with(basic_string_view<charT, Traits> str,
                        const ::std::basic_string<charT, Traits, Allocator>& sub) noexcept {
    return starts_with(str, view(sub));
}

template<typename charT, typename Traits, typename Allocator>
inline bool starts_with(const ::std::basic_string<charT, Traits, Allocator>& str,
                        const charT *sub) noexcept {
    return starts_with(view(str), sub);
}

template<typename charT, typename Traits, typename Allocator>
inline bool starts_with(const ::std::basic_string<charT, Traits, Allocator>& str,
                        basic_string_view<charT, Traits> sub) noexcept {
    return starts_with(view(str), sub);
}

template<typename charT, typename Traits, typename Allocator, typename Allocator2>
inline bool starts_with(const ::std::basic_string<charT, Traits, Allocator>& str,
                        const ::std::basic_string<charT, Traits, Allocator2>& sub) noexcept {
    return starts_with(view(str), view(sub));
}


// ends_with (char)

template<typename charT>
inline typename ::std::enable_if<details::is_char<charT>::value, bool>::type
ends_with(const charT* str, charT c) noexcept {
    if (!(*str)) return false;
    using traits_t = ::std::char_traits<charT>;
    return traits_t::eq(str[traits_t::length(str) - 1], c);
}

template<typename charT, typename Traits>
inline bool ends_with(basic_string_view<charT, Traits> str, charT c) noexcept {
    return !str.empty() && Traits::eq(str.back(), c);
}

template<typename charT, typename Traits, typename Allocator>
inline bool ends_with(const ::std::basic_string<charT, Traits, Allocator>& str, charT c) noexcept {
    return !str.empty() && Traits::eq(str.back(), c);
}


// ends_with (string)

template<typename charT, typename Traits>
inline bool ends_with(basic_string_view<charT, Traits> str,
                      basic_string_view<charT, Traits> sub) noexcept {
    ::std::size_t n = sub.size();
    return str.size() >= n && str.substr(str.size() - n, n) == sub;
}

template<typename charT, typename Traits>
inline bool ends_with(basic_string_view<charT, Traits> str,
                      const charT *sub) noexcept {
    using view_t = basic_string_view<charT, Traits>;
    return ends_with(str, view_t(sub));
}

template<typename charT, typename Traits, typename Allocator>
inline bool ends_with(basic_string_view<charT, Traits> str,
                      const ::std::basic_string<charT, Traits, Allocator>& sub) noexcept {
    return ends_with(str, view(sub));
}

template<typename charT>
inline bool ends_with(const charT *str, const charT *sub) noexcept {
    using view_t = basic_string_view<charT>;
    return ends_with(view_t(str), view_t(sub));
}

template<typename charT, typename Traits>
inline bool ends_with(const charT *str, basic_string_view<charT, Traits> sub) noexcept {
    using view_t = basic_string_view<charT, Traits>;
    return ends_with(view_t(str), sub);
}

template<typename charT, typename Traits, typename Allocator>
inline bool ends_with(const charT *str,
                      const ::std::basic_string<charT, Traits, Allocator>& sub) noexcept {
    using view_t = basic_string_view<charT, Traits>;
    return ends_with(view_t(str), view(sub));
}

template<typename charT, typename Traits, typename Allocator>
inline bool ends_with(const ::std::basic_string<charT, Traits, Allocator>& str,
                      const charT *sub) noexcept {
    using view_t = basic_string_view<charT, Traits>;
    return ends_with(view(str), view_t(sub));
}

template<typename charT, typename Traits, typename Allocator>
inline bool ends_with(const ::std::basic_string<charT, Traits, Allocator>& str,
                      const basic_string_view<charT, Traits> sub) noexcept {
    return ends_with(view(str), sub);
}

template<typename charT, typename Traits, typename Allocator, typename Allocator2>
inline bool ends_with(const ::std::basic_string<charT, Traits, Allocator>& str,
                      const ::std::basic_string<charT, Traits, Allocator2>& sub) noexcept {
    return ends_with(view(str), view(sub));
}


//===============================================
//
//   Trimming
//
//===============================================

template<typename charT, typename Traits>
inline basic_string_view<charT, Traits>
trim_left(basic_string_view<charT, Traits> str) {
    if (str.empty()) return str;
    const charT *p = str.cbegin();
    const charT *end = str.cend();
    while (p != end && chars::is_space(*p)) ++p;
    return basic_string_view<charT, Traits>(p, ::std::size_t(end - p));
}

template<typename charT, typename Traits>
inline basic_string_view<charT, Traits>
trim_right(basic_string_view<charT, Traits> str) {
    if (str.empty()) return str;
    const charT *begin = str.cbegin();
    const charT *q = str.cend();
    while (q != begin && chars::is_space(*(q-1))) --q;
    return basic_string_view<charT, Traits>(begin, ::std::size_t(q - begin));
}

template<typename charT, typename Traits>
inline basic_string_view<charT, Traits>
trim(basic_string_view<charT, Traits> str) {
    if (str.empty()) return str;
    const charT *p = str.cbegin();
    const charT *q = str.cend();
    while (p != q && chars::is_space(*p)) ++p;  // trim left
    while (q != p && chars::is_space(*(q-1))) --q;  // trim right
    return basic_string_view<charT, Traits>(p, ::std::size_t(q - p));
}

template<typename charT, typename Traits, typename Allocator>
inline ::std::basic_string<charT, Traits, Allocator>
trim_left(const ::std::basic_string<charT, Traits, Allocator>& str) {
    basic_string_view<charT, Traits> r = trim_left(view(str));
    return ::std::basic_string<charT, Traits, Allocator>(r.data(), r.size());
}

template<typename charT, typename Traits, typename Allocator>
inline ::std::basic_string<charT, Traits, Allocator>
trim_right(const ::std::basic_string<charT, Traits, Allocator>& str) {
    basic_string_view<charT, Traits> r = trim_right(view(str));
    return ::std::basic_string<charT, Traits, Allocator>(r.data(), r.size());
}

template<typename charT, typename Traits, typename Allocator>
inline ::std::basic_string<charT, Traits, Allocator>
trim(const ::std::basic_string<charT, Traits, Allocator>& str) {
    basic_string_view<charT, Traits> r = trim(view(str));
    return ::std::basic_string<charT, Traits, Allocator>(r.data(), r.size());
}


//===============================================
//
//   Value Parsing
//
//===============================================

namespace details {

// a pointer p is considered as a valid end, if
// p != sv.begin(), and
// [p, sv.end()) are all spaces
//
template<typename Traits>
inline bool is_valid_parse_end(basic_string_view<char, Traits> sv, const char *p) noexcept {
    if (p == sv.begin()) return false;
    const char *sv_end = sv.end();
    while (p != sv_end && chars::is_space(*p)) ++p;
    return p == sv_end;
}

inline bool is_valid_parse_end(const char *sz, const char *p) noexcept {
    if (p == sz) return false;
    while (*p && chars::is_space(*p)) ++p;
    return !(*p);
}

template<bool Longer>
struct integer_parse_helper;

template<>
struct integer_parse_helper<false> {
    using type = long;
    static type run(const char *p, char** ppend) {
        return ::std::strtol(p, ppend, 0);
    }
};

template<>
struct integer_parse_helper<true> {
    using type = long;
    static type run(const char *p, char** ppend) {
        return ::std::strtoll(p, ppend, 0);
    }
};

template<typename T>
struct floating_point_parse_helper;

template<>
struct floating_point_parse_helper<float> {
    using type = float;
    static type run(const char *p, char **pend) {
        return ::std::strtof(p, pend);
    }
};

template<>
struct floating_point_parse_helper<double> {
    using type = double;
    static type run(const char *p, char **pend) {
        return ::std::strtod(p, pend);
    }
};

template<>
struct floating_point_parse_helper<long double> {
    using type = long double;
    static type run(const char *p, char **pend) {
        return ::std::strtold(p, pend);
    }
};

struct bool_parse_helper {
    using type = bool;
    static type run(const char *p, char **pend) {
        // locate the begin
        const char *p0 = p;
        while (*p0 && chars::is_space(*p0)) p0++;

        // empty
        if (!(*p0)) {
            *pend = const_cast<char*>(p);
            return false;
        }

        // locate the word end
        const char *p1 = p0 + 1;
        while (*p1 && !chars::is_space(*p1)) p1++;

        // single non-space character
        if (p1 == p0 + 1) {
            char c = *p0;
            switch (c) {
                case '0':
                case 'F':
                case 'f':
                    *pend = const_cast<char*>(p1);
                    return false;
                case '1':
                case 'T':
                case 't':
                    *pend = const_cast<char*>(p1);
                    return true;
            }
            *pend = const_cast<char*>(p);
            return false;
        }

        // multi-characters
        size_t len = static_cast<size_t>(p1 - p0);
        if (len == 4) {
            if (_icmp<4>(p0, "true")) {
                *pend = const_cast<char*>(p1);
                return true;
            }
        } else if (len == 5) {
            if (_icmp<5>(p0, "false")) {
                *pend = const_cast<char*>(p1);
                return false;
            }
        }

        *pend = const_cast<char*>(p);
        return false;
    }
};

template<typename T>
using default_parse_helper_of =
        conditional_t<::std::is_same<T, bool>::value,
            bool_parse_helper,
        conditional_t<::std::is_integral<T>::value,
            integer_parse_helper<(sizeof(T) > sizeof(long))>,
            floating_point_parse_helper<T>
        >>;

} // end namespace details

// try_parse function for arithmetic types

template<typename T, typename Traits>
inline enable_if_t<::std::is_arithmetic<T>::value, bool>
try_parse(basic_string_view<char, Traits> sv, T& x) {
    using helper = details::default_parse_helper_of<T>;
    char *pend;
    typename helper::type _x = helper::run(sv.begin(), &pend);
    if (details::is_valid_parse_end(sv, pend)) {
        x = static_cast<T>(_x);
        return true;
    } else {
        return false;
    }
}

template<typename T>
inline enable_if_t<::std::is_arithmetic<T>::value, bool>
try_parse(const char *sz, T& x) {
    using helper = details::default_parse_helper_of<T>;
    char *pend;
    typename helper::type _x = helper::run(sz, &pend);
    if (details::is_valid_parse_end(sz, pend)) {
        x = static_cast<T>(_x);
        return true;
    } else {
        return false;
    }
}

template<typename T, typename Traits, typename Allocator>
inline enable_if_t<::std::is_arithmetic<T>::value, bool>
try_parse(const std::basic_string<char, Traits, Allocator>& str, T& x) {
    return try_parse(view(str), x);
}


//===============================================
//
//   Tokenization
//
//===============================================

namespace details {

template<typename charT, typename Traits=::std::char_traits<charT>>
struct is_in_cstr_ {
    const charT *cstr_;
    bool operator()(charT c) const noexcept {
        const charT *p = cstr_;
        while (*p && !Traits::eq(c, *p)) p++;
        return static_cast<bool>(*p);
    }
};

template<typename charT, typename Traits=::std::char_traits<charT>>
struct is_eq_char_ {
    charT c_;
    constexpr bool operator()(charT c) const noexcept {
        return Traits::eq(c, c_);
    }
};

template<typename charT, typename Pred, typename F>
void foreach_token_of_(const charT *str, Pred is_delim, F&& f) {
    // skip leading delimiters
    const charT *p = str;
    while (*p && is_delim(*p)) p++;

    // for each token
    while (*p) {
        const charT *q = p + 1;
        while (*q && !is_delim(*q)) q++;
        size_t tk_len = static_cast<size_t>(q - p);
        if (!(f(p, tk_len))) break;

        // skip delimiters
        p = q;
        while (*p && is_delim(*p)) p++;
    }
}

template<typename charT, typename Traits, typename Pred, typename F>
void foreach_token_of_(basic_string_view<charT, Traits> sv, Pred is_delim, F&& f) {
    // skip leading delimiters
    const charT *p = sv.data();
    const charT *pend = p + sv.size();

    while (p != pend && is_delim(*p)) p++;

    // for each token
    while (p != pend) {
        const charT *q = p + 1;
        while (q != pend && !is_delim(*q)) q++;
        size_t tk_len = static_cast<size_t>(q - p);
        if (!(f(p, tk_len))) break;

        // skip delimiters
        p = q;
        while (p != pend && is_delim(*p)) p++;
    }
}

};


template<typename charT, typename F>
inline void foreach_token_of(const charT *str,  charT delim, F&& f) {
    details::foreach_token_of_(str,
        details::is_eq_char_<charT>{delim}, ::std::forward<F>(f));
}

template<typename charT, typename F>
inline void foreach_token_of(const charT *str, const charT *delims, F&& f) {
    details::foreach_token_of_(str,
        details::is_in_cstr_<charT>{delims}, ::std::forward<F>(f));
}

template<typename charT, typename Traits, typename F>
inline void foreach_token_of(basic_string_view<charT, Traits> sv, charT delim, F&& f) {
    details::foreach_token_of_(sv,
        details::is_eq_char_<charT, Traits>{delim}, ::std::forward<F>(f));
}

template<typename charT, typename Traits, typename F>
inline void foreach_token_of(basic_string_view<charT, Traits> sv, const charT *delims, F&& f) {
    details::foreach_token_of_(sv,
        details::is_in_cstr_<charT, Traits>{delims}, ::std::forward<F>(f));
}

template<typename charT, typename Traits, typename Allocator, typename F>
inline void foreach_token_of(::std::basic_string<charT, Traits, Allocator>& str, charT delim, F&& f) {
    foreach_token_of(view(str), delim, ::std::forward<F>(f));
}

template<typename charT, typename Traits, typename Allocator, typename F>
inline void foreach_token_of(::std::basic_string<charT, Traits, Allocator>& str, const charT* delims, F&& f) {
    foreach_token_of(view(str), delims, ::std::forward<F>(f));
}


}

#endif
