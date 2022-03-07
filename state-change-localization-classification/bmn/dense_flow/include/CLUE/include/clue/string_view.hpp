/**
 * @file string_view.hpp
 *
 * The classes for representing string views.
 *
 * @note The implementation is adapted from libcxx.
 */

#ifndef CLUE_STRING_VIEW__
#define CLUE_STRING_VIEW__

#include <clue/container_common.hpp>
#include <string>
#include <ostream>


namespace clue {

// declarations

template<class charT, class Traits = ::std::char_traits<charT> >
class basic_string_view;

typedef basic_string_view<char>     string_view;
typedef basic_string_view<wchar_t>  wstring_view;
typedef basic_string_view<char16_t> u16string_view;
typedef basic_string_view<char32_t> u32string_view;


// class basic_string_view

template<class charT, class Traits>
class basic_string_view {
public:
    // types
    typedef Traits traits_type;
    typedef charT value_type;
    typedef const charT* pointer;
    typedef const charT* const_pointer;
    typedef const charT& reference;
    typedef const charT& const_reference;
    typedef const_pointer  const_iterator;
    typedef const_iterator iterator;
    typedef ::std::reverse_iterator<const_iterator> const_reverse_iterator;
    typedef const_reverse_iterator reverse_iterator;
    typedef ::std::size_t size_type;
    typedef ::std::ptrdiff_t difference_type;
    static constexpr const size_type npos = -1; // size_type(-1);

private:
    const value_type* data_;
    size_type len_;

public:
    // construct/copy

    constexpr basic_string_view() noexcept :
        data_(nullptr), len_(0) {}

    constexpr basic_string_view(const basic_string_view&) noexcept = default;

    template<class Allocator>
    basic_string_view(const ::std::basic_string<charT, Traits, Allocator>& s) noexcept
        : data_ (s.data()), len_(s.size()) {}

    constexpr basic_string_view(const charT* s, size_type count) noexcept
        : data_(s), len_(count) { }

    constexpr basic_string_view(const charT* s) noexcept
        : data_(s), len_(Traits::length(s)) {}


    // assignment

    basic_string_view& operator=(const basic_string_view&) noexcept = default;


    // iterators

    constexpr const_iterator begin()  const noexcept { return cbegin(); }
    constexpr const_iterator end()    const noexcept { return cend(); }
    constexpr const_iterator cbegin() const noexcept { return data_; }
    constexpr const_iterator cend()   const noexcept { return data_ + len_; }

    constexpr const_reverse_iterator rbegin()   const noexcept { return crbegin(); }
    constexpr const_reverse_iterator rend()     const noexcept { return crend(); }
    constexpr const_reverse_iterator crbegin()  const noexcept { return const_reverse_iterator(cend()); }
    constexpr const_reverse_iterator crend()    const noexcept { return const_reverse_iterator(cbegin()); }


    // element access

    constexpr const_reference operator[](size_type pos) const noexcept {
        return data_[pos];
    }

    constexpr const_reference at(size_type pos) const {
        return pos < len_ ? data_[pos] :
            (throw ::std::out_of_range("basic_string_view::at"), data_[0]);
    }

    constexpr const_reference front() const {
        return data_[0];
    }

    constexpr const_reference back() const {
        return data_[len_-1];
    }

    constexpr const_pointer data() const noexcept {
        return data_;
    }


    // capacity

    constexpr bool      empty()    const noexcept { return len_ == 0; }
    constexpr size_type size()     const noexcept { return len_; }
    constexpr size_type length()   const noexcept { return len_; }
    constexpr size_type max_size() const noexcept { return ::std::numeric_limits<size_type>::max(); }


    // modifiers

    void clear() noexcept {
        data_ = nullptr;
        len_ = 0;
    }

    void remove_prefix(size_type n) noexcept {
        data_ += n;
        len_ -= n;
    }

    void remove_suffix(size_type n) noexcept {
        len_ -= n;
    }

    void swap(basic_string_view& other) noexcept {
        ::std::swap(data_, other.data_);
        ::std::swap(len_, other.len_);
    }


    // conversion & copy

    template<class Allocator>
    explicit operator ::std::basic_string<charT, Traits, Allocator>() const {
        return ::std::basic_string<charT, Traits, Allocator>(begin(), end());
    }

    template<class Allocator = ::std::allocator<charT> >
    ::std::basic_string<charT, Traits, Allocator>
    to_string(const Allocator& a = Allocator()) const {
        return ::std::basic_string<charT, Traits, Allocator>(begin(), end(), a);
    }

    size_type copy(charT* s, size_type n, size_type pos = 0) const {
        if (pos > len_)
            throw ::std::out_of_range("basic_string_view::copy");
        size_type rlen = (::std::min)(n, len_ - pos);
        ::std::copy_n(data_ + pos, rlen, s);
        return rlen;
    }


    // substr

    constexpr basic_string_view substr(size_type pos = 0, size_type n = npos) const {
        return pos > len_ ?
            throw ::std::out_of_range("basic_string_view::substr") :
            basic_string_view(data_ + pos, (::std::min)(n, len_ - pos));
    }


    // compare

    int compare(basic_string_view sv) const noexcept {
        size_type rlen = (::std::min)( size(), sv.size());
        int rval = Traits::compare(data(), sv.data(), rlen);
        return rval == 0 ?
            (size() == sv.size() ? 0 : (size() < sv.size() ? -1 : 1)) :
            rval;
    }

    int compare(size_type pos1, size_type n1, basic_string_view sv) const {
        return substr(pos1, n1).compare(sv);
    }

    int compare(size_type pos1, size_type n1,
                basic_string_view sv, size_type pos2, size_type n2) const {
        return substr(pos1, n1).compare(sv.substr(pos2, n2));
    }

    int compare(const charT* s) const {
        return compare(basic_string_view(s));
    }

    int compare(size_type pos1, size_type n1, const charT* s) const {
        return substr(pos1, n1).compare(basic_string_view(s));
    }

    int compare(size_type pos1, size_type n1, const charT* s, size_type n2) const {
        return substr(pos1, n1).compare(basic_string_view(s, n2));
    }

public:
    // find

    size_type find(charT c, size_type pos = 0) const noexcept {
        return find_first_of(c, pos);
    }

    size_type find(const charT* s, size_type pos, size_type n) const noexcept {
        if (n == 0 || pos + n > size()) return npos;
        const_iterator r = ::std::search(cbegin() + pos, cend(), s, s + n, Traits::eq);
        return get_pos_(r);
    }

    size_type find(basic_string_view s, size_type pos = 0) const noexcept {
        return find(s.data(), pos, s.size());
    }

    size_type find(const charT* s, size_type pos = 0) const noexcept {
        return find(s, pos, Traits::length(s));
    }

    // rfind

    size_type rfind(charT c, size_type pos = npos) const noexcept {
        return find_last_of(c, pos);
    }

    size_type rfind(const charT* s, size_type pos, size_type n) const noexcept {
        pos = pos < size() ? (size() - pos > n ? pos + n : size()) : size();
        const charT *r = ::std::find_end(data(), data() + pos, s, s + n, Traits::eq);
        return (n > 0 && r == data() + pos) ? npos : static_cast<size_type>(r - data());
    }

    size_type rfind(basic_string_view s, size_type pos = npos) const noexcept {
        return rfind(s.data(), pos, s.size());
    }

    size_type rfind(const charT* s, size_type pos = npos) const noexcept {
        return rfind(s, pos, Traits::length(s));
    }

    // find_first_of

    size_type find_first_of(charT c, size_type pos = 0) const noexcept {
        return find_if_(eq_(c), pos);
    }

    size_type find_first_of(const charT* s, size_type pos, size_type n) const noexcept {
        return find_if_(in_(s, n), pos);
    }

    size_type find_first_of(basic_string_view s, size_type pos = 0) const noexcept {
        return find_if_(in_(s), pos);
    }

    size_type find_first_of(const charT* s, size_type pos = 0) const noexcept {
        return find_if_(in_(s), pos);
    }

    // find_last_of

    size_type find_last_of(charT c, size_type pos = npos) const noexcept {
        return rfind_if_(eq_(c), pos);
    }

    size_type find_last_of(const charT* s, size_type pos, size_type n) const noexcept {
        return rfind_if_(in_(s, n), pos);
    }

    size_type find_last_of(basic_string_view s, size_type pos = npos) const noexcept {
        return rfind_if_(in_(s), pos);
    }

    size_type find_last_of(const charT* s, size_type pos = npos) const noexcept {
        return rfind_if_(in_(s), pos);
    }

    // find_first_not_of

    size_type find_first_not_of(charT c, size_type pos = 0) const noexcept {
        return find_if_not_(eq_(c), pos);
    }

    size_type find_first_not_of(const charT* s, size_type pos, size_type n) const noexcept {
        return find_if_not_(in_(s, n), pos);
    }

    size_type find_first_not_of(basic_string_view s, size_type pos = 0) const noexcept {
        return find_if_not_(in_(s), pos);
    }

    size_type find_first_not_of(const charT* s, size_type pos = 0) const noexcept {
        return find_if_not_(in_(s), pos);
    }

    // find_last_not_of

    size_type find_last_not_of(charT c, size_type pos = npos) const noexcept {
        return rfind_if_not_(eq_(c), pos);
    }

    size_type find_last_not_of(const charT* s, size_type pos, size_type n) const noexcept {
        return rfind_if_not_(in_(s, n), pos);
    }

    size_type find_last_not_of(basic_string_view s, size_type pos = npos) const noexcept {
        return rfind_if_not_(in_(s), pos);
    }

    size_type find_last_not_of(const charT* s, size_type pos = npos) const noexcept {
        return rfind_if_not_(in_(s), pos);
    }

private:
    // facilities to support the implementation of find functions

    class eq_pred {
    private:
        charT c_;
    public:
        constexpr eq_pred(charT c) noexcept : c_(c) { }
        bool operator()(charT c) const noexcept {
            return Traits::eq(c, c_);
        }
    };

    class in_pred {
    private:
        const charT *s_;
    public:
        constexpr in_pred(const charT* s) noexcept : s_(s) { }
        bool operator()(charT c) const noexcept {
            const charT *p = s_;
            while (*p && !Traits::eq(c, *p)) ++p;
            return static_cast<bool>(*p);
        }
    };

    class in_rgn_pred {
    private:
        const charT *s_;
        const charT *se_;
    public:
        constexpr in_rgn_pred(const char *s, size_type n) noexcept :
            s_(s), se_(s + n) {}
        bool operator()(charT c) const noexcept {
            const charT *p = s_;
            while (p != se_ && !Traits::eq(c, *p)) ++p;
            return p != se_;
        }
    };

    static constexpr eq_pred eq_(charT c) noexcept {
        return eq_pred(c);
    }

    static constexpr in_pred in_(const charT *s) noexcept {
        return in_pred(s);
    }

    static constexpr in_rgn_pred in_(const char *s, size_type n) noexcept {
        return in_rgn_pred(s, n);
    }

    static constexpr in_rgn_pred in_(basic_string_view s) noexcept {
        return in_rgn_pred(s.data(), s.size());
    }

    constexpr size_type get_pos_(const_iterator it) const noexcept {
        return it == cend() ? npos : static_cast<size_type>(it - cbegin());
    }

    template<typename Pred>
    size_type find_if_(Pred&& pred, size_type pos) const noexcept {
        if (pos >= size()) return npos;
        const_iterator r = ::std::find_if(cbegin() + pos, cend(), pred);
        return get_pos_(r);
    }

    template<typename Pred>
    size_type find_if_not_(Pred&& pred, size_type pos) const noexcept {
        if (pos >= size()) return npos;
        const_iterator r = ::std::find_if_not(cbegin() + pos, cend(), pred);
        return get_pos_(r);
    }

    template<typename Pred>
    size_type rfind_if_(Pred&& pred, size_type pos) const noexcept {
        pos = pos < size() ? pos + 1 : size();
        for (const charT* ps = data() + pos; ps != data();) {
            if (pred(*(--ps))) {
                return static_cast<size_type>(ps - data());
            }
        }
        return npos;
    }

    template<typename Pred>
    size_type rfind_if_not_(Pred&& pred, size_type pos) const noexcept {
        pos = pos < size() ? pos + 1 : size();
        for (const charT* ps = data() + pos; ps != data();) {
            if (!pred(*--ps)) {
                return static_cast<size_type>(ps - data());
            }
        }
        return npos;
    }

}; // end class basic_string_view

// External swap

template<class charT, class Traits>
inline void swap(basic_string_view<charT, Traits>& a, basic_string_view<charT, Traits>& b) noexcept {
    a.swap(b);
}

// Comparison

template<class charT, class Traits>
inline bool operator==(basic_string_view<charT, Traits> lhs, basic_string_view<charT, Traits> rhs) noexcept {
    if (lhs.size() != rhs.size()) return false;
    return lhs.compare(rhs) == 0;
}

template<class charT, class Traits>
inline bool operator!=(basic_string_view<charT, Traits> lhs, basic_string_view<charT, Traits> rhs) noexcept {
    return !(lhs == rhs);
}

template<class charT, class Traits>
inline bool operator< (basic_string_view<charT, Traits> lhs, basic_string_view<charT, Traits> rhs) noexcept {
    return lhs.compare(rhs) < 0;
}

template<class charT, class Traits>
inline bool operator> (basic_string_view<charT, Traits> lhs, basic_string_view<charT, Traits> rhs) noexcept {
    return lhs.compare(rhs) > 0;
}

template<class charT, class Traits>
inline bool operator<=(basic_string_view<charT, Traits> lhs, basic_string_view<charT, Traits> rhs) noexcept {
    return lhs.compare(rhs) <= 0;
}

template<class charT, class Traits>
inline bool operator>=(basic_string_view<charT, Traits> lhs, basic_string_view<charT, Traits> rhs) noexcept {
    return lhs.compare(rhs) >= 0;
}


// stream output

template<class charT, class Traits>
inline ::std::basic_ostream<charT, Traits>& operator<< (
        ::std::basic_ostream<charT, Traits>& os,
        basic_string_view<charT, Traits> sv) {
    try {
        typename ::std::basic_ostream<charT, Traits>::sentry s(os);
        if (s) {
            const ::std::size_t len = sv.size();

            ::std::ostreambuf_iterator<charT, Traits> it_out(os);
            size_t width = os.width();
            if (width > len) {
                if ((os.flags() & ::std::ios_base::adjustfield) == ::std::ios_base::left) {
                    ::std::copy(sv.begin(), sv.end(), it_out);
                    it_out = ::std::fill_n(it_out, width - len, os.fill());
                } else {
                    it_out = ::std::fill_n(it_out, width - len, os.fill());
                    ::std::copy(sv.begin(), sv.end(), it_out);
                }
            } else {
                ::std::copy(sv.begin(), sv.end(), it_out);
            }
        }
    }
    catch (...) {
        os.setstate(::std::ios_base::badbit | ::std::ios_base::failbit);
        throw;
    }
    return os;
}

}  // end namespace clue


namespace std {

template<class charT, class Traits>
struct hash<clue::basic_string_view<charT, Traits> >
    : public unary_function<clue::basic_string_view<charT, Traits>, size_t> {

    size_t operator()(const clue::basic_string_view<charT, Traits>& sv) const {
        // TODO: implement more efficient hash (that is also consistent with that of std::string
        return hash_(sv.to_string());
    }

private:
    hash<basic_string<charT, Traits>> hash_;
};

}


#endif
