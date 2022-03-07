/**
 * @file optional.hpp
 *
 * The optional class.
 *
 * @note The implementation is adapted from libcxx.
 */

#ifndef CLUE_OPTIONAL__
#define CLUE_OPTIONAL__

#include <clue/container_common.hpp>

namespace clue {

template <typename T> class optional;

struct in_place_t{};
constexpr in_place_t in_place{};

struct nullopt_t {
    explicit constexpr nullopt_t(int) noexcept {}
};
constexpr nullopt_t nullopt{0};

class bad_optional_access : public ::std::logic_error {
public:
	bad_optional_access() :
        ::std::logic_error("Bad optional Access") {}

    virtual ~bad_optional_access() noexcept {}
};


namespace details {

template<typename T, bool = ::std::is_trivially_destructible<T>::value>
class optional_base
{
protected:
    typedef T value_type;
    union
    {
        char nil_;
        value_type val_;
    };
    bool init_ = false;

    ~optional_base() {
        if (init_)
            val_.~value_type();
    }

    constexpr optional_base() noexcept : nil_('\0') {}

    constexpr optional_base(const value_type& v) :
        val_(v), init_(true) {}

    constexpr optional_base(value_type&& v) :
        val_(::std::move(v)), init_(true) {}

    template <class... Args>
    constexpr explicit optional_base(in_place_t, Args&&... args) :
        val_(::std::forward<Args>(args)...), init_(true) {}

    optional_base(const optional_base& x) :
        init_(x.init_) {
        if (init_)
            ::new(::std::addressof(val_)) value_type(x.val_);
    }

    optional_base(optional_base&& x)
        noexcept(::std::is_nothrow_move_constructible<value_type>::value) :
        init_(x.init_) {
        if (init_)
            ::new(::std::addressof(val_)) value_type(::std::move(x.val_));
    }
};


template<typename T>
class optional_base<T, true>
{
protected:
    typedef T value_type;
    union
    {
        char nil_;
        value_type val_;
    };
    bool init_ = false;

    constexpr optional_base() noexcept: nil_('\0') {}

    constexpr optional_base(const value_type& v) :
        val_(v), init_(true) {}

    constexpr optional_base(value_type&& v) :
        val_(::std::move(v)), init_(true) {}

    template <class... Args>
    constexpr explicit optional_base(in_place_t, Args&&... args) :
        val_(::std::forward<Args>(args)...), init_(true) {}

    optional_base(const optional_base& x) : init_(x.init_) {
        if (init_)
        ::new(::std::addressof(val_)) value_type(x.val_);
    }

    optional_base(optional_base&& x)
        noexcept(::std::is_nothrow_move_constructible<value_type>::value) :
        init_(x.init_) {
        if (init_)
        ::new(::std::addressof(val_)) value_type(::std::move(x.val_));
    }
};

} // end namespace details


template <class T>
class optional : private details::optional_base<T> {

    static_assert(!::std::is_reference<T>::value,
        "optional<T> with T being a reference type is ill-formed.");
    static_assert(!::std::is_same<typename ::std::remove_cv<T>::type, in_place_t>::value,
        "optional<T> with T being an in_place_t type is ill-formed.");
    static_assert(!::std::is_same<typename ::std::remove_cv<T>::type, nullopt_t>::value,
        "optional<T> with T being a nullopt_t type is ill-formed.");
    static_assert(::std::is_object<T>::value,
        "optional<T> with a non-object type T is undefined behavior.");
    static_assert(::std::is_nothrow_destructible<T>::value,
        "optional<T> where T is not nothrow-destructible is undefined behavior.");

    typedef details::optional_base<T> __base;

public:
    // member types

    typedef T value_type;


    // constructors

    constexpr optional() noexcept {}
    constexpr optional(nullopt_t) noexcept {}

    optional(const optional&) = default;

    optional(optional&&) = default;

    constexpr optional(const value_type& v)
        : __base(v) {}

    constexpr optional(value_type&& v)
        : __base(::std::move(v)) {}

    template <class... Args,
              class = typename ::std::enable_if<
                  ::std::is_constructible<value_type, Args...>::value>::type>
    constexpr explicit optional(in_place_t ip, Args&&... args)
        : __base(ip, ::std::forward<Args>(args)...) {}

    template <class U, class... Args,
              class = typename ::std::enable_if<
                  ::std::is_constructible<value_type, ::std::initializer_list<U>&, Args...>::value>::type>
    constexpr explicit optional(in_place_t ip, ::std::initializer_list<U> ilist, Args&&... args)
        : __base(ip, ilist, ::std::forward<Args>(args)...) {}


    // destructor

    ~optional() = default;


    // assignment

    optional& operator=(nullopt_t) noexcept {
        if (this->init_)
        {
            this->val_.~value_type();
            this->init_ = false;
        }
        return *this;
    }


    optional& operator=(const optional& rhs) {
        if (this->init_ == rhs.init_) {
            if (this->init_)
                this->val_ = rhs.val_;
        } else {
            if (this->init_)
                this->val_.~value_type();
            else
                ::new(::std::addressof(this->val_)) value_type(rhs.val_);
            this->init_ = rhs.init_;
        }
        return *this;
    }

    optional& operator= (optional&& rhs)
        noexcept(::std::is_nothrow_move_assignable<value_type>::value &&
                 ::std::is_nothrow_move_constructible<value_type>::value) {

        if (this->init_ == rhs.init_) {
            if (this->init_)
                this->val_ = ::std::move(rhs.val_);
        } else {
            if (this->init_)
                this->val_.~value_type();
            else
                ::new(::std::addressof(this->val_)) value_type(::std::move(rhs.val_));
            this->init_ = rhs.init_;
        }
        return *this;
    }

    template <class U,
              class = typename ::std::enable_if<
                  ::std::is_same<typename ::std::remove_reference<U>::type, value_type>::value &&
                  ::std::is_constructible<value_type, U>::value &&
                  ::std::is_assignable<value_type&, U>::value>::type>
    optional& operator=(U&& v) {
        if (this->init_) {
            this->val_ = ::std::forward<U>(v);
        } else {
            ::new(::std::addressof(this->val_)) value_type(::std::forward<U>(v));
            this->init_ = true;
        }
        return *this;
    }


    // swap

    void swap(optional& rhs)
        noexcept(::std::is_nothrow_move_constructible<value_type>::value) {

        using ::std::swap;
        if (this->init_ == rhs.init_) {
            if (this->init_)
                swap(this->val_, rhs.val_);
        } else {
            if (this->init_) {
                ::new(::std::addressof(rhs.val_)) value_type(::std::move(this->val_));
                this->val_.~value_type();
            } else {
                ::new(::std::addressof(this->val_)) value_type(::std::move(rhs.val_));
                rhs.val_.~value_type();
            }
            swap(this->init_, rhs.init_);
        }
    }


    // emplace

    template <class... Args,
              class = typename ::std::enable_if<
                  ::std::is_constructible<value_type, Args...>::value>::type>
    void emplace(Args&&... args) {
        *this = nullopt;
        ::new(::std::addressof(this->val_)) value_type(::std::forward<Args>(args)...);
        this->init_ = true;
    }

    template <class U, class... Args,
              class = typename ::std::enable_if<
              ::std::is_constructible<value_type, ::std::initializer_list<U>&, Args...>::value>::type>
    void emplace(::std::initializer_list<U> initl, Args&&... args) {
        *this = nullopt;
        ::new(::std::addressof(this->val_)) value_type(initl, ::std::forward<Args>(args)...);
        this->init_ = true;
    }


    // operator-> and operator*

    constexpr value_type const* operator->() const {
        return ::std::addressof(this->val_);
    }

    value_type* operator->() {
        return ::std::addressof(this->val_);
    }

    constexpr const value_type& operator*() const {
        return this->val_;
    }

    value_type& operator*() {
        return this->val_;
    }


    // operator bool

    constexpr explicit operator bool() const noexcept {
        return this->init_;
    }


    // value and value_or

    constexpr value_type const& value() const {
        return this->init_ ? this->val_ : (throw bad_optional_access(), value_type());
    }

    value_type& value() {
        if (!this->init_)
            throw bad_optional_access();
        return this->val_;
    }

    template <typename U>
    constexpr value_type value_or(U&& v) const& {
        static_assert(::std::is_copy_constructible<value_type>::value,
                      "optional<T>::value_or: T must be copy constructible");
        static_assert(::std::is_convertible<U, value_type>::value,
                      "optional<T>::value_or: U must be convertible to T");
        return this->init_ ?
            this->val_ : static_cast<value_type>(::std::forward<U>(v));
    }

    template <class U>
    value_type value_or(U&& v) && {
        static_assert(::std::is_move_constructible<value_type>::value,
                      "optional<T>::value_or: T must be move constructible");
        static_assert(::std::is_convertible<U, value_type>::value,
                      "optional<T>::value_or: U must be convertible to T");
        return this->init_ ?
            ::std::move(this->val_) : static_cast<value_type>(::std::forward<U>(v));
    }

}; // end class optional


// Comparison with optional

template <class T>
inline constexpr bool operator==(const optional<T>& x, const optional<T>& y) {
    return static_cast<bool>(x) != static_cast<bool>(y) ? false :
           (static_cast<bool>(x) ? (*x == *y) : true);
}

template <class T>
inline constexpr bool operator!=(const optional<T>& x, const optional<T>& y) {
    return !(x == y);
}

template <class T>
inline constexpr bool operator<(const optional<T>& x, const optional<T>& y) {
    return !static_cast<bool>(y) ? false :
           (static_cast<bool>(x) ? (*x < *y) : true);
}

template <class T>
inline constexpr bool operator>(const optional<T>& x, const optional<T>& y) {
    return y < x;
}

template <class T>
inline constexpr bool operator<=(const optional<T>& x, const optional<T>& y) {
    return !(y < x);
}

template <class T>
inline constexpr bool operator>=(const optional<T>& x, const optional<T>& y) {
    return !(x < y);
}


// Comparisons with nullopt
template <class T>
inline constexpr bool operator==(const optional<T>& x, nullopt_t) noexcept {
    return !static_cast<bool>(x);
}

template <class T>
inline constexpr bool operator==(nullopt_t, const optional<T>& x) noexcept {
    return !static_cast<bool>(x);
}

template <class T>
inline constexpr bool operator!=(const optional<T>& x, nullopt_t) noexcept {
    return static_cast<bool>(x);
}

template <class T>
inline constexpr bool operator!=(nullopt_t, const optional<T>& x) noexcept {
    return static_cast<bool>(x);
}

template <class T>
inline constexpr bool operator<(const optional<T>&, nullopt_t) noexcept {
    return false;
}

template <class T>
inline constexpr bool operator<(nullopt_t, const optional<T>& x) noexcept {
    return static_cast<bool>(x);
}

template <class T>
inline constexpr bool operator<=(const optional<T>& x, nullopt_t) noexcept {
    return !static_cast<bool>(x);
}

template <class T>
inline constexpr bool operator<=(nullopt_t, const optional<T>& x) noexcept {
    return true;
}

template <class T>
inline constexpr bool operator>(const optional<T>& x, nullopt_t) noexcept {
    return static_cast<bool>(x);
}

template <class T>
inline constexpr bool operator>(nullopt_t, const optional<T>& x) noexcept {
    return false;
}

template <class T>
inline constexpr bool operator>=(const optional<T>&, nullopt_t) noexcept {
    return true;
}

template <class T>
inline constexpr bool operator>=(nullopt_t, const optional<T>& x) noexcept {
    return !static_cast<bool>(x);
}

// Comparisons with T

template <class T>
inline constexpr bool operator==(const optional<T>& x, const T& v) {
    return static_cast<bool>(x) ? *x == v : false;
}

template <class T>
inline constexpr bool operator==(const T& v, const optional<T>& x) {
    return static_cast<bool>(x) ? *x == v : false;
}

template <class T>
inline constexpr bool operator!=(const optional<T>& x, const T& v)
{
    return static_cast<bool>(x) ? !(*x == v) : true;
}

template <class T>
inline constexpr bool operator!=(const T& v, const optional<T>& x) {
    return static_cast<bool>(x) ? !(*x == v) : true;
}

template <class T>
inline constexpr bool operator<(const optional<T>& x, const T& v) {
    return static_cast<bool>(x) ? (*x < v) : true;
}

template <class T>
inline constexpr bool operator<(const T& v, const optional<T>& x) {
    return static_cast<bool>(x) ? (v < *x) : false;
}

template <class T>
inline constexpr bool operator<=(const optional<T>& x, const T& v) {
    return !(x > v);
}

template <class T>
inline constexpr bool operator<=(const T& v, const optional<T>& x) {
    return !(v > x);
}

template <class T>
inline constexpr bool operator>(const optional<T>& x, const T& v) {
    return static_cast<bool>(x) ? v < x : false;
}

template <class T>
inline constexpr bool operator>(const T& v, const optional<T>& x) {
    return static_cast<bool>(x) ? x < v : true;
}

template <class T>
inline constexpr bool operator>=(const optional<T>& x, const T& v) {
    return !(x < v);
}

template <class T>
inline constexpr bool operator>=(const T& v, const optional<T>& x) {
    return !(v < x);
}


// make_optional

template <class T>
inline constexpr optional<typename ::std::decay<T>::type>
make_optional(T&& v) {
    return optional<typename ::std::decay<T>::type>(::std::forward<T>(v));
}

// external swap

template <class _Tp>
inline void swap(optional<_Tp>& x, optional<_Tp>& y) noexcept(noexcept(x.swap(y))) {
    x.swap(y);
}

} // end namespace clue


namespace std {

// specialize std::hash
template <class T>
struct hash<clue::optional<T> > {
    typedef clue::optional<T> argument_type;
    typedef size_t result_type;

    result_type operator()(const argument_type& arg) const noexcept {
        return static_cast<bool>(arg) ? hash<T>()(*arg) : 0;
    }
};

}


#endif
