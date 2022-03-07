#ifndef CLUE_VALUE_RANGE__
#define CLUE_VALUE_RANGE__

#include <clue/container_common.hpp>

namespace clue {


namespace details {

struct generic_minus {
    template<typename T>
    auto operator()(const T& x, const T& y) const -> decltype(x - y) {
        return x - y;
    }
};

template<typename T, bool=::std::is_unsigned<T>::value>
struct default_difference_helper {
    typedef typename ::std::make_signed<T>::type type;
};

template<typename T>
struct default_difference_helper<T, false> {
    typedef T type;
};

template<typename T>
struct is_valid_range_argtype {
    static constexpr bool value =
        ::std::is_object<T>::value &&
        !::std::is_const<T>::value &&
        !::std::is_volatile<T>::value &&
        ::std::is_copy_constructible<T>::value &&
        ::std::is_copy_assignable<T>::value &&
        ::std::is_nothrow_move_constructible<T>::value &&
        ::std::is_nothrow_move_assignable<T>::value;
};

};

template<typename T>
struct default_difference {
private:
    typedef typename ::std::result_of<details::generic_minus(T, T)>::type minus_ret_t;
public:
    typedef typename details::default_difference_helper<T>::type type;
};


// range traits

template<typename T, typename D>
struct value_range_traits {
    typedef D difference_type;

    static void increment(T& x) noexcept { ++x; }
    static void decrement(T& x) noexcept { --x; }
    static void increment(T& x, D d) noexcept { x += d; }
    static void decrement(T& x, D d) noexcept { x -= d; }

    constexpr static T next(T x) noexcept { return x + 1; }
    constexpr static T prev(T x) noexcept { return x - 1; }
    constexpr static T next(T x, D d) noexcept { return x + d; }
    constexpr static T prev(T x, D d) noexcept { return x - d; }

    constexpr static bool eq(T x, T y) noexcept { return x == y; }
    constexpr static bool lt(T x, T y) noexcept { return x <  y; }
    constexpr static bool le(T x, T y) noexcept { return x <= y; }

    constexpr static difference_type difference(T x, T y) noexcept {
        return x - y;
    }
};

namespace details {

template<typename T, typename Traits>
class value_range_iterator {
private:
    T v_;

public:
    typedef T value_type;
    typedef T reference;
    typedef const T* pointer;
    typedef typename Traits::difference_type difference_type;
    typedef ::std::random_access_iterator_tag iterator_category;

public:
    constexpr value_range_iterator(const T& v) :
        v_(v) {}

    // comparison

    constexpr bool operator <  (const value_range_iterator& r) const noexcept {
        return Traits::lt(v_, r.v_);
    }

    constexpr bool operator <= (const value_range_iterator& r) const noexcept {
        return Traits::le(v_, r.v_);
    }

    constexpr bool operator >  (const value_range_iterator& r) const noexcept {
        return Traits::lt(r.v_, v_);
    }

    constexpr bool operator >= (const value_range_iterator& r) const noexcept {
        return Traits::le(r.v_, v_);
    }

    constexpr bool operator == (const value_range_iterator& r) const noexcept {
        return Traits::eq(v_, r.v_);
    }

    constexpr bool operator != (const value_range_iterator& r) const noexcept {
        return !Traits::eq(v_, r.v_);
    }

    // dereference

    constexpr T operator* () const noexcept {
        return T(v_);
    }

    constexpr T operator[](difference_type n) const noexcept {
        return Traits::advance(v_, n);
    }

    // increment & decrement

    value_range_iterator& operator++() noexcept {
        Traits::increment(v_);
        return *this;
    }

    value_range_iterator& operator--() noexcept {
        Traits::decrement(v_);
        return *this;
    }

    value_range_iterator operator++(int) noexcept {
        T t(v_); Traits::increment(v_);
        return value_range_iterator(t);
    }

    value_range_iterator operator--(int) noexcept {
        T t(v_); Traits::decrement(v_);
        return value_range_iterator(t);
    }

    // arithmetics
    constexpr value_range_iterator operator + (difference_type n) const noexcept {
        return value_range_iterator(Traits::next(v_, n));
    }

    constexpr value_range_iterator operator - (difference_type n) const noexcept {
        return value_range_iterator(Traits::prev(v_, n));
    }

    value_range_iterator& operator += (difference_type n) noexcept {
        Traits::increment(v_, n);
        return *this;
    }

    value_range_iterator& operator -= (difference_type n) noexcept {
        Traits::decrement(v_, n);
        return *this;
    }

    constexpr difference_type operator - (value_range_iterator r) const noexcept {
        return Traits::difference(v_, r.v_);
    }
};


template<typename T, typename S, typename Traits>
class stepped_value_range_iterator {
private:
    T v_;
    S s_;

public:
    typedef T value_type;
    typedef T reference;
    typedef const T* pointer;
    typedef typename Traits::difference_type difference_type;
    typedef ::std::random_access_iterator_tag iterator_category;

public:
    constexpr stepped_value_range_iterator(const T& v, const S& s) :
        v_(v), s_(s) {}

    // comparison

    constexpr bool operator <  (const stepped_value_range_iterator& r) const noexcept {
        return Traits::lt(v_, r.v_);
    }

    constexpr bool operator <= (const stepped_value_range_iterator& r) const noexcept {
        return Traits::le(v_, r.v_);
    }

    constexpr bool operator >  (const stepped_value_range_iterator& r) const noexcept {
        return Traits::lt(r.v_, v_);
    }

    constexpr bool operator >= (const stepped_value_range_iterator& r) const noexcept {
        return Traits::le(r.v_, v_);
    }

    constexpr bool operator == (const stepped_value_range_iterator& r) const noexcept {
        return Traits::eq(v_, r.v_);
    }

    constexpr bool operator != (const stepped_value_range_iterator& r) const noexcept {
        return !Traits::eq(v_, r.v_);
    }

    // dereference

    constexpr T operator* () const noexcept {
        return T(v_);
    }

    constexpr T operator[](difference_type n) const noexcept {
        return Traits::advance(v_, step_(n));
    }

    // increment & decrement

    stepped_value_range_iterator& operator++() noexcept {
        Traits::increment(v_, step_());
        return *this;
    }

    stepped_value_range_iterator& operator--() noexcept {
        Traits::decrement(v_, step_());
        return *this;
    }

    stepped_value_range_iterator operator++(int) noexcept {
        T t(v_); Traits::increment(v_, step_());
        return stepped_value_range_iterator(t, s_);
    }

    stepped_value_range_iterator operator--(int) noexcept {
        T t(v_); Traits::decrement(v_, step_());
        return stepped_value_range_iterator(t, s_);
    }

    // arithmetics
    constexpr stepped_value_range_iterator operator + (difference_type n) const noexcept {
        return stepped_value_range_iterator(Traits::next(v_, step_(n)), s_);
    }

    constexpr stepped_value_range_iterator operator - (difference_type n) const noexcept {
        return stepped_value_range_iterator(Traits::prev(v_, step_(n)), s_);
    }

    stepped_value_range_iterator& operator += (difference_type n) noexcept {
        Traits::increment(v_, step_(n));
        return *this;
    }

    stepped_value_range_iterator& operator -= (difference_type n) noexcept {
        Traits::decrement(v_, step_(n));
        return *this;
    }

    constexpr difference_type operator - (stepped_value_range_iterator r) const noexcept {
        return Traits::difference(v_, r.v_) / step_();
    }

private:
    constexpr difference_type step_() const noexcept {
        return static_cast<difference_type>(s_);
    }

    constexpr difference_type step_(difference_type n) const noexcept {
        return static_cast<difference_type>(s_) * n;
    }
};


} // end namespace details


template<typename T,
         typename D=typename default_difference<T>::type,
         typename Traits=value_range_traits<T, D>>
class value_range {
    static_assert(details::is_valid_range_argtype<T>::value,
            "value_range<T>: T is not a valid type argument.");

public:
    // types
    typedef T value_type;
    typedef D difference_type;
    typedef Traits traits_type;
    typedef typename ::std::size_t size_type;
    typedef size_type step_type;

    typedef const T& reference;
    typedef const T& const_reference;
    typedef const T* pointer;
    typedef const T* const_pointer;

    typedef details::value_range_iterator<T, Traits> iterator;
    typedef iterator const_iterator;

private:
    T vbegin_;
    T vend_;

public:
    // constructor/copy/swap

    constexpr value_range(const T& vbegin, const T& vend) :
        vbegin_(vbegin), vend_(vend) {}

    constexpr value_range(const value_range&) = default;

    ~value_range() = default;

    value_range& operator=(const value_range&) = default;

    void swap(value_range& other) noexcept {
        using ::std::swap;
        swap(vbegin_, other.vbegin_);
        swap(vend_, other.vend_);
    }

    // properties

    constexpr size_type size() const noexcept {
        return static_cast<size_type>(Traits::difference(vend_, vbegin_));
    }

    constexpr step_type step() const noexcept {
        return 1;
    }

    constexpr bool empty() const noexcept {
        return Traits::eq(vbegin_, vend_);
    }

    // element access

    constexpr       T  front() const noexcept { return vbegin_; }
    constexpr       T  back()  const noexcept { return Traits::prev(vend_); }
    constexpr const T& begin_value() const noexcept { return vbegin_; }
    constexpr const T& end_value()   const noexcept { return vend_; }

    constexpr T operator[](size_type pos) const {
        return Traits::next(vbegin_, pos);
    }

    constexpr T at(size_type pos) const {
        return pos < size() ?
                vbegin_ + pos :
                (throw ::std::out_of_range("value_range::at"), vbegin_);
    }

    // iterators

    constexpr const_iterator begin()  const { return const_iterator(vbegin_); }
    constexpr const_iterator end()    const { return const_iterator(vend_); }
    constexpr const_iterator cbegin() const { return begin(); }
    constexpr const_iterator cend()   const { return end();   }

    // equality comparison

    constexpr bool operator==(const value_range& r) const noexcept {
        return Traits::eq(vbegin_, r.vbegin_) &&
               Traits::eq(vend_,  r.vend_);
    }

    constexpr bool operator!=(const value_range& r) const noexcept {
        return !(operator == (r));
    }

}; // end class value_range


template<typename T,
         typename S,
         typename D=typename default_difference<T>::type,
         typename Traits=value_range_traits<T, D>>
class stepped_value_range {
    static_assert(::std::is_integral<T>::value && ::std::is_integral<S>::value,
            "stepped_range<T, S>: only cases where both T and S are unsigned integers are supported.");
public:
    // types
    typedef T value_type;
    typedef S step_type;
    typedef D difference_type;
    typedef Traits traits_type;
    typedef typename ::std::size_t size_type;

    typedef const T& reference;
    typedef const T& const_reference;
    typedef const T* pointer;
    typedef const T* const_pointer;

    typedef details::stepped_value_range_iterator<T, S, Traits> iterator;
    typedef iterator const_iterator;

private:
    T vbegin_;
    T vend_;
    S step_;
    size_type len_;

public:
    // constructor/copy/swap

    stepped_value_range(const T& vbegin, const T& vend, const S& step) :
        vbegin_(vbegin),
        vend_(vend),
        step_(step),
        len_((vend - vbegin + (step - 1)) / step) {
        CLUE_ASSERT(step > 0);
    }

    constexpr stepped_value_range(const stepped_value_range&) = default;

    ~stepped_value_range() = default;

    stepped_value_range& operator=(const stepped_value_range&) = default;

    void swap(stepped_value_range& other) noexcept {
        using ::std::swap;
        swap(vbegin_, other.vbegin_);
        swap(vend_, other.vend_);
        swap(step_, other.step_);
        swap(len_, other.len_);
    }

    // properties

    constexpr size_type size() const noexcept {
        return len_;
    }

    constexpr step_type step() const noexcept {
        return step_;
    }

    constexpr bool empty() const noexcept {
        return len_ == 0;
    }

    // element access

    constexpr T front() const noexcept {
        return vbegin_;
    }

    constexpr T back()  const noexcept {
        return Traits::next(vbegin_,
            static_cast<difference_type>(step_ * (len_ - 1)));
    }

    constexpr const T& begin_value() const noexcept {
        return vbegin_;
    }

    constexpr const T& end_value() const noexcept {
        return vend_;
    }

    constexpr T operator[](size_type pos) const {
        return Traits::next(vbegin_, static_cast<difference_type>(step_ * pos));
    }

    constexpr T at(size_type pos) const {
        return pos < size() ?
                operator[](pos) :
                (throw ::std::out_of_range("value_range::at"), vbegin_);
    }

    // iterators

    constexpr const_iterator begin() const {
        return const_iterator(vbegin_, step_);
    }

    constexpr const_iterator end() const {
        return const_iterator(
            Traits::next(vbegin_, static_cast<difference_type>(step_ * len_)),
            step_);
    }

    constexpr const_iterator cbegin() const { return begin(); }
    constexpr const_iterator cend()   const { return end();   }

    // equality comparison

    constexpr bool operator==(const stepped_value_range& r) const noexcept {
        return Traits::eq(vbegin_, r.vbegin_) &&
               Traits::eq(step_,  r.step_) &&
               Traits::eq(len_, r.len_);
    }

    constexpr bool operator!=(const stepped_value_range& r) const noexcept {
        return !(operator == (r));
    }

}; // end class stepped_value_range


template<typename T>
constexpr value_range<T> vrange(const T& u) {
    return value_range<T>(static_cast<T>(0), u);
}

template<typename T>
constexpr value_range<T> vrange(const T& a, const T& b) {
    return value_range<T>(a, b);
}

template<typename T, typename Traits>
inline void swap(value_range<T,Traits>& lhs, value_range<T,Traits>& rhs) {
    lhs.swap(rhs);
}

template<class Container>
inline value_range<typename Container::size_type> indices(const Container& c) {
    return value_range<typename Container::size_type>(0, c.size());
}


}

#endif
