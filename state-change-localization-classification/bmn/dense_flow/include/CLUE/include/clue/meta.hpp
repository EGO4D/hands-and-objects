/**
 * @file meta.hpp
 *
 * Meta-programming facilities.
 */

#ifndef CLUE_META__
#define CLUE_META__

#include <clue/type_traits.hpp>

namespace clue {

namespace meta {

using ::std::size_t;
using ::std::integral_constant;

//===============================================
//
//   types
//
//===============================================

template<typename T>
struct type_ {
    using type = T;
};

template<typename A>
using get_type = typename A::type;

struct nil_{};

template<bool V>
using bool_ = integral_constant<bool, V>;

using true_  = bool_<true>;
using false_ = bool_<false>;

template<char  V> using char_  = integral_constant<char,  V>;
template<int   V> using int_   = integral_constant<int,   V>;
template<long  V> using long_  = integral_constant<long,  V>;
template<short V> using short_ = integral_constant<short, V>;

template<unsigned char  V> using uchar_  = integral_constant<unsigned char,  V>;
template<unsigned int   V> using uint_   = integral_constant<unsigned int,   V>;
template<unsigned long  V> using ulong_  = integral_constant<unsigned long,  V>;
template<unsigned short V> using ushort_ = integral_constant<unsigned short, V>;

template<size_t V> using size_ = integral_constant<size_t, V>;

template<class A>
using value_type_of = typename A::value_type;


// pair

template<typename T1, typename T2>
struct pair_ {
    using first_type = T1;
    using second_type = T2;
};

template<class A> struct first;
template<class A> struct second;

template<class A> using first_t  = typename first<A>::type;
template<class A> using second_t = typename second<A>::type;

template<typename T1, typename T2>
struct first<pair_<T1, T2>> {
    using type = T1;
};

template<typename T1, typename T2>
struct second<pair_<T1, T2>> {
    using type = T2;
};


//===============================================
//
//   index sequences
//
//===============================================

template<size_t... Inds>
struct index_seq{
    using type = index_seq;
};

namespace details {

template<class S1, class S2>
struct concat_index_seq;

template<size_t... Inds1, size_t... Inds2>
struct concat_index_seq<index_seq<Inds1...>, index_seq<Inds2...>>
    : public index_seq<Inds1..., (sizeof...(Inds1)+Inds2)...> {};

template<size_t N>
struct make_index_seq_impl;

template<>
struct make_index_seq_impl<0> {
    using type = index_seq<>;
};

template<>
struct make_index_seq_impl<1> {
    using type = index_seq<0>;
};

template<>
struct make_index_seq_impl<2> {
    using type = index_seq<0, 1>;
};

template<>
struct make_index_seq_impl<3> {
    using type = index_seq<0, 1, 2>;
};

template<size_t N>
struct make_index_seq_impl {
    using type = typename concat_index_seq<
        typename make_index_seq_impl<N/2>::type,
        typename make_index_seq_impl<N - N/2>::type>::type;
};

} // end namespace details

template<size_t N>
using make_index_seq = typename details::make_index_seq_impl<N>::type;


//===============================================
//
//   basic functions
//
//===============================================

template<typename A> using id = A;
template<typename A> using identity = A;


// arithmetic functions

template<typename A>
using negate = integral_constant<value_type_of<A>, -A::value>;

template<typename A>
using next = integral_constant<value_type_of<A>, A::value+1>;

template<typename A>
using prev = integral_constant<value_type_of<A>, A::value-1>;

template<typename A, typename B>
using plus = integral_constant<value_type_of<A>, A::value + B::value>;

template<typename A, typename B>
using minus = integral_constant<value_type_of<A>, A::value - B::value>;

template<typename A, typename B>
using mul = integral_constant<value_type_of<A>, A::value * B::value>;

template<typename A, typename B>
using div = integral_constant<value_type_of<A>, A::value / B::value>;

template<typename A, typename B>
using mod = integral_constant<value_type_of<A>, A::value % B::value>;

template<typename A, typename B> using multiplies = mul<A, B>;
template<typename A, typename B> using divides = div<A, B>;
template<typename A, typename B> using modulo = mod<A, B>;

// comparison functions

template<typename A, typename B> using eq = bool_<(A::value == B::value)>;
template<typename A, typename B> using ne = bool_<(A::value != B::value)>;
template<typename A, typename B> using gt = bool_<(A::value >  B::value)>;
template<typename A, typename B> using ge = bool_<(A::value >= B::value)>;
template<typename A, typename B> using lt = bool_<(A::value <  B::value)>;
template<typename A, typename B> using le = bool_<(A::value <= B::value)>;

template<typename A, typename B> using equal_to      = eq<A, B>;
template<typename A, typename B> using not_equal_to  = ne<A, B>;
template<typename A, typename B> using greater       = gt<A, B>;
template<typename A, typename B> using greater_equal = ge<A, B>;
template<typename A, typename B> using less          = lt<A, B>;
template<typename A, typename B> using less_equal    = le<A, B>;

// logical functions

template<typename A> using not_ = bool_<!A::value>;
template<typename A, typename B> using xor_ = bool_<A::value != B::value>;

namespace details {

template<bool Av, typename B>
struct and_helper : public bool_<B::value> {};

template<typename B>
struct and_helper<false, B> : public bool_<false> {};

template<bool Av, typename B>
struct or_helper : public bool_<true> {};

template<typename B>
struct or_helper<false, B> : public bool_<B::value> {};

}

template<typename A, typename B> using and_ = details::and_helper<A::value, B>;
template<typename A, typename B> using or_  = details::or_helper<A::value, B>;


//===============================================
//
//  select
//
//===============================================

template<typename... Args> struct select;

namespace details {

template<bool c1, typename... Rest>
struct select_helper;

template<typename T1, typename... Rest>
struct select_helper<true, T1, Rest...> {
    using type = T1;
};

template<typename T1, typename C2, typename... Rest>
struct select_helper<false, T1, C2, Rest...> {
    using type = typename select_helper<C2::value, Rest...>::type;
};

template<typename T1, typename T2>
struct select_helper<false, T1, T2> {
    using type = T2;
};

};

template<typename C1, typename... Rest>
struct select<C1, Rest...> {
    using type = typename details::select_helper<C1::value, Rest...>::type;
};

template<typename T>
struct select<T> {
    using type = T;
};

template<typename... Args>
using select_t = typename select<Args...>::type;


//===============================================
//
//   variadic reduction
//
//===============================================

// sum

template<typename... Args> struct sum;

template<typename A, typename... Rest>
struct sum<A, Rest...> : public plus<A, sum<Rest...>> {};

template<typename A>
struct sum<A> : public id<A> {};

// prod

template<typename... Args> struct prod;

template<typename A, typename... Rest>
struct prod<A, Rest...> : public mul<A, prod<Rest...>> {};

template<typename A>
struct prod<A> : public id<A> {};

// max

template<typename... Args> struct max;

template<typename A, typename... Rest>
struct max<A, Rest...> : public max<A, max<Rest...>> {};

template<typename A>
struct max<A> : public id<A> {};

template<typename A, typename B>
struct max<A, B> : public integral_constant<
    value_type_of<A>, (A::value > B::value ? A::value : B::value)> {};


// min

template<typename... Args> struct min;

template<typename A, typename... Rest>
struct min<A, Rest...> : public min<A, min<Rest...>> {};

template<typename A>
struct min<A> : public id<A> {};

template<typename A, typename B>
struct min<A, B> : public integral_constant<
     value_type_of<A>, (A::value < B::value ? A::value : B::value)> {};


namespace details {

// all_helper

template<bool a, typename... Rest> struct all_helper;

template<typename... Rest>
struct all_helper<false, Rest...> : public bool_<false> {};

template<typename A, typename... Rest>
struct all_helper<true, A, Rest...> : public all_helper<A::value, Rest...> {};

template<typename A>
struct all_helper<true, A> : public bool_<A::value> {};

template<>
struct all_helper<true> : public std::true_type {};

// any_helper

template<bool a, typename... Rest> struct any_helper;

template<typename... Rest>
struct any_helper<true, Rest...> : public bool_<true> {};

template<typename A, typename... Rest>
struct any_helper<false, A, Rest...> : public any_helper<A::value, Rest...> {};

template<typename A>
struct any_helper<false, A> : public bool_<A::value> {};

template<>
struct any_helper<false> : public bool_<false> {};

// generic count_if

template<typename A>
struct cond_to_size : public size_<(A::value ? 1 : 0)> {};

template<template<typename> class Pred, typename... Args>
struct count_if_impl;

template<template<typename> class Pred, typename X, typename... Rest>
struct count_if_impl<Pred, X, Rest...> : public plus<
    cond_to_size<Pred<X>>,
    count_if_impl<Pred, Rest...>> {};

template<template<typename> class Pred, typename X>
struct count_if_impl<Pred, X> : public cond_to_size<Pred<X>> {};

template<template<typename> class Pred>
struct count_if_impl<Pred> : public size_<0> {};

} // end namespace details


// all

template<typename... Args> struct all;

template<typename A, typename... Rest>
struct all<A, Rest...> :
    public details::all_helper<A::value, Rest...> {};

template<>
struct all<> : public bool_<true> {};

template<typename A>
struct all<A> : public bool_<A::value> {};

// any

template<typename... Args> struct any;

template<typename A, typename... Rest>
struct any<A, Rest...> :
    public details::any_helper<A::value, Rest...> {};

template<>
struct any<> : public bool_<false> {};

template<typename A>
struct any<A> : public bool_<A::value> {};

// count_true

template<typename... Args>
struct count_true :
     public details::count_if_impl<id, Args...> {};

// count_false

template<typename... Args>
struct count_false :
    public details::count_if_impl<not_, Args...> {};

// all_same

template<typename... Args>
struct all_same;

template<typename A>
struct all_same<A> : public true_ {};

template<typename A, typename B, typename... Rest>
struct all_same<A, B, Rest...> :
    public bool_<all_same<B, Rest...>::value && std::is_same<A, B>::value> {};



} // end namespace meta
} // end namespace clue

#endif
