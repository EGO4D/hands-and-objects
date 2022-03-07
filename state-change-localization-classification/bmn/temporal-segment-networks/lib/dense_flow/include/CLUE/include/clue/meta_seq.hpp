/**
 * @file meta_seq.hpp
 *
 * Meta-programming tools for working with a sequence of types
 */

#ifndef CLUE_META_SEQ__
#define CLUE_META_SEQ__

#include <clue/meta.hpp>

namespace clue {
namespace meta {

//===============================================
//
//   seq_
//
//===============================================

template<typename... Elems> struct seq_;

// length

template<class Seq> struct size;

template<typename... Elems>
struct size<seq_<Elems...>> :
    public size_<sizeof...(Elems)> {};

// empty

template<class Seq> struct empty;

template<typename... Elems>
struct empty<seq_<Elems...>> : public bool_<sizeof...(Elems) == 0> {};


//===============================================
//
//   Extract part
//
//===============================================

// forward declarations

template<class Seq> struct front;
template<class Seq> struct back;
template<class Seq, size_t N> struct at;

template<class Seq> using front_t = typename front<Seq>::type;
template<class Seq> using back_t  = typename back<Seq>::type;

template<class Seq, size_t N>
using at_t = typename at<Seq, N>::type;


// front

template<typename X, typename... Rest>
struct front<seq_<X, Rest...>> {
    using type = X;
};

// back

template<typename X, typename... Rest>
struct back<seq_<X, Rest...>> {
    using type = typename back<seq_<Rest...>>::type;
};

template<typename X>
struct back<seq_<X>> {
    using type = X;
};

// at

namespace details {

template<size_t N, typename... Elems> struct seq_at_helper;

template<typename X, typename... Rest>
struct seq_at_helper<0, X, Rest...> {
    using type = X;
};

template<size_t N, typename X, typename... Rest>
struct seq_at_helper<N, X, Rest...> {
    using type = typename seq_at_helper<N-1, Rest...>::type;
};

}

template<size_t N, typename... Elems>
struct at<seq_<Elems...>, N> {
    using type = typename details::seq_at_helper<N, Elems...>::type;
};

// first

template<typename X1, typename... Rest>
struct first<seq_<X1, Rest...>> {
    using type = X1;
};

// second

template<typename X1, typename X2, typename... Rest>
struct second<seq_<X1, X2, Rest...>> {
    using type = X2;
};


//===============================================
//
//   Modifiers
//
//===============================================


// forward declarations

template<class Seq> struct clear;
template<class Seq> struct pop_front;
template<class Seq> struct pop_back;
template<class Seq, typename X> struct push_front;
template<class Seq, typename X> struct push_back;

template<class Seq> using clear_t = typename clear<Seq>::type;
template<class Seq> using pop_front_t = typename pop_front<Seq>::type;
template<class Seq> using pop_back_t  = typename pop_back<Seq>::type;

template<class Seq, typename X>
using push_front_t = typename push_front<Seq, X>::type;

template<class Seq, typename X>
using push_back_t = typename push_back<Seq, X>::type;


// clear

template<typename... Elems>
struct clear<seq_<Elems...>> {
    using type = seq_<>;
};

// pop_front

template<typename X, typename... Rest>
struct pop_front<seq_<X, Rest...>> {
    using type = seq_<Rest...>;
};

template<typename X>
struct pop_front<seq_<X>> {
    using type = seq_<>;
};

// pop_back

template<typename X, typename... Rest>
struct pop_back<seq_<X, Rest...>> {
    using type = typename pop_back<seq_<Rest...>>::type;
};

template<typename X>
struct pop_back<seq_<X>> {
    using type = seq_<>;
};

// push_front

template<typename X, typename... Elems>
struct push_front<seq_<Elems...>, X> {
    using type = seq_<X, Elems...>;
};

template<typename X>
struct push_front<seq_<>, X> {
    using type = seq_<X>;
};

// push_back

template<typename X, typename... Elems>
struct push_back<seq_<Elems...>, X> {
    using type = seq_<Elems..., X>;
};

template<typename X>
struct push_back<seq_<>, X> {
    using type = seq_<X>;
};


//===============================================
//
//   Sequence reduction
//
//===============================================

// sum

template<typename... Elems>
struct sum<seq_<Elems...>> : public sum<Elems...> {};

// prod

template<typename... Elems>
struct prod<seq_<Elems...>> : public prod<Elems...> {};

// max

template<typename... Elems>
struct max<seq_<Elems...>> : public max<Elems...> {};

// min

template<typename... Elems>
struct min<seq_<Elems...>> : public min<Elems...> {};

// all

template<typename... Elems>
struct all<seq_<Elems...>> : public all<Elems...> {};

// any

template<typename... Elems>
struct any<seq_<Elems...>> : public any<Elems...> {};

// count_true

template<typename... Elems>
struct count_true<seq_<Elems...>> : public count_true<Elems...> {};

// count_false

template<typename... Elems>
struct count_false<seq_<Elems...>> : public count_false<Elems...> {};



//===============================================
//
//   Algorithms
//
//===============================================

// forward declarations

template<class S1, class S2> struct cat;
template<class S1, class S2> struct zip;

template<typename X, size_t N> struct repeat;

template<class Seq> struct reverse;

template<template<typename X> class F, class Seq>
struct transform;

template<template<typename X, typename Y> class F, class S1, class S2>
struct transform2;

template<template<typename X> class Pred, class Seq>
struct filter;

template<typename X, class Seq>
struct exists;

template<template<typename X> class Pred, class Seq>
struct exists_if;

template<typename X, class Seq>
struct count;

template<template<typename X> class Pred, class Seq>
struct count_if;


template<class S1, class S2>   using cat_t    = typename cat<S1, S2>::type;
template<class S1, class S2>   using zip_t    = typename zip<S1, S2>::type;
template<typename X, size_t N> using repeat_t = typename repeat<X, N>::type;

template<class Seq> using reverse_t = typename reverse<Seq>::type;

template<template<typename X> class F, class Seq>
using transform_t = typename transform<F, Seq>::type;

template<template<typename X, typename Y> class F, class S1, class S2>
using transform2_t = typename transform2<F, S1, S2>::type;

template<template<typename X> class Pred, class Seq>
using filter_t = typename filter<Pred, Seq>::type;


// implementations

// cat

template<class S1, class S2> struct cat;

template<typename... Args1, typename... Args2>
struct cat<seq_<Args1...>, seq_<Args2...>> {
    using type = seq_<Args1..., Args2...>;
};

// zip

template<typename... Args1, typename... Args2>
struct zip<seq_<Args1...>, seq_<Args2...>> {
    using type = seq_<pair_<Args1, Args2>...>;
};

// repeat

template<typename X, size_t N>
struct repeat {
    using type = typename push_front<
            typename repeat<X, N-1>::type, X>::type;
};

template<typename X>
struct repeat<X, 0> {
    using type = seq_<>;
};

template<typename X>
struct repeat<X, 1> {
    using type = seq_<X>;
};

// reverse

template<typename X, typename... Rest>
struct reverse<seq_<X, Rest...>> {
    using type = typename push_back<
            typename reverse<seq_<Rest...>>::type, X>::type;
};

template<>
struct reverse<seq_<>> {
    using type = seq_<>;
};

template<typename A>
struct reverse<seq_<A>> {
    using type = seq_<A>;
};

template<typename A, typename B>
struct reverse<seq_<A, B>> {
    using type = seq_<B, A>;
};


// transform

template<template<typename X> class F, typename... Elems>
struct transform<F, seq_<Elems...>> {
    using type = seq_<F<Elems>...>;
};

template<template<typename X, typename Y> class F,
         typename... Elems1, typename... Elems2>
struct transform2<F, seq_<Elems1...>, seq_<Elems2...>> {
    using type = seq_<F<Elems1, Elems2>...>;
};


// filter

template<template<typename> class Pred, typename X, typename... Rest>
struct filter<Pred, seq_<X, Rest...>> {
private:
    using filtered_rest = typename filter<Pred, seq_<Rest...>>::type;
public:
    using type = conditional_t<Pred<X>::value,
            push_front_t<filtered_rest, X>,
            filtered_rest>;
};

template<template<typename> class Pred, typename X>
struct filter<Pred, seq_<X>> {
public:
    using type = conditional_t<Pred<X>::value, seq_<X>, seq_<>>;
};

template<template<typename> class Pred>
struct filter<Pred, seq_<>> {
public:
    using type = seq_<>;
};


// exists

template<typename X, typename X1, typename... Rest>
struct exists<X, seq_<X1, Rest...>> :
    public details::or_helper<
        ::std::is_same<X, X1>::value,
        exists<X, seq_<Rest...>>> {};

template<typename X, typename X1>
struct exists<X, seq_<X1>> : public bool_<::std::is_same<X, X1>::value> {};

template<typename X>
struct exists<X, seq_<>> : public bool_<false> {};


// exists_if

template<template<typename> class Pred, typename X, typename... Rest>
struct exists_if<Pred, seq_<X, Rest...>> :
    public details::or_helper<
        Pred<X>::value,
        exists_if<Pred, seq_<Rest...>>> {};

template<template<typename> class Pred, typename X>
struct exists_if<Pred, seq_<X>> : public bool_<Pred<X>::value> {};

template<template<typename> class Pred>
struct exists_if<Pred, seq_<>> : public bool_<false> {};


// count

template<typename X, typename X1, typename... Rest>
struct count<X, seq_<X1, Rest...>> :
    public plus<
        details::cond_to_size<::std::is_same<X, X1>>,
        count<X, seq_<Rest...>>> {};

template<typename X, typename X1>
struct count<X, seq_<X1>> :
    public details::cond_to_size<::std::is_same<X, X1>> {};

template<typename X>
struct count<X, seq_<>> : public size_<0> {};


// count_if

template<template<typename X> class Pred, typename... Elems>
struct count_if<Pred, seq_<Elems...>> :
    public details::count_if_impl<Pred, Elems...> {};


} // end namespace mpl
} // end namespace clue

#endif
