#include <clue/meta.hpp>
#include <gtest/gtest.h>

namespace meta = clue::meta;

#define CHECK_META_F1(R, F, A) ASSERT_EQ(R::value, (F<A>::value))
#define CHECK_META_F2(R, F, A, B) ASSERT_EQ(R::value, (F<A, B>::value))

using i1 = meta::int_<1>;
using i2 = meta::int_<2>;
using i3 = meta::int_<3>;
using i4 = meta::int_<4>;
using i5 = meta::int_<5>;
using i6 = meta::int_<6>;

using b0 = meta::nil_;
using bt = meta::true_;
using bf = meta::false_;

TEST(Meta, Pairs) {
    using p = meta::pair_<i2, i5>;
    ASSERT_TRUE((std::is_same<meta::first_t<p>,  i2>::value));
    ASSERT_TRUE((std::is_same<meta::second_t<p>, i5>::value));
}


TEST(Meta, IndexSeq) {
    using I_0 = meta::index_seq<>;
    using I_1 = meta::index_seq<0>;
    using I_2 = meta::index_seq<0,1>;
    using I_3 = meta::index_seq<0,1,2>;
    using I_4 = meta::index_seq<0,1,2,3>;
    using I_5 = meta::index_seq<0,1,2,3,4>;
    using I_6 = meta::index_seq<0,1,2,3,4,5>;
    using I_7 = meta::index_seq<0,1,2,3,4,5,6>;
    using I_8 = meta::index_seq<0,1,2,3,4,5,6,7>;
    using I_9 = meta::index_seq<0,1,2,3,4,5,6,7,8>;
    using I_10 = meta::index_seq<0,1,2,3,4,5,6,7,8,9>;
    using I_11 = meta::index_seq<0,1,2,3,4,5,6,7,8,9,10>;
    using I_12 = meta::index_seq<0,1,2,3,4,5,6,7,8,9,10,11>;

    ASSERT_TRUE((std::is_same<I_0, meta::make_index_seq<0>>::value));
    ASSERT_TRUE((std::is_same<I_1, meta::make_index_seq<1>>::value));
    ASSERT_TRUE((std::is_same<I_2, meta::make_index_seq<2>>::value));
    ASSERT_TRUE((std::is_same<I_3, meta::make_index_seq<3>>::value));
    ASSERT_TRUE((std::is_same<I_4, meta::make_index_seq<4>>::value));
    ASSERT_TRUE((std::is_same<I_5, meta::make_index_seq<5>>::value));
    ASSERT_TRUE((std::is_same<I_6, meta::make_index_seq<6>>::value));
    ASSERT_TRUE((std::is_same<I_7, meta::make_index_seq<7>>::value));
    ASSERT_TRUE((std::is_same<I_8, meta::make_index_seq<8>>::value));
    ASSERT_TRUE((std::is_same<I_9, meta::make_index_seq<9>>::value));
    ASSERT_TRUE((std::is_same<I_10, meta::make_index_seq<10>>::value));
    ASSERT_TRUE((std::is_same<I_11, meta::make_index_seq<11>>::value));
    ASSERT_TRUE((std::is_same<I_12, meta::make_index_seq<12>>::value));
}


inline int sum_() {
    return 0;
}

template<typename T, typename... Ts>
inline int sum_(T x, Ts... rest) {
    return x + sum_(rest...);
}

template<typename... Ts, size_t... Is>
inline int tup_sum_(const std::tuple<Ts...>& tup, meta::index_seq<Is...>) {
    return sum_(std::get<Is>(tup)...);
}

template<typename... Ts>
inline int tup_sum(const std::tuple<Ts...>& tup) {
    return tup_sum_(tup, meta::make_index_seq<sizeof...(Ts)>{});
}

TEST(Meta, TupleSum) {
    ASSERT_EQ(0,  tup_sum(std::make_tuple()));
    ASSERT_EQ(1,  tup_sum(std::make_tuple(1)));
    ASSERT_EQ(3,  tup_sum(std::make_tuple(1, 2)));
    ASSERT_EQ(6,  tup_sum(std::make_tuple(1, 2, 3)));
    ASSERT_EQ(10, tup_sum(std::make_tuple(1, 2, 3, 4)));
    ASSERT_EQ(15, tup_sum(std::make_tuple(1, 2, 3, 4, 5)));
    ASSERT_EQ(21, tup_sum(std::make_tuple(1, 2, 3, 4, 5, 6)));
    ASSERT_EQ(28, tup_sum(std::make_tuple(1, 2, 3, 4, 5, 6, 7)));
    ASSERT_EQ(36, tup_sum(std::make_tuple(1, 2, 3, 4, 5, 6, 7, 8)));
}


TEST(Meta, ArithmeticFuns) {
    CHECK_META_F1(meta::int_<-3>, meta::negate, i3);
    CHECK_META_F1(i4, meta::next, i3);
    CHECK_META_F1(i2, meta::prev, i3);

    CHECK_META_F2(meta::int_<5>, meta::plus,  i2, i3);
    CHECK_META_F2(meta::int_<3>, meta::minus, i5, i2);
    CHECK_META_F2(meta::int_<6>, meta::mul,   i2, i3);
    CHECK_META_F2(meta::int_<3>, meta::div,   i6, i2);
    CHECK_META_F2(meta::int_<2>, meta::mod,   i5, i3);
}

TEST(Meta, CompareFuns) {
    CHECK_META_F2(bt, meta::eq, i2, i2);
    CHECK_META_F2(bf, meta::eq, i2, i3);
    CHECK_META_F2(bf, meta::eq, i3, i2);

    CHECK_META_F2(bf, meta::ne, i2, i2);
    CHECK_META_F2(bt, meta::ne, i2, i3);
    CHECK_META_F2(bt, meta::ne, i3, i2);

    CHECK_META_F2(bf, meta::lt, i2, i2);
    CHECK_META_F2(bt, meta::lt, i2, i3);
    CHECK_META_F2(bf, meta::lt, i3, i2);

    CHECK_META_F2(bt, meta::le, i2, i2);
    CHECK_META_F2(bt, meta::le, i2, i3);
    CHECK_META_F2(bf, meta::le, i3, i2);

    CHECK_META_F2(bf, meta::gt, i2, i2);
    CHECK_META_F2(bf, meta::gt, i2, i3);
    CHECK_META_F2(bt, meta::gt, i3, i2);

    CHECK_META_F2(bt, meta::ge, i2, i2);
    CHECK_META_F2(bf, meta::ge, i2, i3);
    CHECK_META_F2(bt, meta::ge, i3, i2);
}

TEST(Meta, LogicalFuns) {
    CHECK_META_F1(bt, meta::not_, bf);
    CHECK_META_F1(bf, meta::not_, bt);

    CHECK_META_F2(bf, meta::and_, bf, bf);
    CHECK_META_F2(bf, meta::and_, bf, bt);
    CHECK_META_F2(bf, meta::and_, bt, bf);
    CHECK_META_F2(bt, meta::and_, bt, bt);

    CHECK_META_F2(bf, meta::or_, bf, bf);
    CHECK_META_F2(bt, meta::or_, bf, bt);
    CHECK_META_F2(bt, meta::or_, bt, bf);
    CHECK_META_F2(bt, meta::or_, bt, bt);
}

TEST(Meta, LazyAndOr) {
    ASSERT_EQ(false, (meta::and_<bf, meta::nil_>::value));
    ASSERT_EQ(false, (meta::and_<bf, meta::true_>::value));

    ASSERT_EQ(true, (meta::or_<bt, meta::nil_>::value));
    ASSERT_EQ(true, (meta::or_<bt, meta::false_>::value));
}


TEST(Meta, Select) {
    using t = meta::true_;
    using f = meta::false_;

    using r1 = meta::select_t<i1>;

    ASSERT_TRUE((::std::is_same<r1, i1>::value));

    using r2t = meta::select_t<t, i1, i2>;
    using r2f = meta::select_t<f, i1, i2>;

    ASSERT_TRUE((::std::is_same<r2t, i1>::value));
    ASSERT_TRUE((::std::is_same<r2f, i2>::value));

    using r3tt = meta::select_t<t, i1, t, i2, i3>;
    using r3tf = meta::select_t<t, i1, f, i2, i3>;
    using r3ft = meta::select_t<f, i1, t, i2, i3>;
    using r3ff = meta::select_t<f, i1, f, i2, i3>;

    ASSERT_TRUE((::std::is_same<r3tt, i1>::value));
    ASSERT_TRUE((::std::is_same<r3tf, i1>::value));
    ASSERT_TRUE((::std::is_same<r3ft, i2>::value));
    ASSERT_TRUE((::std::is_same<r3ff, i3>::value));
}


TEST(Meta, ValueReduce) {
    ASSERT_EQ(2,  (meta::sum<i2>::value));
    ASSERT_EQ(5,  (meta::sum<i2, i3>::value));
    ASSERT_EQ(6,  (meta::sum<i2, i3, i1>::value));
    ASSERT_EQ(10, (meta::sum<i2, i3, i1, i4>::value));

    ASSERT_EQ(2,  (meta::prod<i2>::value));
    ASSERT_EQ(6,  (meta::prod<i2, i3>::value));
    ASSERT_EQ(6,  (meta::prod<i2, i3, i1>::value));
    ASSERT_EQ(24, (meta::prod<i2, i3, i1, i4>::value));

    ASSERT_EQ(2,  (meta::max<i2>::value));
    ASSERT_EQ(3,  (meta::max<i2, i3>::value));
    ASSERT_EQ(3,  (meta::max<i2, i3, i1>::value));
    ASSERT_EQ(4,  (meta::max<i2, i3, i1, i4>::value));

    ASSERT_EQ(2,  (meta::min<i2>::value));
    ASSERT_EQ(2,  (meta::min<i2, i3>::value));
    ASSERT_EQ(1,  (meta::min<i2, i3, i1>::value));
    ASSERT_EQ(1,  (meta::min<i2, i3, i1, i4>::value));
}

TEST(Meta, CountBool) {
    ASSERT_EQ(0, (meta::count_true<>::value));
    ASSERT_EQ(1, (meta::count_true<bt>::value));
    ASSERT_EQ(1, (meta::count_true<bt, bf>::value));
    ASSERT_EQ(1, (meta::count_true<bt, bf, bf>::value));
    ASSERT_EQ(2, (meta::count_true<bt, bf, bf, bt>::value));

    ASSERT_EQ(0, (meta::count_false<>::value));
    ASSERT_EQ(0, (meta::count_false<bt>::value));
    ASSERT_EQ(1, (meta::count_false<bt, bf>::value));
    ASSERT_EQ(2, (meta::count_false<bt, bf, bf>::value));
    ASSERT_EQ(2, (meta::count_false<bt, bf, bf, bt>::value));
}

TEST(Meta, All) {
    bool t = true, f = false;

    ASSERT_EQ(t, (meta::all<>::value));
    ASSERT_EQ(f, (meta::all<bf>::value));
    ASSERT_EQ(t, (meta::all<bt>::value));

    ASSERT_EQ(f, (meta::all<bf, bf>::value));
    ASSERT_EQ(f, (meta::all<bf, bt>::value));
    ASSERT_EQ(f, (meta::all<bt, bf>::value));
    ASSERT_EQ(t, (meta::all<bt, bt>::value));

    ASSERT_EQ(f, (meta::all<bf, bf, bf>::value));
    ASSERT_EQ(f, (meta::all<bf, bf, bt>::value));
    ASSERT_EQ(f, (meta::all<bf, bt, bf>::value));
    ASSERT_EQ(f, (meta::all<bf, bt, bt>::value));
    ASSERT_EQ(f, (meta::all<bt, bf, bf>::value));
    ASSERT_EQ(f, (meta::all<bt, bf, bt>::value));
    ASSERT_EQ(f, (meta::all<bt, bt, bf>::value));
    ASSERT_EQ(t, (meta::all<bt, bt, bt>::value));
}

TEST(Meta, Any) {
    bool t = true, f = false;

    ASSERT_EQ(f, (meta::any<>::value));
    ASSERT_EQ(f, (meta::any<bf>::value));
    ASSERT_EQ(t, (meta::any<bt>::value));

    ASSERT_EQ(f, (meta::any<bf, bf>::value));
    ASSERT_EQ(t, (meta::any<bf, bt>::value));
    ASSERT_EQ(t, (meta::any<bt, bf>::value));
    ASSERT_EQ(t, (meta::any<bt, bt>::value));

    ASSERT_EQ(f, (meta::any<bf, bf, bf>::value));
    ASSERT_EQ(t, (meta::any<bf, bf, bt>::value));
    ASSERT_EQ(t, (meta::any<bf, bt, bf>::value));
    ASSERT_EQ(t, (meta::any<bf, bt, bt>::value));
    ASSERT_EQ(t, (meta::any<bt, bf, bf>::value));
    ASSERT_EQ(t, (meta::any<bt, bf, bt>::value));
    ASSERT_EQ(t, (meta::any<bt, bt, bf>::value));
    ASSERT_EQ(t, (meta::any<bt, bt, bt>::value));
}


TEST(Meta, LazyAllAny) {
    ASSERT_EQ(false, (meta::all<bf, b0>::value));
    ASSERT_EQ(false, (meta::all<bf, bf, b0>::value));
    ASSERT_EQ(false, (meta::all<bf, bt, b0>::value));
    ASSERT_EQ(false, (meta::all<bt, bf, b0>::value));

    ASSERT_EQ(true,  (meta::any<bt, b0>::value));
    ASSERT_EQ(true,  (meta::any<bt, bt, b0>::value));
    ASSERT_EQ(true,  (meta::any<bf, bt, b0>::value));
    ASSERT_EQ(true,  (meta::any<bt, bf, b0>::value));
}


TEST(Meta, AllSame) {
    ASSERT_EQ(true, (meta::all_same<i1>::value));

    ASSERT_EQ(true,  (meta::all_same<i1, i1>::value));
    ASSERT_EQ(false, (meta::all_same<i1, i2>::value));

    ASSERT_EQ(true,  (meta::all_same<i1, i1, i1>::value));
    ASSERT_EQ(false, (meta::all_same<i1, i1, i2>::value));
    ASSERT_EQ(false, (meta::all_same<i1, i2, i1>::value));
    ASSERT_EQ(false, (meta::all_same<i1, i2, i2>::value));
    ASSERT_EQ(false, (meta::all_same<i2, i1, i1>::value));
    ASSERT_EQ(false, (meta::all_same<i2, i1, i2>::value));
    ASSERT_EQ(false, (meta::all_same<i2, i2, i1>::value));
    ASSERT_EQ(true,  (meta::all_same<i2, i2, i2>::value));

    ASSERT_EQ(true,  (meta::all_same<i1, i1, i1, i1>::value));
    ASSERT_EQ(false, (meta::all_same<i2, i1, i1, i1>::value));
    ASSERT_EQ(false, (meta::all_same<i1, i2, i1, i1>::value));
    ASSERT_EQ(false, (meta::all_same<i1, i1, i2, i1>::value));
    ASSERT_EQ(false, (meta::all_same<i1, i1, i1, i2>::value));
}
