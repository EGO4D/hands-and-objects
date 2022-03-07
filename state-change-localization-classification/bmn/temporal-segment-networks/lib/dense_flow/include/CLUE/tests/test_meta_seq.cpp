#include <clue/meta_seq.hpp>
#include <gtest/gtest.h>

namespace meta = clue::meta;
using meta::seq_;

#define CHECK_META_T(R, Expr) ASSERT_TRUE((std::is_same<R, Expr>::value))

using i1 = meta::int_<1>;
using i2 = meta::int_<2>;
using i3 = meta::int_<3>;
using i4 = meta::int_<4>;
using i5 = meta::int_<5>;
using i6 = meta::int_<6>;


TEST(MetaSeq, Properties) {
    using L0 = seq_<>;
    using L1 = seq_<i1>;
    using L2 = seq_<i1, i2>;
    using L3 = seq_<i1, i2, i3>;

    ASSERT_EQ(0, (meta::size<L0>::value));
    ASSERT_EQ(1, (meta::size<L1>::value));
    ASSERT_EQ(2, (meta::size<L2>::value));
    ASSERT_EQ(3, (meta::size<L3>::value));

    ASSERT_EQ(true,  (meta::empty<L0>::value));
    ASSERT_EQ(false, (meta::empty<L1>::value));
    ASSERT_EQ(false, (meta::empty<L2>::value));
    ASSERT_EQ(false, (meta::empty<L3>::value));
}

TEST(MetaSeq, Parts) {
    using L1 = seq_<i1>;
    using L2 = seq_<i1, i2>;
    using L3 = seq_<i1, i2, i3>;

    // front

    CHECK_META_T(i1, meta::front_t<L1>);
    CHECK_META_T(i1, meta::front_t<L2>);
    CHECK_META_T(i1, meta::front_t<L3>);

    // back

    CHECK_META_T(i1, meta::back_t<L1>);
    CHECK_META_T(i2, meta::back_t<L2>);
    CHECK_META_T(i3, meta::back_t<L3>);

    // at
    using l1_0 = meta::at_t<L1, 0>;
    using l2_0 = meta::at_t<L2, 0>;
    using l2_1 = meta::at_t<L2, 1>;
    using l3_0 = meta::at_t<L3, 0>;
    using l3_1 = meta::at_t<L3, 1>;
    using l3_2 = meta::at_t<L3, 2>;

    CHECK_META_T(i1, l1_0);
    CHECK_META_T(i1, l2_0);
    CHECK_META_T(i2, l2_1);
    CHECK_META_T(i1, l3_0);
    CHECK_META_T(i2, l3_1);
    CHECK_META_T(i3, l3_2);

    // first

    CHECK_META_T(i1, meta::first_t<L1>);
    CHECK_META_T(i1, meta::first_t<L2>);
    CHECK_META_T(i1, meta::first_t<L3>);

    // second

    CHECK_META_T(i2, meta::second_t<L2>);
    CHECK_META_T(i2, meta::second_t<L3>);
}


TEST(MetaSeq, Clear) {
    using L1 = seq_<i1>;
    using L2 = seq_<i1, i2>;
    using L3 = seq_<i1, i2, i3>;

    CHECK_META_T(seq_<>, meta::clear_t<L1>);
    CHECK_META_T(seq_<>, meta::clear_t<L2>);
    CHECK_META_T(seq_<>, meta::clear_t<L3>);
}


TEST(MetaSeq, PopFront) {
    using L3 = seq_<i1, i2, i3>;

    using p1_r = seq_<i2, i3>;
    CHECK_META_T(p1_r, meta::pop_front_t<L3>);

    using p2_r = seq_<i3>;
    CHECK_META_T(p2_r, meta::pop_front_t<p1_r>);

    using p3_r = seq_<>;
    CHECK_META_T(p3_r, meta::pop_front_t<p2_r>);
}


TEST(MetaSeq, PushFront) {
    using L0 = seq_<>;
    using L1 = seq_<i1>;
    using L2 = seq_<i1, i2>;
    using L3 = seq_<i1, i2, i3>;

    using l0_pp = meta::push_front_t<L0, int>;
    using l0_pp_r = seq_<int>;
    CHECK_META_T(l0_pp_r, l0_pp);

    using l1_pp = meta::push_front_t<L1, int>;
    using l1_pp_r = seq_<int, i1>;
    CHECK_META_T(l1_pp_r, l1_pp);

    using l2_pp = meta::push_front_t<L2, int>;
    using l2_pp_r = seq_<int, i1, i2>;
    CHECK_META_T(l2_pp_r, l2_pp);

    using l3_pp = meta::push_front_t<L3, int>;
    using l3_pp_r = seq_<int, i1, i2, i3>;
    CHECK_META_T(l3_pp_r, l3_pp);
}

TEST(MetaSeq, PushBack) {
    using L0 = seq_<>;
    using L1 = seq_<i1>;
    using L2 = seq_<i1, i2>;
    using L3 = seq_<i1, i2, i3>;

    using l0_ap = meta::push_back_t<L0, int>;
    using l0_ap_r = seq_<int>;
    CHECK_META_T(l0_ap_r, l0_ap);

    using l1_ap = meta::push_back_t<L1, int>;
    using l1_ap_r = seq_<i1, int>;
    CHECK_META_T(l1_ap_r, l1_ap);

    using l2_ap = meta::push_back_t<L2, int>;
    using l2_ap_r = seq_<i1, i2, int>;
    CHECK_META_T(l2_ap_r, l2_ap);

    using l3_ap = meta::push_back_t<L3, int>;
    using l3_ap_r = seq_<i1, i2, i3, int>;
    CHECK_META_T(l3_ap_r, l3_ap);
}


TEST(MetaSeq, Reduction) {
    using bt = meta::true_;
    using bf = meta::false_;

    // all

    using B0 = seq_<>;
    using B1 = seq_<bt>;
    using B3 = seq_<bt, bf, bt>;

    ASSERT_EQ(true,  meta::all<B0>::value);
    ASSERT_EQ(true,  meta::all<B1>::value);
    ASSERT_EQ(false, meta::all<B3>::value);

    // any

    ASSERT_EQ(false, meta::any<B0>::value);
    ASSERT_EQ(true,  meta::any<B1>::value);
    ASSERT_EQ(true,  meta::any<B3>::value);

    // count_true

    ASSERT_EQ(0, meta::count_true<B0>::value);
    ASSERT_EQ(1, meta::count_true<B1>::value);
    ASSERT_EQ(2, meta::count_true<B3>::value);

    // count_false

    ASSERT_EQ(0, meta::count_false<B0>::value);
    ASSERT_EQ(0, meta::count_false<B1>::value);
    ASSERT_EQ(1, meta::count_false<B3>::value);

    // sum

    using V1 = seq_<i3>;
    using V3 = seq_<i2, i3, i4>;

    ASSERT_EQ(3, meta::sum<V1>::value);
    ASSERT_EQ(9, meta::sum<V3>::value);

    // verify it does not interfere with the original form

    using _r1 = meta::sum<i3>;
    ASSERT_EQ(3, _r1::value);

    using _r3 = meta::sum<i2, i3, i4>;
    ASSERT_EQ(9, _r3::value);

    // prod

    ASSERT_EQ(3,  meta::prod<V1>::value);
    ASSERT_EQ(24, meta::prod<V3>::value);

    // max

    ASSERT_EQ(3, meta::max<V1>::value);
    ASSERT_EQ(4, meta::max<V3>::value);

    // min

    ASSERT_EQ(3, meta::min<V1>::value);
    ASSERT_EQ(2, meta::min<V3>::value);
}


TEST(MetaSeq, Cat) {
    using L0 = seq_<>;
    using L1 = seq_<i1>;
    using L2 = seq_<i1, i2>;
    using L3 = seq_<i1, i2, i3>;

    using cat_0_0 = meta::cat_t<L0, L0>;
    using cat_0_0_r = seq_<>;
    CHECK_META_T(cat_0_0_r, cat_0_0);

    using cat_0_1 = meta::cat_t<L0, L1>;
    using cat_0_1_r = seq_<i1>;
    CHECK_META_T(cat_0_1_r, cat_0_1);

    using cat_1_0 = meta::cat_t<L1, L0>;
    using cat_1_0_r = seq_<i1>;
    CHECK_META_T(cat_1_0_r, cat_1_0);

    using cat_2_0 = meta::cat_t<L2, L0>;
    using cat_2_0_r = seq_<i1, i2>;
    CHECK_META_T(cat_2_0_r, cat_2_0);

    using cat_0_2 = meta::cat_t<L0, L2>;
    using cat_0_2_r = seq_<i1, i2>;
    CHECK_META_T(cat_0_2_r, cat_0_2);

    using cat_1_2 = meta::cat_t<L1, L2>;
    using cat_1_2_r = seq_<i1, i1, i2>;
    CHECK_META_T(cat_1_2_r, cat_1_2);

    using cat_2_1 = meta::cat_t<L2, L1>;
    using cat_2_1_r = seq_<i1, i2, i1>;
    CHECK_META_T(cat_2_1_r, cat_2_1);

    using cat_2_3 = meta::cat_t<L2, L3>;
    using cat_2_3_r = seq_<i1, i2, i1, i2, i3>;
    CHECK_META_T(cat_2_3_r, cat_2_3);
}


TEST(MetaSeq, Zip) {
    using meta::pair_;

    using zip_0 = meta::zip_t<seq_<>, seq_<>>;
    using zip_0_r = seq_<>;
    CHECK_META_T(zip_0_r, zip_0);

    using zip_1 = meta::zip_t<seq_<i1>, seq_<i4>>;
    using zip_1_r = seq_<pair_<i1, i4>>;
    CHECK_META_T(zip_1_r, zip_1);

    using zip_2 = meta::zip_t<seq_<i1, i2>, seq_<i4, i5>>;
    using zip_2_r = seq_<pair_<i1, i4>, pair_<i2, i5>>;
    CHECK_META_T(zip_2_r, zip_2);

    using zip_3 = meta::zip_t<seq_<i1, i2, i3>, seq_<i4, i5, i6>>;
    using zip_3_r = seq_<pair_<i1, i4>, pair_<i2, i5>, pair_<i3, i6>>;
    CHECK_META_T(zip_3_r, zip_3);
}


TEST(MetaSeq, Repeat) {

    using rep_0 = meta::repeat_t<int, 0>;
    using rep_0_r = seq_<>;
    CHECK_META_T(rep_0_r, rep_0);

    using rep_1 = meta::repeat_t<int, 1>;
    using rep_1_r = seq_<int>;
    CHECK_META_T(rep_1_r, rep_1);

    using rep_2 = meta::repeat_t<int, 2>;
    using rep_2_r = seq_<int, int>;
    CHECK_META_T(rep_2_r, rep_2);

    using rep_3 = meta::repeat_t<int, 3>;
    using rep_3_r = seq_<int, int, int>;
    CHECK_META_T(rep_3_r, rep_3);
}


TEST(MetaSeq, Reverse) {
    using L0 = seq_<>;
    using L1 = seq_<i1>;
    using L2 = seq_<i1, i2>;
    using L3 = seq_<i1, i2, i3>;
    using L4 = seq_<i1, i2, i3, i4>;

    using l0_rv = meta::reverse_t<L0>;
    using l0_rv_r = seq_<>;
    CHECK_META_T(l0_rv_r, l0_rv);

    using l1_rv = meta::reverse_t<L1>;
    using l1_rv_r = seq_<i1>;
    CHECK_META_T(l1_rv_r, l1_rv);

    using l2_rv = meta::reverse_t<L2>;
    using l2_rv_r = seq_<i2, i1>;
    CHECK_META_T(l2_rv_r, l2_rv);

    using l3_rv = meta::reverse_t<L3>;
    using l3_rv_r = seq_<i3, i2, i1>;
    CHECK_META_T(l3_rv_r, l3_rv);

    using l4_rv = meta::reverse_t<L4>;
    using l4_rv_r = seq_<i4, i3, i2, i1>;
    CHECK_META_T(l4_rv_r, l4_rv);
}


TEST(MetaSeq, Transform) {
    using L0 = seq_<>;
    using L1 = seq_<i1>;
    using L2 = seq_<i1, i2>;
    using L3 = seq_<i1, i2, i3>;

    using map_0 = meta::transform_t<meta::next, L0>;
    using map_0_r = seq_<>;
    CHECK_META_T(map_0_r, map_0);

    using map_1 = meta::transform_t<meta::next, L1>;
    using map_1_r = seq_<i2>;
    CHECK_META_T(map_1_r, map_1);

    using map_2 = meta::transform_t<meta::next, L2>;
    using map_2_r = seq_<i2, i3>;
    CHECK_META_T(map_2_r, map_2);

    using map_3 = meta::transform_t<meta::next, L3>;
    using map_3_r = seq_<i2, i3, i4>;
    CHECK_META_T(map_3_r, map_3);
}


TEST(MetaSeq, BinaryTransform) {
    using meta::int_;

    using L0 = seq_<>;
    using L1 = seq_<i1>;
    using L2 = seq_<i1, i2>;
    using L3 = seq_<i1, i2, i3>;

    using R0 = seq_<>;
    using R1 = seq_<i4>;
    using R2 = seq_<i4, i5>;
    using R3 = seq_<i4, i5, i6>;

    using map_0 = meta::transform2_t<meta::plus, L0, R0>;
    using map_0_r = seq_<>;
    CHECK_META_T(map_0_r, map_0);

    using map_1 = meta::transform2_t<meta::plus, L1, R1>;
    using map_1_r = seq_<int_<5>>;
    CHECK_META_T(map_1_r, map_1);

    using map_2 = meta::transform2_t<meta::plus, L2, R2>;
    using map_2_r = seq_<int_<5>, int_<7>>;
    CHECK_META_T(map_2_r, map_2);

    using map_3 = meta::transform2_t<meta::plus, L3, R3>;
    using map_3_r = seq_<int_<5>, int_<7>, int_<9>>;
    CHECK_META_T(map_3_r, map_3);
}


template<typename X>
struct _is_even : public meta::bool_<X::value % 2 == 0> {};

template<typename X>
struct _is_odd : public meta::bool_<X::value % 2 == 1> {};


TEST(MetaSeq, Filter) {

    using L0   = seq_<>;
    using L1   = seq_<i1>;
    using L2   = seq_<i2>;
    using L12  = seq_<i1, i2>;
    using L23  = seq_<i2, i3>;
    using L123 = seq_<i1, i2, i3>;
    using L234 = seq_<i2, i3, i4>;

    using L0_f = meta::filter_t<_is_even, L0>;
    using L0_f_r = seq_<>;
    CHECK_META_T(L0_f_r, L0_f);

    using L1_f = meta::filter_t<_is_even, L1>;
    using L1_f_r = seq_<>;
    CHECK_META_T(L1_f_r, L1_f);

    using L2_f = meta::filter_t<_is_even, L2>;
    using L2_f_r = seq_<i2>;
    CHECK_META_T(L2_f_r, L2_f);

    using L12_f = meta::filter_t<_is_even, L12>;
    using L12_f_r = seq_<i2>;
    CHECK_META_T(L12_f_r, L12_f);

    using L23_f = meta::filter_t<_is_even, L23>;
    using L23_f_r = seq_<i2>;
    CHECK_META_T(L23_f_r, L23_f);

    using L123_f = meta::filter_t<_is_even, L123>;
    using L123_f_r = seq_<i2>;
    CHECK_META_T(L123_f_r, L123_f);

    using L234_f = meta::filter_t<_is_even, L234>;
    using L234_f_r = seq_<i2, i4>;
    CHECK_META_T(L234_f_r, L234_f);
}


TEST(MetaSeq, Exists) {

    using L0 = seq_<>;
    using L1 = seq_<i1>;
    using L2 = seq_<i1, i2>;
    using L3 = seq_<i1, i2, i3>;

    ASSERT_EQ(false, (::meta::exists<i1, L0>::value));
    ASSERT_EQ(true,  (::meta::exists<i1, L1>::value));
    ASSERT_EQ(true,  (::meta::exists<i1, L2>::value));
    ASSERT_EQ(true,  (::meta::exists<i1, L3>::value));

    ASSERT_EQ(false, (::meta::exists<i2, L0>::value));
    ASSERT_EQ(false, (::meta::exists<i2, L1>::value));
    ASSERT_EQ(true,  (::meta::exists<i2, L2>::value));
    ASSERT_EQ(true,  (::meta::exists<i2, L3>::value));

    ASSERT_EQ(false, (::meta::exists<i3, L0>::value));
    ASSERT_EQ(false, (::meta::exists<i3, L1>::value));
    ASSERT_EQ(false, (::meta::exists<i3, L2>::value));
    ASSERT_EQ(true,  (::meta::exists<i3, L3>::value));

    ASSERT_EQ(false, (::meta::exists<i4, L0>::value));
    ASSERT_EQ(false, (::meta::exists<i4, L1>::value));
    ASSERT_EQ(false, (::meta::exists<i4, L2>::value));
    ASSERT_EQ(false, (::meta::exists<i4, L3>::value));
}


TEST(MetaSeq, ExistsIf) {
    using L0 = seq_<>;
    using L1 = seq_<i1>;
    using L2 = seq_<i1, i2>;
    using L3 = seq_<i1, i2, i3>;

    ASSERT_EQ(false, (::meta::exists_if<_is_even, L0>::value));
    ASSERT_EQ(false, (::meta::exists_if<_is_even, L1>::value));
    ASSERT_EQ(true,  (::meta::exists_if<_is_even, L2>::value));
    ASSERT_EQ(true,  (::meta::exists_if<_is_even, L3>::value));

    ASSERT_EQ(false, (::meta::exists_if<_is_odd, L0>::value));
    ASSERT_EQ(true,  (::meta::exists_if<_is_odd, L1>::value));
    ASSERT_EQ(true,  (::meta::exists_if<_is_odd, L2>::value));
    ASSERT_EQ(true,  (::meta::exists_if<_is_odd, L3>::value));
}


TEST(MetaSeq, Count) {
    using A = seq_<>;
    using B = seq_<i1, i2>;
    using C = seq_<i1, i2, i2, i1, i2>;

    ASSERT_EQ(0, (meta::count<i1, A>::value));
    ASSERT_EQ(1, (meta::count<i1, B>::value));
    ASSERT_EQ(2, (meta::count<i1, C>::value));

    ASSERT_EQ(0, (meta::count<i2, A>::value));
    ASSERT_EQ(1, (meta::count<i2, B>::value));
    ASSERT_EQ(3, (meta::count<i2, C>::value));

    ASSERT_EQ(0, (meta::count<i3, A>::value));
    ASSERT_EQ(0, (meta::count<i3, B>::value));
    ASSERT_EQ(0, (meta::count<i3, C>::value));
}

TEST(MetaSeq, CountIf) {
    using A = seq_<>;
    using B = seq_<i1, i2>;
    using C = seq_<i1, i2, i2, i1, i2>;

    ASSERT_EQ(0, (meta::count_if<_is_even, A>::value));
    ASSERT_EQ(1, (meta::count_if<_is_even, B>::value));
    ASSERT_EQ(3, (meta::count_if<_is_even, C>::value));

    ASSERT_EQ(0, (meta::count_if<_is_odd, A>::value));
    ASSERT_EQ(1, (meta::count_if<_is_odd, B>::value));
    ASSERT_EQ(2, (meta::count_if<_is_odd, C>::value));
}

