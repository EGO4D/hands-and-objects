#include <clue/value_range.hpp>
#include <gtest/gtest.h>
#include <vector>

using clue::vrange;

template<typename T>
void test_unit_range(const clue::value_range<T>& rgn, const T& a, const T& b) {
    using std::size_t;
    using difference_type = typename clue::value_range<T>::difference_type;

    ASSERT_EQ(a,   rgn.begin_value());
    ASSERT_EQ(b,   rgn.end_value());
    ASSERT_EQ(a,   rgn.front());
    ASSERT_EQ(b-1, rgn.back());
    ASSERT_EQ(1,   rgn.step());

    ASSERT_EQ(size_t(b - a), rgn.size());
    ASSERT_EQ((a == b), rgn.empty());

    auto ifirst = rgn.begin();
    auto ilast  = rgn.end();

    if (!rgn.empty()) {
        ASSERT_EQ(a, *ifirst);
        ASSERT_EQ(b, *ilast);
    }
    ASSERT_EQ((a == b), (ifirst == ilast));
    ASSERT_EQ((a != b), (ifirst != ilast));
    ASSERT_EQ((a <  b), (ifirst <  ilast));
    ASSERT_EQ((a <= b), (ifirst <= ilast));
    ASSERT_EQ((a >  b), (ifirst >  ilast));
    ASSERT_EQ((a >= b), (ifirst >= ilast));

    difference_type n = (difference_type)(rgn.size());
    ASSERT_EQ(ilast, ifirst + n);
    ASSERT_EQ(ifirst, ilast - n);
    ASSERT_EQ(n, ilast - ifirst);
    ASSERT_EQ(n, std::distance(ifirst, ilast));

    for (size_t i = 0; i < (size_t)n; ++i) {
        ASSERT_EQ(a + i, rgn[i]);
        ASSERT_EQ(a + i, rgn.at(i));
    }
    ASSERT_THROW(rgn.at(size_t(n)), std::out_of_range);

    if (!rgn.empty()) {
        auto i1 = ifirst;
        ASSERT_EQ(a,   *(i1++));
        ASSERT_EQ(a+1, *i1);
        ASSERT_EQ(a+1, *(i1--));
        ASSERT_EQ(a,   *i1);

        auto i2 = ifirst;
        ASSERT_EQ(a+1, *(++i2));
        ASSERT_EQ(a+1, *i2);
        ASSERT_EQ(a,   *(--i2));
        ASSERT_EQ(a,   *i2);

        i1 += 1;
        ASSERT_EQ(a+1, *i1);
        i1 -= 1;
        ASSERT_EQ(a, *i1);

        ASSERT_EQ(a+1, *(ifirst+1));
        ASSERT_EQ(b-1, *(ilast-1));
    }

    std::vector<T> v_gt;
    for (T x = a; x != b; ++x) v_gt.push_back(x);
    ASSERT_TRUE(v_gt.size() == rgn.size());

    std::vector<T> v1;
    for (auto x: rgn) v1.push_back(x);
    ASSERT_EQ(v_gt, v1);

    std::vector<T> v2(ifirst, ilast);
    ASSERT_EQ(v_gt, v2);
}


template<typename T, typename S>
void test_stepped_range(const clue::stepped_value_range<T, S>& rgn, const T& a, const T& b, const S& s) {
    using std::size_t;
    using difference_type = typename clue::value_range<T>::difference_type;

    ASSERT_EQ(a, rgn.begin_value());
    ASSERT_EQ(b, rgn.end_value());
    ASSERT_EQ(s, rgn.step());

    size_t len = rgn.size();
    if (a < b) {
        ASSERT_GT(len, 0);
        ASSERT_LT(a + (len-1) * s, b);
    }
    ASSERT_GE(a + len * s,     b);
    ASSERT_EQ((a == b), rgn.empty());
    ASSERT_EQ((len == 0), rgn.empty());

    if (!rgn.empty()) {
        ASSERT_EQ(a,   rgn.front());
        ASSERT_EQ(a + (len-1) * s, rgn.back());
    }

    auto ifirst = rgn.begin();
    auto ilast  = rgn.end();

    if (!rgn.empty()) {
        ASSERT_EQ(a, *ifirst);
        ASSERT_EQ(a + len * s, *ilast);
    }
    ASSERT_EQ((a == b), (ifirst == ilast));
    ASSERT_EQ((a != b), (ifirst != ilast));
    ASSERT_EQ((a <  b), (ifirst <  ilast));
    ASSERT_EQ((a <= b), (ifirst <= ilast));
    ASSERT_EQ((a >  b), (ifirst >  ilast));
    ASSERT_EQ((a >= b), (ifirst >= ilast));

    difference_type n = (difference_type)(len);
    ASSERT_EQ(ilast, ifirst + n);
    ASSERT_EQ(ifirst, ilast - n);
    ASSERT_EQ(n, ilast - ifirst);
    ASSERT_EQ(n, std::distance(ifirst, ilast));

    for (size_t i = 0; i < len; ++i) {
        ASSERT_EQ(a + i * s, rgn[i]);
        ASSERT_EQ(a + i * s, rgn.at(i));
    }
    ASSERT_THROW(rgn.at(size_t(n)), std::out_of_range);

    T e = a + len * s;
    if (!rgn.empty()) {
        auto i1 = ifirst;
        ASSERT_EQ(a,   *(i1++));
        ASSERT_EQ(a+s, *i1);
        ASSERT_EQ(a+s, *(i1--));
        ASSERT_EQ(a,   *i1);

        auto i2 = ifirst;
        ASSERT_EQ(a+s, *(++i2));
        ASSERT_EQ(a+s, *i2);
        ASSERT_EQ(a,   *(--i2));
        ASSERT_EQ(a,   *i2);

        i1 += 1;
        ASSERT_EQ(a+s, *i1);
        i1 -= 1;
        ASSERT_EQ(a, *i1);

        ASSERT_EQ(a+s, *(ifirst+1));
        ASSERT_EQ(e-s, *(ilast-1));
    }

    std::vector<T> v_gt;
    for (T x = a; x < b; x += s) v_gt.push_back(x);
    ASSERT_TRUE(v_gt.size() == rgn.size());

    std::vector<T> v1;
    for (auto x: rgn) v1.push_back(x);
    ASSERT_EQ(v_gt, v1);

    std::vector<T> v2(ifirst, ilast);
    ASSERT_EQ(v_gt, v2);
}



TEST(ValueRanges, IntRanges) {

    using irange = clue::value_range<int>;
    ASSERT_TRUE((std::is_same<irange::difference_type, int>::value));

    test_unit_range(vrange(0, 0), 0, 0);
    test_unit_range(vrange(5, 5), 5, 5);
    test_unit_range(vrange(8),    0, 8);
    test_unit_range(vrange(3, 8), 3, 8);
}

TEST(ValueRanges, SizeRanges) {
    using std::size_t;
    using srange = clue::value_range<std::size_t>;
    ASSERT_TRUE((std::is_same<srange::difference_type, std::ptrdiff_t>::value));

    test_unit_range(srange(0, 0), size_t(0), size_t(0));
    test_unit_range(srange(5, 5), size_t(5), size_t(5));
    test_unit_range(srange(3, 8), size_t(3), size_t(8));
}

TEST(ValueRanges, DoubleRanges) {
    using drange = clue::value_range<double>;
    ASSERT_TRUE((std::is_same<drange::difference_type, double>::value));

    test_unit_range(drange(0.0, 0.0), 0.0, 0.0);
    test_unit_range(drange(5.0, 5.0), 5.0, 5.0);
    test_unit_range(drange(3.0, 8.0), 3.0, 8.0);
}

TEST(ValueRanges, CharRanges) {
    using crange = clue::value_range<char>;
    ASSERT_TRUE((std::is_same<crange::difference_type, char>::value));

    test_unit_range(crange('\0', '\0'), '\0', '\0');
    test_unit_range(crange('a', 'a'), 'a', 'a');
    test_unit_range(crange('a', 'g'), 'a', 'g');
}

TEST(ValueRanges, Equality) {
    using irange = clue::value_range<int>;

    ASSERT_EQ(true,  irange(2, 5) == irange(2, 5));
    ASSERT_EQ(false, irange(2, 5) != irange(2, 5));

    ASSERT_EQ(false, irange(2, 5) == irange(2, 6));
    ASSERT_EQ(true,  irange(2, 5) != irange(2, 6));

    ASSERT_EQ(false, irange(2, 5) == irange(3, 5));
    ASSERT_EQ(true,  irange(2, 5) != irange(3, 5));

    ASSERT_EQ(false, irange(2, 5) == irange(3, 6));
    ASSERT_EQ(true,  irange(2, 5) != irange(3, 6));
}

TEST(ValueRanges, Indices) {
    using srange = clue::value_range<std::size_t>;
    using clue::indices;

    std::vector<int> s0;
    ASSERT_EQ(srange(0, 0), indices(s0));

    std::vector<int> s1{1, 2, 3};
    ASSERT_EQ(srange(0, 3), indices(s1));
}

TEST(ValueRanges, StlAlgorithms) {
    // verify that it works well with STL algorithms

    clue::value_range<int> rgn(3, 8); // 3, ..., 7

    auto i_min = std::min_element(rgn.begin(), rgn.end());
    ASSERT_EQ(3, *i_min);

    auto i_max = std::max_element(rgn.begin(), rgn.end());
    ASSERT_EQ(7, *i_max);

    auto i1 = std::find(rgn.begin(), rgn.end(), 5);
    ASSERT_EQ(5, *i1);

    auto i2 = std::find(rgn.begin(), rgn.end(), 9);
    ASSERT_TRUE(i2 == rgn.end());

    auto i3 = std::find_if(rgn.begin(), rgn.end(),
            [](int x){ return x > 5; });
    ASSERT_EQ(6, *i3);

    auto r3 = std::count_if(rgn.begin(), rgn.end(),
            [](int x){ return x > 5; });
    ASSERT_EQ(2, r3);

    std::vector<int> tr(rgn.size(), 0);
    std::transform(rgn.begin(), rgn.end(), tr.begin(),
            [](int x){ return x * x; });
    std::vector<int> tr_r{9, 16, 25, 36, 49};
    ASSERT_EQ(tr_r, tr);
}


TEST(SteppedRanges, Basics) {
    using std::size_t;
    using srange = clue::stepped_value_range<std::size_t, std::size_t>;

    test_stepped_range(srange(0, 0, 1), size_t(0), size_t(0), size_t(1));
    test_stepped_range(srange(5, 5, 1), size_t(5), size_t(5), size_t(1));

    std::vector<size_t> steps38{1, 2, 3, 4, 5, 6};
    for (size_t s: steps38) {
        test_stepped_range(srange(3, 8, s), size_t(3), size_t(8), s);
    }

    std::vector<size_t> steps28{1, 2, 3, 4, 5, 6};
    for (size_t s: steps28) {
        test_stepped_range(srange(2, 8, s), size_t(2), size_t(8), s);
    }
}
