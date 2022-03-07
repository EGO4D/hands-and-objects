#include <clue/array_view.hpp>
#include <gtest/gtest.h>
#include <vector>
#include <type_traits>

using clue::array_view;

TEST(ArrayView, Empty) {

    array_view<int> v;

    ASSERT_EQ(nullptr, v.data());
    ASSERT_EQ(0,       v.size());
    ASSERT_EQ(true,    v.empty());

    ASSERT_THROW(v.at(0), std::out_of_range);

    ASSERT_TRUE(v.cbegin() == v.begin());
    ASSERT_TRUE(v.cend()   == v.end());
    ASSERT_TRUE(v.cbegin() == v.cend());
    ASSERT_TRUE(v.begin()  == v.end());
}


TEST(ArrayView, MutableView) {

    const size_t len = 5;
    int s[len] = {12, 24, 36, 48, 60};
    array_view<int> v(s, len);

    ASSERT_EQ(s,     v.data());
    ASSERT_EQ(len,   v.size());
    ASSERT_EQ(false, v.empty());

    for (size_t i = 0; i < len; ++i) {
        ASSERT_EQ(s[i], v[i]);
        ASSERT_EQ(s[i], v.at(i));
    }
    ASSERT_THROW(v.at(len), std::out_of_range);

    ASSERT_EQ(s[0],     v.front());
    ASSERT_EQ(s[len-1], v.back());
    ASSERT_EQ(s,         &(v.front()));
    ASSERT_EQ(s+(len-1), &(v.back()));

    v[2] = -1;
    ASSERT_EQ(-1, s[2]);

    v.at(3) = -2;
    ASSERT_EQ(-2, s[3]);

    ASSERT_TRUE(v.cbegin() == v.begin());
    ASSERT_TRUE(v.cend()   == v.end());
    ASSERT_TRUE(v.cbegin() + len == v.cend());
    ASSERT_TRUE(v.begin()  + len == v.end());
}


TEST(ArrayView, ConstView) {

    const size_t len = 5;
    int s[len] = {12, 24, 36, 48, 60};
    array_view<const int> v(s, len);
    ASSERT_TRUE((std::is_same<array_view<const int>::value_type, int>::value));

    ASSERT_EQ(s,     v.data());
    ASSERT_EQ(len,   v.size());
    ASSERT_EQ(false, v.empty());

    for (size_t i = 0; i < len; ++i) {
        ASSERT_EQ(s[i], v[i]);
        ASSERT_EQ(s[i], v.at(i));
    }
    ASSERT_THROW(v.at(len), std::out_of_range);

    ASSERT_EQ(s[0],     v.front());
    ASSERT_EQ(s[len-1], v.back());
    ASSERT_EQ(s,         &(v.front()));
    ASSERT_EQ(s+(len-1), &(v.back()));

    ASSERT_TRUE(v.cbegin() == v.begin());
    ASSERT_TRUE(v.cend()   == v.end());
    ASSERT_TRUE(v.cbegin() + len == v.cend());
    ASSERT_TRUE(v.begin()  + len == v.end());
}


TEST(ArrayView, Iterations) {

    const size_t len = 5;
    int s[len] = {12, 24, 36, 48, 60};

    std::vector<int> v0 {12, 24, 36, 48, 60};
    std::vector<int> vr0{60, 48, 36, 24, 12};

    array_view<int> cv(s, len);
    std::vector<int> v1(cv.begin(), cv.end());
    ASSERT_EQ(v0, v1);

    std::vector<int> v1c(cv.cbegin(), cv.cend());
    ASSERT_EQ(v0, v1c);

    std::vector<int> vr1(cv.rbegin(), cv.rend());
    ASSERT_EQ(vr0, vr1);

    std::vector<int> vr1c(cv.crbegin(), cv.crend());
    ASSERT_EQ(vr0, vr1c);

    std::vector<int> v2;
    for (auto x: cv) v2.push_back(x);
    ASSERT_EQ(v0, v2);
}


TEST(ArrayView, Aview) {
    using clue::aview;

    std::vector<int> s {12, 24, 36, 48, 60};
    const std::vector<int>& cs = s;

    auto v = aview(s.data(), s.size());
    ASSERT_TRUE((std::is_same<decltype(v), array_view<int>>::value));
    ASSERT_EQ(s.data(), v.data());
    ASSERT_EQ(s.size(), v.size());

    auto cv = aview(cs.data(), cs.size());
    ASSERT_TRUE((std::is_same<decltype(cv), array_view<const int>>::value));
    ASSERT_EQ(cs.data(), v.data());
    ASSERT_EQ(cs.size(), v.size());
}
