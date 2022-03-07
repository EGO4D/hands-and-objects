#include <clue/reindexed_view.hpp>
#include <clue/value_range.hpp>
#include <vector>
#include <gtest/gtest.h>


std::vector<int> S{12, 24, 36, 48, 60};
typedef clue::reindexed_view<const std::vector<int>, const std::vector<size_t>> cview_t;
typedef clue::reindexed_view<      std::vector<int>, const std::vector<size_t>>  view_t;

TEST(ReindexedView, Types) {
    ASSERT_TRUE((std::is_same<view_t::value_type, int  >::value));
    ASSERT_TRUE((std::is_same<view_t::reference,  int& >::value));
    ASSERT_TRUE((std::is_same<view_t::pointer,    int* >::value));

    ASSERT_TRUE((std::is_same<cview_t::value_type, int        >::value));
    ASSERT_TRUE((std::is_same<cview_t::reference,  const int& >::value));
    ASSERT_TRUE((std::is_same<cview_t::pointer,    const int* >::value));

    using cview_citerator = cview_t::const_iterator;
    using cview_iterator  = cview_t::iterator;
    using  view_citerator = view_t::const_iterator;
    using  view_iterator =  view_t::iterator;

    ASSERT_TRUE((std::is_same<cview_citerator::value_type, int        >::value));
    ASSERT_TRUE((std::is_same<cview_citerator::reference,  const int& >::value));
    ASSERT_TRUE((std::is_same<cview_citerator::pointer,    const int* >::value));

    ASSERT_TRUE((std::is_same<cview_iterator::value_type, int        >::value));
    ASSERT_TRUE((std::is_same<cview_iterator::reference,  const int& >::value));
    ASSERT_TRUE((std::is_same<cview_iterator::pointer,    const int* >::value));

    ASSERT_TRUE((std::is_same<view_citerator::value_type, int        >::value));
    ASSERT_TRUE((std::is_same<view_citerator::reference,  const int& >::value));
    ASSERT_TRUE((std::is_same<view_citerator::pointer,    const int* >::value));

    ASSERT_TRUE((std::is_same<view_iterator::value_type, int  >::value));
    ASSERT_TRUE((std::is_same<view_iterator::reference,  int& >::value));
    ASSERT_TRUE((std::is_same<view_iterator::pointer,    int* >::value));
}


TEST(ReindexedView, Empty) {
    std::vector<size_t> inds;

    view_t v0(S, inds);
    ASSERT_EQ(0, v0.size());
    ASSERT_EQ(true, v0.empty());
    ASSERT_THROW(v0.at(0), std::out_of_range);

    cview_t v0c(S, inds);
    ASSERT_EQ(0, v0c.size());
    ASSERT_EQ(true, v0c.empty());
    ASSERT_THROW(v0c.at(0), std::out_of_range);
}

TEST(ReindexedView, ConstView) {
    const std::vector<int>& src = S;
    std::vector<size_t> inds{2, 1, 3};
    cview_t v(src, inds);

    ASSERT_EQ(inds.size(), v.size());
    ASSERT_EQ(false, v.empty());

    for (size_t i = 0; i < inds.size(); ++i) {
        ASSERT_EQ(src[inds[i]], v.at(i));
        ASSERT_EQ(src[inds[i]], v[i]);
    }
    ASSERT_THROW(v.at(inds.size()), std::out_of_range);

    ASSERT_EQ(src[inds.front()], v.front());
    ASSERT_EQ(src[inds.back()], v.back());

    std::vector<size_t> inds_bad{100};
    cview_t v_bad(src, inds_bad);
    ASSERT_THROW(v_bad.at(0), std::out_of_range);
}

TEST(ReindexedView, MutableView) {
    std::vector<int> src(S);
    std::vector<size_t> inds{2, 1, 3};
    view_t v(src, inds);

    ASSERT_EQ(inds.size(), v.size());
    ASSERT_EQ(false, v.empty());

    for (size_t i = 0; i < inds.size(); ++i) {
        ASSERT_EQ(src[inds[i]], v.at(i));
        ASSERT_EQ(src[inds[i]], v[i]);
    }
    ASSERT_THROW(v.at(inds.size()), std::out_of_range);

    ASSERT_EQ(src[inds.front()], v.front());
    ASSERT_EQ(src[inds.back()], v.back());

    view_t v2(src, inds);
    v2[0] = -123;
    v2[1] = -456;
    ASSERT_EQ(-123, src[inds[0]]);
    ASSERT_EQ(-456, src[inds[1]]);
}

TEST(ReindexedView, ReadonlyIterations) {
    const std::vector<int>& src = S;
    std::vector<size_t> inds{2, 1, 3};
    std::vector<int> r0{src[2], src[1], src[3]};

    cview_t v1(src, inds);
    std::vector<int> r1(v1.begin(), v1.end());
    ASSERT_EQ(r0, r1);
    std::vector<int> r1c(v1.cbegin(), v1.cend());
    ASSERT_EQ(r0, r1c);

    std::vector<int> r2;
    for (auto x: v1) r2.push_back(x);
    ASSERT_EQ(r0, r2);
}

TEST(ReindexedView, MutatingIterations) {
    std::vector<int> src(S);
    std::vector<size_t> inds{2, 1, 3};
    std::vector<int> r0{src[2], src[1], src[3]};

    view_t v1(src, inds);
    std::vector<int> r1(v1.begin(), v1.end());
    ASSERT_EQ(r0, r1);
    std::vector<int> r1c(v1.cbegin(), v1.cend());
    ASSERT_EQ(r0, r1c);

    std::vector<int> r2_r{src[2]+1, src[1]+1, src[3]+1};
    for (auto& x: v1) {
        x += 1;
    }
    std::vector<int> r2{src[2], src[1], src[3]};
    ASSERT_EQ(r2_r, r2);
}


TEST(ReindexedView, WithValueRange) {
    using irange = clue::value_range<std::size_t>;

    std::vector<int> src(S);
    irange inds(1, 4);
    auto v1 = reindexed(src, inds);

    std::vector<int> r1r{src[1], src[2], src[3]};
    std::vector<int> r1(v1.begin(), v1.end());
    ASSERT_EQ(r1r, r1);

    v1[1] = -123;
    ASSERT_EQ(-123, src[2]);

    std::vector<int> r2r{src[1], -123, src[3]};
    std::vector<int> r2;
    for (auto x: v1) r2.push_back(x);
    ASSERT_EQ(r2r, r2);
}


TEST(ReindexedView, WithSteppedRange) {
    using srange = clue::stepped_value_range<std::size_t, std::size_t>;

    std::vector<int> src(S);
    srange inds(0, 5, 2);
    auto v1 = reindexed(src, inds);

    std::vector<int> r1r{src[0], src[2], src[4]};
    std::vector<int> r1(v1.begin(), v1.end());
    ASSERT_EQ(r1r, r1);

    v1[1] = -456;
    ASSERT_EQ(-456, src[2]);

    std::vector<int> r2r{src[0], -456, src[4]};
    std::vector<int> r2;
    for (auto x: v1) r2.push_back(x);
    ASSERT_EQ(r2r, r2);
}
