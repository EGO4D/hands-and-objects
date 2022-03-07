#include <clue/string_view.hpp>
#include <gtest/gtest.h>
#include <sstream>
#include <iomanip>

namespace stdx = clue;

using stdx::string_view;


void test_strview_basics(const string_view& sv, const char *p, size_t n) {
    ASSERT_EQ(p, sv.data());
    ASSERT_EQ(n, sv.size());
    ASSERT_EQ(n, sv.length());
    ASSERT_EQ((n == 0), sv.empty());

    ASSERT_EQ(p, sv.cbegin());
    ASSERT_EQ(p, sv.begin());
    ASSERT_EQ(p + n, sv.cend());
    ASSERT_EQ(p + n, sv.end());

    using reviter_t = std::reverse_iterator<string_view::const_iterator>;

    ASSERT_EQ(reviter_t(sv.end()),    sv.rbegin());
    ASSERT_EQ(reviter_t(sv.begin()),  sv.rend());
    ASSERT_EQ(reviter_t(sv.cend()),   sv.crbegin());
    ASSERT_EQ(reviter_t(sv.cbegin()), sv.crend());

    for (size_t i = 0; i < n; ++i) {
        ASSERT_EQ(p[i], sv[i]);
        ASSERT_EQ(p[i], sv.at(i));
    }
    ASSERT_THROW(sv.at(n), std::out_of_range);
    ASSERT_THROW(sv.at(string_view::npos), std::out_of_range);

    if (n > 0) {
        ASSERT_EQ(p,         &(sv.front()));
        ASSERT_EQ(p + (n-1), &(sv.back()));
        ASSERT_EQ(p[0],        sv.front());
        ASSERT_EQ(p[n-1],      sv.back());
    }
}


TEST(StringView, Basics) {

    test_strview_basics(string_view(), nullptr, 0);

    const char *sz1 = "abcde";
    test_strview_basics(string_view(sz1), sz1, 5);
    test_strview_basics(string_view(sz1, 3), sz1, 3);

    std::string s1(sz1);
    test_strview_basics(string_view(s1), s1.data(), 5);

    string_view sv2("xyz");
    test_strview_basics(string_view(sv2), sv2.data(), 3);
}

TEST(StringView, Swap) {

    string_view sv0;
    const char *sz1 = "abc";
    string_view sv1(sz1);

    swap(sv0, sv1);
    ASSERT_EQ(sz1, sv0.data());
    ASSERT_EQ(3, sv0.size());
    ASSERT_EQ(nullptr, sv1.data());
    ASSERT_EQ(0, sv1.size());
}


TEST(StringView, Conversion) {

    string_view sv1("abcde");
    ASSERT_EQ("abcde", sv1.to_string());
    ASSERT_EQ("abcde", (std::string)(sv1));

    string_view sv2(sv1.data(), 3);
    ASSERT_EQ("abc", sv2.to_string());
    ASSERT_EQ("abc", (std::string)(sv2));
}


TEST(StringView, Modifiers) {

    using std::swap;

    string_view sv1("abcd");
    string_view sv2("xyz");

    sv1.swap(sv2);

    ASSERT_EQ("xyz",  sv1.to_string());
    ASSERT_EQ("abcd", sv2.to_string());

    swap(sv1, sv2);

    ASSERT_EQ("abcd", sv1.to_string());
    ASSERT_EQ("xyz",  sv2.to_string());

    sv1.remove_prefix(2);
    ASSERT_EQ("cd", sv1.to_string());

    sv2.remove_suffix(1);
    ASSERT_EQ("xy", sv2.to_string());
}


TEST(StringView, Copy) {

    char s[5] = {'w', 'x', 'y', 'z', '\0'};
    string_view sv2("abcd");

    size_t r = sv2.copy(s, 4);
    ASSERT_EQ(4, r);
    ASSERT_EQ("abcd", std::string(s));

    r = sv2.copy(s, 3, 1);
    ASSERT_EQ(3, r);
    ASSERT_EQ("bcdd", std::string(s));
}


TEST(StringView, Substr) {

    string_view s0("abcd=xyz");

    ASSERT_EQ("abcd=xyz", s0.substr().to_string());
    ASSERT_EQ("abcd",     s0.substr(0, 4).to_string());
    ASSERT_EQ("xyz",      s0.substr(5).to_string());
    ASSERT_EQ("xyz",      s0.substr(5, 100).to_string());
    ASSERT_EQ("cd=xy",    s0.substr(2, 5).to_string());
}


void test_strview_compare(const string_view& a, const string_view& b) {

    std::string as = a.to_string();
    std::string bs = b.to_string();

    auto sg = [](int x) { return x == 0 ? 0 : (x < 0 ? -1 : 1); };
    int c = as.compare(bs);

    ASSERT_EQ(sg(c), sg(a.compare(b)));
    ASSERT_EQ(sg(c), sg(a.compare(bs.c_str())));

    ASSERT_EQ(c == 0, a == b);
    ASSERT_EQ(c != 0, a != b);
    ASSERT_EQ(c <  0, a <  b);
    ASSERT_EQ(c <= 0, a <= b);
    ASSERT_EQ(c >  0, a >  b);
    ASSERT_EQ(c >= 0, a >= b);

    ASSERT_EQ((-c) == 0, b == a);
    ASSERT_EQ((-c) != 0, b != a);
    ASSERT_EQ((-c) <  0, b <  a);
    ASSERT_EQ((-c) <= 0, b <= a);
    ASSERT_EQ((-c) >  0, b >  a);
    ASSERT_EQ((-c) >= 0, b >= a);
}

TEST(StringView, Compare) {

    string_view s0;
    string_view s1("abcd");
    string_view s2("abcde");
    string_view s3("xyz");
    string_view s4("abdc");

    test_strview_compare(s0, s1);
    test_strview_compare(s1, s2);
    test_strview_compare(s1, s3);
    test_strview_compare(s1, s4);
}


TEST(StringView, FindChars) {

    string_view s("abcdabc");

    ASSERT_EQ(0, s.find('a'));
    ASSERT_EQ(1, s.find('b'));
    ASSERT_EQ(2, s.find('c'));
    ASSERT_EQ(3, s.find('d'));
    ASSERT_EQ(3, s.find('d', 3));

    size_t npos = string_view::npos;
    ASSERT_EQ(4, s.find('a', 4));
    ASSERT_EQ(5, s.find('b', 4));
    ASSERT_EQ(6, s.find('c', 4));
    ASSERT_EQ(npos, s.find('d', 4));

    ASSERT_EQ(0,    s.find_first_of("abc"));
    ASSERT_EQ(npos, s.find_first_of("xyz"));
    ASSERT_EQ(0,    s.find_first_of(string_view("abc")));
    ASSERT_EQ(npos, s.find_first_of(string_view("xyz")));

    ASSERT_EQ(4,    s.find_first_of("abc", 4));
    ASSERT_EQ(npos, s.find_first_of("xyz", 4));
    ASSERT_EQ(4,    s.find_first_of(string_view("abc"), 4));
    ASSERT_EQ(npos, s.find_first_of(string_view("xyz"), 4));

    ASSERT_EQ(3,    s.find_first_not_of("abc"));
    ASSERT_EQ(0,    s.find_first_not_of("xyz"));
    ASSERT_EQ(3,    s.find_first_not_of(string_view("abc")));
    ASSERT_EQ(0,    s.find_first_not_of(string_view("xyz")));

    ASSERT_EQ(npos, s.find_first_not_of("abc", 4));
    ASSERT_EQ(4,    s.find_first_not_of("xyz", 4));
    ASSERT_EQ(npos, s.find_first_not_of(string_view("abc"), 4));
    ASSERT_EQ(4,    s.find_first_not_of(string_view("xyz"), 4));
}


TEST(StringView, RfindChars) {

    string_view s("abcdabc");
    size_t npos = string_view::npos;

    ASSERT_EQ(4, s.rfind('a'));
    ASSERT_EQ(5, s.rfind('b'));
    ASSERT_EQ(6, s.rfind('c'));
    ASSERT_EQ(3, s.rfind('d'));

    ASSERT_EQ(0,    s.rfind('a', 0));
    ASSERT_EQ(npos, s.rfind('d', 0));
    ASSERT_EQ(3,    s.rfind('d', 3));

    ASSERT_EQ(6,    s.find_last_of("abc"));
    ASSERT_EQ(npos, s.find_last_of("xyz"));
    ASSERT_EQ(6,    s.find_last_of(string_view("abc")));
    ASSERT_EQ(npos, s.find_last_of(string_view("xyz")));

    ASSERT_EQ(2,    s.find_last_of("abc", 3));
    ASSERT_EQ(npos, s.find_last_of("xyz", 3));
    ASSERT_EQ(2,    s.find_last_of(string_view("abc"), 3));
    ASSERT_EQ(npos, s.find_last_of(string_view("xyz"), 3));

    ASSERT_EQ(3,    s.find_last_not_of("abc"));
    ASSERT_EQ(6,    s.find_last_not_of("xyz"));
    ASSERT_EQ(3,    s.find_last_not_of(string_view("abc")));
    ASSERT_EQ(6,    s.find_last_not_of(string_view("xyz")));

    ASSERT_EQ(3,    s.find_last_not_of("abc", 3));
    ASSERT_EQ(3,    s.find_last_not_of("xyz", 3));
    ASSERT_EQ(npos, s.find_last_not_of("abcd", 3));
    ASSERT_EQ(3,    s.find_last_not_of(string_view("abc"), 3));
    ASSERT_EQ(3,    s.find_last_not_of(string_view("xyz"), 3));
    ASSERT_EQ(npos, s.find_last_not_of(string_view("abcd"), 3));
}


TEST(StringView, FindSubstr) {

    string_view s("abcdabc");
    size_t npos = string_view::npos;

    ASSERT_EQ(0,    s.find("abc"));
    ASSERT_EQ(4,    s.find("abc", 1));
    ASSERT_EQ(4,    s.find("abc", 3));
    ASSERT_EQ(0,    s.find("abcd"));
    ASSERT_EQ(npos, s.find("abcd", 1));
    ASSERT_EQ(npos, s.find("xyz"));

    ASSERT_EQ(0,    s.find(string_view("abc")));
    ASSERT_EQ(4,    s.find(string_view("abc"), 1));
    ASSERT_EQ(4,    s.find(string_view("abc"), 3));
    ASSERT_EQ(0,    s.find(string_view("abcd")));
    ASSERT_EQ(npos, s.find(string_view("abcd"), 1));
    ASSERT_EQ(npos, s.find(string_view("xyz")));

    ASSERT_EQ(0,    s.find("abc",  0, 3));
    ASSERT_EQ(4,    s.find("abc",  1, 3));
    ASSERT_EQ(4,    s.find("abc",  3, 3));
    ASSERT_EQ(0,    s.find("abcd", 0, 4));
    ASSERT_EQ(4,    s.find("abcd", 1, 3));
    ASSERT_EQ(npos, s.find("abcd", 1, 4));
    ASSERT_EQ(npos, s.find("xyz",  0, 3));
}


TEST(StringView, RfindSubstr) {

    string_view s("abcdabc");
    size_t npos = string_view::npos;

    ASSERT_EQ(4,    s.rfind("abc"));
    ASSERT_EQ(npos, s.rfind("xyz"));
    ASSERT_EQ(3,    s.rfind("dabc"));
    ASSERT_EQ(3,    s.rfind("dabc", 3));
    ASSERT_EQ(npos, s.rfind("dabc", 2));

    ASSERT_EQ(4,    s.rfind(string_view("abc")));
    ASSERT_EQ(npos, s.rfind(string_view("xyz")));
    ASSERT_EQ(3,    s.rfind(string_view("dabc")));
    ASSERT_EQ(3,    s.rfind(string_view("dabc"), 3));
    ASSERT_EQ(npos, s.rfind(string_view("dabc"), 2));

    ASSERT_EQ(4,    s.rfind("abc",  npos, 3));
    ASSERT_EQ(npos, s.rfind("xyz",  npos, 3));
    ASSERT_EQ(3,    s.rfind("dabc", npos, 4));
    ASSERT_EQ(3,    s.rfind("dabc", 3,    4));
    ASSERT_EQ(npos, s.rfind("dabc", 2,    4));
}


TEST(StringView, StreamOutput) {
    string_view sv0;

    std::stringstream ss0;
    std::stringstream ss0_c;
    ss0 << sv0;
    ss0_c << sv0.to_string();
    ASSERT_EQ(ss0_c.str(), ss0.str());
    ASSERT_EQ("", ss0.str());

    string_view sv1("abc");

    std::stringstream ss1;
    std::stringstream ss1_c;
    ss1 << sv1;
    ss1_c << sv1.to_string();
    ASSERT_EQ(ss1_c.str(), ss1.str());
    ASSERT_EQ("abc", ss1.str());

    std::stringstream ss2;
    std::stringstream ss2_c;
    ss2 << std::setw(5) << sv1;
    ss2_c << std::setw(5) << sv1.to_string();
    ASSERT_EQ(ss2_c.str(), ss2.str());
    ASSERT_EQ("  abc", ss2.str());

    std::stringstream ss3;
    std::stringstream ss3_c;
    ss3 << std::setw(5) << std::left << std::setfill('*') << sv1;
    ss3_c << std::setw(5) << std::left << std::setfill('*') << sv1.to_string();
    ASSERT_EQ(ss3_c.str(), ss3.str());
    ASSERT_EQ("abc**", ss3.str());
}
