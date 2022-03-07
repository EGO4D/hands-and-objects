#include <clue/stringex.hpp>
#include <gtest/gtest.h>
#include <limits>

using std::string;
using clue::string_view;

TEST(StringEx, StrView) {
    using clue::view;

    string s0;
    string_view v0 = view(s0);
    ASSERT_EQ(0, v0.size());

    string s1("abc");
    string_view v1 = view(s1);
    ASSERT_EQ(3, v1.size());
    ASSERT_EQ(s1.c_str(), v1.data());
}

TEST(StringEx, Prefix) {
    using clue::prefix;

    ASSERT_EQ(string_view(""),    prefix(string_view("abc"), 0));
    ASSERT_EQ(string_view("a"),   prefix(string_view("abc"), 1));
    ASSERT_EQ(string_view("ab"),  prefix(string_view("abc"), 2));
    ASSERT_EQ(string_view("abc"), prefix(string_view("abc"), 3));
    ASSERT_EQ(string_view("abc"), prefix(string_view("abc"), 4));

    ASSERT_EQ(string(""),    prefix(string("abc"), 0));
    ASSERT_EQ(string("a"),   prefix(string("abc"), 1));
    ASSERT_EQ(string("ab"),  prefix(string("abc"), 2));
    ASSERT_EQ(string("abc"), prefix(string("abc"), 3));
    ASSERT_EQ(string("abc"), prefix(string("abc"), 4));
}

TEST(StringEx, Suffix) {
    using clue::suffix;

    ASSERT_EQ(string_view(""),    suffix(string_view("abc"), 0));
    ASSERT_EQ(string_view("c"),   suffix(string_view("abc"), 1));
    ASSERT_EQ(string_view("bc"),  suffix(string_view("abc"), 2));
    ASSERT_EQ(string_view("abc"), suffix(string_view("abc"), 3));
    ASSERT_EQ(string_view("abc"), suffix(string_view("abc"), 4));

    ASSERT_EQ(string(""),    suffix(string("abc"), 0));
    ASSERT_EQ(string("c"),   suffix(string("abc"), 1));
    ASSERT_EQ(string("bc"),  suffix(string("abc"), 2));
    ASSERT_EQ(string("abc"), suffix(string("abc"), 3));
    ASSERT_EQ(string("abc"), suffix(string("abc"), 4));
}


template<typename T>
void test_starts_with_char() {
    using clue::starts_with;

    ASSERT_EQ(false, starts_with(T(""),   'a'));
    ASSERT_EQ(true,  starts_with(T("a"),  'a'));
    ASSERT_EQ(true,  starts_with(T("ab"), 'a'));
    ASSERT_EQ(false, starts_with(T("ba"), 'a'));
    ASSERT_EQ(false, starts_with(T("xy"), 'a'));
}
TEST(StringEx, StartsWithChar) {
    test_starts_with_char<const char*>();
    test_starts_with_char<string_view>();
    test_starts_with_char<string>();
}

template<typename T, typename S>
void test_starts_with() {
    using clue::starts_with;

    ASSERT_EQ(true,  starts_with(T(""), S("")));
    ASSERT_EQ(false, starts_with(T(""), S("a")));
    ASSERT_EQ(false, starts_with(T(""), S("abc")));

    ASSERT_EQ(true,  starts_with(T("abc"), S("")));
    ASSERT_EQ(true,  starts_with(T("abc"), S("ab")));
    ASSERT_EQ(true,  starts_with(T("abc"), S("abc")));
    ASSERT_EQ(false, starts_with(T("abc"), S("x")));
    ASSERT_EQ(false, starts_with(T("abc"), S("abd")));
    ASSERT_EQ(false, starts_with(T("abc"), S("abcd")));
}
TEST(StringEx, StartsWith) {
    test_starts_with<const char*, const char*>();
    test_starts_with<const char*, string_view>();
    test_starts_with<const char*, string>();

    test_starts_with<string_view, const char*>();
    test_starts_with<string_view, string_view>();
    test_starts_with<string_view, string>();

    test_starts_with<string, const char*>();
    test_starts_with<string, string_view>();
    test_starts_with<string, string>();
}

template<typename T>
void test_ends_with_char() {
    using clue::ends_with;

    ASSERT_EQ(false, ends_with(T(""),   'a'));
    ASSERT_EQ(true,  ends_with(T("a"),  'a'));
    ASSERT_EQ(false, ends_with(T("ab"), 'a'));
    ASSERT_EQ(true,  ends_with(T("ba"), 'a'));
    ASSERT_EQ(false, ends_with(T("xy"), 'a'));
    ASSERT_EQ(true,  ends_with(T("xyza"), 'a'));
}
TEST(StringEx, EndsWithChar) {
    test_ends_with_char<const char*>();
    test_ends_with_char<string_view>();
    test_ends_with_char<string>();
}


template<typename T, typename S>
void test_ends_with() {
    using clue::ends_with;

    ASSERT_EQ(true,  ends_with(T(""), S("")));
    ASSERT_EQ(false, ends_with(T(""), S("a")));
    ASSERT_EQ(false, ends_with(T(""), S("abc")));

    ASSERT_EQ(true,  ends_with(T("abc"), S("")));
    ASSERT_EQ(true,  ends_with(T("abc"), S("bc")));
    ASSERT_EQ(true,  ends_with(T("abc"), S("abc")));
    ASSERT_EQ(false, ends_with(T("abc"), S("x")));
    ASSERT_EQ(false, ends_with(T("abc"), S("xbc")));
    ASSERT_EQ(false, ends_with(T("abc"), S("xabc")));
}
TEST(StringEx, EndsWith) {
    test_ends_with<const char*, const char*>();
    test_ends_with<const char*, string_view>();
    test_ends_with<const char*, string>();

    test_ends_with<string_view, const char*>();
    test_ends_with<string_view, string_view>();
    test_ends_with<string_view, string>();

    test_ends_with<string, const char*>();
    test_ends_with<string, string_view>();
    test_ends_with<string, string>();
}


template<typename T>
void test_trim() {
    using clue::trim_left;
    using clue::trim_right;
    using clue::trim;

    ASSERT_EQ(T(""), trim_left  (T("")));
    ASSERT_EQ(T(""), trim_right (T("")));
    ASSERT_EQ(T(""), trim       (T("")));

    ASSERT_EQ(T(""), trim_left  (T("\t\n")));
    ASSERT_EQ(T(""), trim_right (T("\t\n")));
    ASSERT_EQ(T(""), trim       (T("\t\n")));

    ASSERT_EQ(T("a"), trim_left  (T("a")));
    ASSERT_EQ(T("a"), trim_right (T("a")));
    ASSERT_EQ(T("a"), trim       (T("a")));

    ASSERT_EQ(T("abc"), trim_left  (T("abc")));
    ASSERT_EQ(T("abc"), trim_right (T("abc")));
    ASSERT_EQ(T("abc"), trim       (T("abc")));

    ASSERT_EQ(T("abc xy\n"),    trim_left  (T("abc xy\n")));
    ASSERT_EQ(T("abc xy"),      trim_right (T("abc xy\n")));
    ASSERT_EQ(T("abc xy"),      trim       (T("abc xy\n")));

    ASSERT_EQ(T("abc xy   \n"), trim_left  (T("abc xy   \n")));
    ASSERT_EQ(T("abc xy"),      trim_right (T("abc xy   \n")));
    ASSERT_EQ(T("abc xy"),      trim       (T("abc xy   \n")));

    ASSERT_EQ(T("abc xy"),      trim_left  (T("\t\tabc xy")));
    ASSERT_EQ(T("\t\tabc xy"),  trim_right (T("\t\tabc xy")));
    ASSERT_EQ(T("abc xy"),      trim       (T("\t\tabc xy")));

    ASSERT_EQ(T("abc xy\n"),    trim_left  (T("\t\tabc xy\n")));
    ASSERT_EQ(T("\t\tabc xy"),  trim_right (T("\t\tabc xy\n")));
    ASSERT_EQ(T("abc xy"),      trim       (T("\t\tabc xy\n")));
}


TEST(StringEx, Trim) {
    test_trim<string_view>();
    test_trim<string>();
}


template<typename T>
void test_try_parse(const char *sz, bool expect_ret, T expect_val) {
    using clue::try_parse;

    T x = static_cast<T>(0);
    ASSERT_EQ(expect_ret, try_parse(sz, x));
    if (expect_ret) {
        ASSERT_EQ(expect_val, x);
    } else {
        ASSERT_EQ(0, x);
    }

    x = static_cast<T>(0);
    ASSERT_EQ(expect_ret, try_parse(clue::string_view(sz), x));
    if (expect_ret) {
        ASSERT_EQ(expect_val, x);
    } else {
        ASSERT_EQ(0, x);
    }

    x = static_cast<T>(0);
    ASSERT_EQ(expect_ret, try_parse(std::string(sz), x));
    if (expect_ret) {
        ASSERT_EQ(expect_val, x);
    } else {
        ASSERT_EQ(0, x);
    }
}

TEST(StringEx, TryParseBool) {
    bool t = true;
    bool f = false;

    test_try_parse<bool>("0", true, f);
    test_try_parse<bool>("1", true, t);

    test_try_parse<bool>("t", true, t);
    test_try_parse<bool>("f", true, f);
    test_try_parse<bool>("T", true, t);
    test_try_parse<bool>("F", true, f);

    test_try_parse<bool>("true",  true, t);
    test_try_parse<bool>("false", true, f);
    test_try_parse<bool>("True",  true, t);
    test_try_parse<bool>("False", true, f);
    test_try_parse<bool>("TRUE",  true, t);
    test_try_parse<bool>("FALSE", true, f);

    test_try_parse<bool>("  0  ", true, f);
    test_try_parse<bool>("  1  ", true, t);
    test_try_parse<bool>("  true\n",  true, t);
    test_try_parse<bool>("  false\n", true, f);

    test_try_parse<bool>("tr", false, f);
    test_try_parse<bool>("fa", false, f);
    test_try_parse<bool>("truee",  false, f);
    test_try_parse<bool>("falsee", false, f);
}


TEST(StringEx, TryParseInt) {
    test_try_parse<int>("123",     true, 123);
    test_try_parse<int>("  123",   true, 123);
    test_try_parse<int>("-123  ",   true, -123);
    test_try_parse<int>("  -123\n", true, -123);

    test_try_parse<int>("",     false, 0);
    test_try_parse<int>("a123", false, 0);
    test_try_parse<int>("123a", false, 0);

    test_try_parse<long>("1234", true, 1234L);
    test_try_parse<long>("2147483647", true, 2147483647L);
    test_try_parse<long long>("9223372036854775807", true, 9223372036854775807LL);
}

TEST(StringEx, TryParseReal) {
    test_try_parse<float>("1.25",      true,  1.25f);
    test_try_parse<float>(" -1.25",    true, -1.25f);
    test_try_parse<double>("  1.25 ",  true,  1.25);
    test_try_parse<double>(" -1.25\n", true, -1.25);
    test_try_parse<double>("100\n", true, 100.0);

    test_try_parse<double>("",     false, 0);
    test_try_parse<double>("a123", false, 0);
    test_try_parse<double>("123a", false, 0);
}


TEST(StringEx, Tokenize) {

    const char *cstr = "abc ef 1234 xyz";
    string_view sv(cstr, 10);   // "abc ef 123"
    std::string str = sv.to_string();
    std::vector<std::string> tks0{"abc", "ef", "1234", "xyz"};
    std::vector<std::string> tks1{"abc", "ef", "123"};

    std::vector<std::string> v;
    auto f = [&](const char *p, size_t n) {
        v.push_back(std::string(p, n));
        return true;
    };

    v.clear();
    clue::foreach_token_of(cstr, ' ', f);
    ASSERT_EQ(tks0, v);

    v.clear();
    clue::foreach_token_of(cstr, " ", f);
    ASSERT_EQ(tks0, v);

    v.clear();
    clue::foreach_token_of(sv, ' ', f);
    ASSERT_EQ(tks1, v);

    v.clear();
    clue::foreach_token_of(sv, " ", f);
    ASSERT_EQ(tks1, v);

    v.clear();
    clue::foreach_token_of(str, ' ', f);
    ASSERT_EQ(tks1, v);

    v.clear();
    clue::foreach_token_of(str, " ", f);
    ASSERT_EQ(tks1, v);

    const char *xstr = " abc ; xy, uvw ,";

    v.clear();
    clue::foreach_token_of(xstr, ";, ", f);
    std::vector<std::string> tks2{"abc", "xy", "uvw"};
    ASSERT_EQ(tks2, v);
}
