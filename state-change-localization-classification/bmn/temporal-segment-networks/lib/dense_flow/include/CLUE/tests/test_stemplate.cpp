#include <clue/stemplate.hpp>
#include <gtest/gtest.h>
#include <unordered_map>

using namespace clue;

TEST(STemplate, Basics) {

    std::unordered_map<std::string, std::string> dict;
    dict["a"] = "Alice";
    dict["b"] = "Bob";
    dict["c"] = "Cavin";

    stemplate s0("");
    ASSERT_EQ("", s0.with(dict).str());

    stemplate s1("xyz");
    ASSERT_EQ("xyz", s1.with(dict).str());

    stemplate s2("{{a}}");
    ASSERT_EQ("Alice", s2.with(dict).str());

    stemplate s2a("{{ a }}");
    ASSERT_EQ("Alice", s2a.with(dict).str());

    stemplate s3("call {{ a }}");
    ASSERT_EQ("call Alice", s3.with(dict).str());

    stemplate s4("{{a}}{{ b }}");
    ASSERT_EQ("AliceBob", s4.with(dict).str());

    stemplate s5("[{{ a }} -> {{b}}.{{c}}]");
    ASSERT_EQ("[Alice -> Bob.Cavin]", s5.with(dict).str());

    stemplate s6("{{a}} {} {{b}}");
    ASSERT_EQ("Alice {} Bob", s6.with(dict).str());

    stemplate s_err("{{a}} + {{d}}.");
    ASSERT_THROW(s_err.with(dict).str(), std::out_of_range);
}
