#include <clue/type_name.hpp>
#include <gtest/gtest.h>

using namespace clue;

#ifdef CLUE_HAS_DEMANGLE

struct MyType {};

template<class... Args>
struct MyTemplate{};

TEST(TypeNames, BasicTypes) {
    ASSERT_TRUE(has_demangle());

    ASSERT_EQ("int", type_name<int>());
    ASSERT_EQ("double", type_name<double>());
    ASSERT_EQ("MyType", type_name(MyType{}));
}

TEST(TypeNames, TemplateTypes) {
    ASSERT_EQ("MyTemplate<int>", type_name<MyTemplate<int>>());
    ASSERT_EQ("MyTemplate<int, double>", (type_name<MyTemplate<int,double>>()));
}


#endif // CLUE_HAS_DEMANGLE
