#include <clue/misc.hpp>
#include <gtest/gtest.h>

TEST(Misc, MakeUnique) {
    using clue::make_unique;

    auto p = make_unique<std::string>("abc");
    static_assert(std::is_same<decltype(p), std::unique_ptr<std::string>>::value,
            "clue::make_unique yields wrong type.");

    ASSERT_TRUE((bool)p);
    ASSERT_EQ("abc", *p);
}


TEST(Misc, TempBuffer) {
    using clue::temporary_buffer;

    temporary_buffer<int> buf(12);
    ASSERT_TRUE(buf.data() != nullptr);
    ASSERT_GE(buf.capacity(), 12);
}
