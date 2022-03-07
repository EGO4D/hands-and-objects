#include <clue/textio.hpp>
#include <clue/sformat.hpp>
#include <gtest/gtest.h>


const char* Text =
"Lorem Ipsum is simply dummy text of the printing and typesetting \n"
"industry. Lorem Ipsum has been the industry's standard dummy text \n"
"ever since the 1500s, when an unknown printer took a galley of type \n"
"and scrambled it to make a type specimen book. It has survived not \n"
"only five centuries, but also the leap into electronic typesetting, \n"
"remaining essentially unchanged. It was popularised in the 1960s with \n"
"the release of Letraset sheets containing Lorem Ipsum passages, and \n"
"more recently with desktop publishing software like Aldus PageMaker \n"
"including versions of Lorem Ipsum. \n";

TEST(TextIO, ReadFile) {
    std::string tname = clue::sstr(
        "/tmp/clue_test_textio_", time(NULL), ".txt");
    std::ofstream out(tname);
    out << Text;
    out.close();

    std::string rtext = clue::read_file_content(tname);
    ASSERT_EQ(Text, rtext);
}


TEST(TextIO, LineStream) {
    const char *text = "abc\n  efg  \n\nxyz\n12";
    clue::line_stream lstr(text);

    std::vector<std::string> lines(lstr.begin(), lstr.end());

    ASSERT_EQ(5, lines.size());

    ASSERT_EQ("abc\n", lines[0]);
    ASSERT_EQ("  efg  \n", lines[1]);
    ASSERT_EQ("\n", lines[2]);
    ASSERT_EQ("xyz\n", lines[3]);
    ASSERT_EQ("12", lines[4]);
}
