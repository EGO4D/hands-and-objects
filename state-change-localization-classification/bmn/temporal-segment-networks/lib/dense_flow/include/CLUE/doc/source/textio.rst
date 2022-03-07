Text IO
========

The library provides some convenient utilties to read/write text files. These functionalities are implemented in ``<clue/textio.hpp>``.

.. cpp:function:: std::string read_file_content(filename)

    Read all the content in a file into a string.

    Here, ``filename`` can be of type ``const char*`` or ``std::string``.

.. cpp:class:: line_stream

    Line stream class. It wraps a text string into a stream of lines. So one can iterate the lines using STL-style iterators.

    The iterated values are of type ``clue::string_view`` that provides a view into the part of the text corresponding to the current line. **Note:** The string view includes the line-delimiter ``'\n'``.

    The class has three constructors, respectively accepting a C-string toegther with a length, a C-string, or a standard C++ string.

**Example:** The following example reads text from a file, and print its lines with line number prefixes.

.. code-block:: cpp

    #include <clue/textio.hpp>
    #include <iostream>

    using clue::string_view;
    using clue::read_file_content;
    using clue::line_stream;

    int main() {
        std::string text = read_file_content("myfile.txt");

        line_stream lstr(text);
        size_t line_no = 0;
        for (string_view line: lstr) {
            std::cout << ++line_no << ": " << line;
        }
    }
