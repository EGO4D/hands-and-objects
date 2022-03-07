Extensions of String Functionalities
======================================

This library provides a set of functions to complement the methods of ``std::string`` (or ``string_view``). These functions are useful in many practical applications.

.. note::

    To be consistent with the standard, these extended functionalities are provided as global functions (within the namespace ``clue``) instead of member functions.

Combining string views together with these functionalities would make string analysis much easier. Before going into details, let's first look at a practical examples.

Suppose, we have a text file like this:

.. code-block:: python

    # This is a list of attribues
    # The symbol `#` is to indicate comments

    bar = 100, 20, 3
    foo = 13, 568, 24
    xyz = 75, 62, 39, 18

The following code snippet uses the functionalities provided in *CLUE++* to parse them into a list of records.

.. code-block:: cpp

    // a simple record class
    struct Record {
        std::string name;
        std::vector<int> nums;

        Record(const std::string& name) : name(name) {}

        void add(int v) {
            nums.push_back(v);
        }
    };

    inline std::ostream& operator << (std::ostream& os, const Record& r) {
        os << r.name << ": ";
        for (int v: r.nums) os << v << ' ';
        return os;
    }

    // the following code reads the file and parses the content
    using namespace clue;

    // open a file
    std::istringstream fin(filename)

    // get first line
    char buf[256];
    fin.getline(buf, 256);

    while (fin) {
        // construct a string view out of buffer,
        // and trim leading and trailing spaces
        auto sv = trim(string_view(buf));

        // process each line
        // ignoring empty lines or comments
        if (!sv.empty() && !starts_with(sv, '#')) {
            // locate '='
            size_t ieq = sv.find('=');

            // note: sub-string of a string view remains a view
            // no copying is done here
            auto name = trim(sv.substr(0, ieq));
            auto rhs = trim(sv.substr(ieq + 1));

            // construct a record
            Record record(name.to_string());

            // parse the each term of right-hand-side
            // by tokenizing
            foreach_token_of(rhs, ", ", [&](const char *p, size_t n){
                int v = 0;
                if (try_parse(string_view(p, n), v)) {
                    record.add(v);
                } else {
                    throw std::runtime_error("Invalid integer number.");
                }
                return true;
            });

            // print the record
            std::cout << record << std::endl;
        }

        // get next line
        fin.getline(buf, 256);
    }

In this code snippet, we utilize five aspects of functionalities in *CLUE++*:

- ``string_view``, which constructs a like-weight view (without making a copy) on a memory block to provide string-related API. For example, you can do ``sv.find(c)`` and ``sv.substr(...)``. Particularly, ``sv.substr(...)`` results in another string view of the sub-part, without making any copies.

- ``trim``, which yields another string view, with leading and trailing spaces excluded.

- ``starts_with``, which checks whether a string starts with a certain character of sub-string. *CLUE++* also provides ``ends_with`` to check the suffix, and ``prefix``/``suffix`` to extract the prefixes or suffixes.

- ``foreach_token_of``, which performs tokenization in a functional way. In particular, it allows a callback function/functor to process each token, instead of making string copies of all the tokens.

- ``try_parse``, which trys to parse a string into a numeric value, and returns whether the parsing succeeded.

For string views, please refer to :ref:`stringview` for detailed exposition. Below, we introduce other string-related functionalities provided by *CLUE++*.


Make string view
-----------------

.. cpp:function:: constexpr view(s)

    Make a view of a standard string ``s``.

    If ``s`` is of class ``std::basic_string<charT, Traits, Allocator>``, then the returned object will be of class ``basic_string_view<charT, Traits>``. In particular, if ``s`` is of class ``std::string``, the returned type would be ``string_view``.


Prefix and suffix
-------------------

.. cpp:function:: constexpr prefix(s, size_t n)

    Get a prefix (*i.e.* a substring that starts at ``0``), whose length is at most ``n``.

    :param s: The input string ``s``, which can be a standard string or a string view.
    :param n: The maximum length of the prefix.

    This is equivalent to ``s.substr(0, min(s.size(), n))``.

.. cpp:function:: constexpr suffix(s, size_t n)

    Get a suffix (*i.e.* a substring that ends at the end of ``s``), whose length is at most ``n``.

    :param s: The input string ``s``, which can be a standard string or a string view.
    :param n: The maximum length of the suffix.

    This is equivalent to ``s.substr(k, m)`` with ``m = min(s.size(), n)`` and ``k = s.size() - m``.

.. cpp:function:: bool starts_with(str, sub)

    Test whether a string ``str`` starts with a prefix ``sub``.

    Here, ``str`` and ``sub`` can be either a null-terminated C-string, a string view, or a standard string.

.. cpp:function:: bool ends_with(str, sub)

    Test whether a string ``str`` ends with a suffix ``sub``.

    Here, ``str`` and ``sub`` can be either a null-terminated C-string, a string view, or a standard string.


Trim strings
-------------

.. cpp:function:: trim(str)

    Trim both the leading and trailing spaces of ``str``, where ``str`` can be either a standard string or a string view.

    :return: the trimmed sub-string. It is a view when ``str`` is a string view, or a copy of the sub-string when ``str`` is an instance of a standard string.

.. cpp:function:: trim_left(str)

    Trim the leading spaces of ``str``, where ``str`` can be either a standard string or a string view.

    :return: the trimmed sub-string. It is a view when ``str`` is a string view, or a copy of the sub-string when ``str`` is an instance of a standard string.

.. cpp:function:: trim_right(str)

    Trim the trailing spaces of ``str``, where ``str`` can be either a standard string or a string view.

    :return: the trimmed sub-string. It is a view when ``str`` is a string view, or a copy of the sub-string when ``str`` is an instance of a standard string.

Parse values
-------------

.. cpp:function:: bool try_parse(str, T& v)

    Try to parse a given string ``str`` into a value ``v``. It returns whether the parsing succeeded.

    :param str:  The input string to be parsed, which can be either a C-string, a string view, or a standard string.
    :param v:    The output variable, which will be updated upon successful parsing.

    To be more specific, if the function succeeded in parsing the number (*i.e.* the given string is a valid number representation for type ``T``), the parsed value will be written to ``v`` and it returns ``true``, otherwise, it returns ``false`` (the value of ``v`` won't be altered upon failure).

    :note: Internally, this function may call ``strtol``, ``strtoll``, ``strtof``, or ``strtod``, depending on the type ``T``.

.. note::
    This function allows preceding and trailing spaces in ``str`` (for convenience in practice), meaning that ``"123"``, ``"123  "``, and ``"  123\n"``, etc are all considered valid when parsing an integer. However, empty strings, strings with spaces in the middle (*e.g.* ``123 456``), or strings with undesirable characters (*e.g.* ``123a``) are considered invalid.

    For integers, the function allows base-specific prefixes. For example, ``"0x1ab"`` are considered an integer in the  hexadecimal form, while ``"0123"`` are considered an integer in the octal form.

    For floating point numbers, both fixed decimal notation and scientific notation are supported.

    For boolean values, the function can recognize the following patterns: ``"0"`` and ``"1"``, ``"t"`` and ``"f"``, as well as ``"true"`` and ``"false"``. Here, the comparison with these patterns are case-insensitive.

**Examples:**

.. code-block:: cpp

    using namespace clue;

    int x;
    try_parse("123", x);   // x <- 123, returns true
    try_parse("a123", x);  // returns false (x is not updated)

    double y;
    try_parse("12.75", y);  // y <- 12.75, returns true

    bool z;
    try_parse("0", z);      // z <- false, returns true
    try_parse("false", z);  // z <- false, returns true
    try_parse("T", z);      // z <- true, returns true

    // in real codes, you may write this in case you don't really know
    // exactly what value type to expect

    auto s = get_some_string_from_text();
    bool x_bool;
    int x_int;
    double x_real;

    if (try_parse(s, x_bool)) {
        std::cout << "got a boolean value: " << x_bool << std::endl;
    } else if (try_parse(s, x_int)) {
        std::cout << "got an integer: " << x_int << std::endl;
    } else if (try_parse(s, x_real)) {
        std::cout << "got a real number: " << x_real << std::endl;
    } else {
        throw std::runtime_error("Can't recognize the value!");
    }

Tokenize
---------

Extracting tokens from a string is a basic and important task in many text processing applications. ANSI C provides a ``strtok`` function for tokenizing, which, however, will destruct the source string. Some tokenizing functions in other libraries may return a vector of strings. This way involves making copies of all extracted tokens, which is often unnecessary.

In this library, we provide tokenizing functions in a new form that takes advantage of the lambda functions introduced in C++11. This new way is both efficient and user friendly. Here is an example:

.. code-block:: cpp

    using namespace clue;

    const char *str = "123, 456, 789, 2468";

    std::vector<long> values;
    foreach_token_of(str, ", ", [&](const char *p, size_t len){
        // directly convert the token to an integer,
        // without making a copy of the token
        values.push_back(std::strtol(p, nullptr, 10));

        // always continue to take in next token
        // if return false, the tokenizing process will stop
        return true;
    });


Formally, the function signature is given as below.

.. cpp:function:: void foreach_token_of(str, delimiters, f)

    Extract tokens from the string str, with given delimiter, and apply f to each token.

    :param str:  The input string, which can be either of the following type:

        - C-string (*e.g.* ``const char*``)
        - Standard string (*e.g.* ``std:string``)
        - String view (*e.g.* ``string_view``)

    :param delimiters: The delimiters for separating tokens, which can be either a character or a C-string (if a character ``c`` matches any char in the given ``delimiters``, then ``c`` is considered as a delimiter).

    :param f:  The call back function for processing tokens. Here, ``f`` should be a function, a lambda function, or a functor that takes in two inputs (the base address of the token and its length), and returns a boolean value that indicates whether to continue.

    This function stops when all tokens have been extracted and processed *or* when the callback function ``f`` returns ``false``.
