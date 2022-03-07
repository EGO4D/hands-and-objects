Formatting
===========

*CLUE* provides several convenience functions to facilitate string formatting. These functions are light-weight wrappers based on ``snprintf`` and C++'s ``stringstream``.
These functions are provided by the header ``<clue/sformat.hpp>``.

.. cpp:function::  sstr(args...)

    Concatenating multiple arguments into a string, through a string stream (an object of class ``std::ostringstream``).

    .. note::

        The arguments here need not be strings. The only requirement is that they can be inserted to a standard output stream.

    **Examples:**

    .. code-block:: cpp

        #include <clue/sformat.hpp>

        sstr(12);  // -> "12"
        sstr(1, 2, 3); // -> "123"
        sstr(1, " + ", 2, " = ", 3); // -> "1 + 2 = 3"

        struct MyPair {
            int x;
            int y;
        };

        inline std::ostream& operator << (std::ostream& out, const MyPair& p) {
            out << '(' << p.x << ", " << p.y << ')';
            return out;
        }

        sstr("a = ", MyPair{1,2}); // -> "a = (1, 2)"


.. cpp:function:: cfmt(fmt, x)

    Wraps a numeric value ``x`` into a light-weight wrapper of class ``cfmt_t<T>``. This wrapper uses ``snprintf``-formatting
    with pattern ``fmt``, when inserted to a standard output stream.

    **Examples:**

    .. code-block:: cpp

        cout << cfmt("%d", 12);   // cout << "12"
        cout << cfmt("%.6f", 2);  // cout << "12.000000"


.. cpp:function:: cfmt_s(fmt, args...)

    Encapsulate the result of ``snprintf``-formatting into a function. This function accepts multiple arguments.
    It returns an object of class ``std::string``.

    **Examples:**

    .. code-block:: cpp

        cfmt_s("%04d", 12);  // -> "0012"
        cfmt_s("%d + %d = %d", 1, 2, 3); // -> "1 + 2 = 3"


.. cpp:function:: delimits(seq, delimiter)

    Wraps a sequence ``seq`` into a light-weight wrapper of class ``Delimits<Seq>``. The elements of the sequence will be outputed
    with a separator ``delimiter``, when the wrapper is inserted to a standard output stream.

    .. note::

        Here, ``seq`` can be of arbitrary collection type ``Seq``. The only requirement is that ``Seq`` provides the ``begin()``
        and ``end()`` methods.

    **Examples:**

    .. code-block:: cpp

        std::vector xs{1, 2, 3};
        cout << delimits(xs, "+");  // cout << "1+2+3"

        std::vector ys{5};
        cout << delimits(ys, ",");  // cout << "5"

        sstr('[', delimits(xs, ", "), ']');  // -> "[1, 2, 3]"
