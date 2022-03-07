.. _stringview:

String View
============

In the `C++ Extensions for Library Fundamentals (N4480) <http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2015/n4480.html>`_, a class template ``basic_string_view`` is introduced. Each instance of such a class refers to a constant contiguous sequence of characters (or *char-like objects*). This class provides a light-weight representation (with only a pointer and a size) of a *sub-string* that implements many of the methods available for ``std::string``.

The string views are very useful in practice, especially for those applications that heavily rely on sub-string operations (but don't need to modify the string content). For such applications, string views can be a drop-in replacement of standard strings (*i.e.* instances of ``std::string``) as they provide a similar set of interface, but are generally much more efficient (they don't make copies).

This library provides string view classes, where our implementation strictly follows the `Technical Specification (N4480) <http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2015/n4480.html>`_, except that all the classes and functions are within the namespace ``clue`` (instead of ``std::experimental``). The standard document for this class is available `here <http://en.cppreference.com/w/cpp/experimental/basic_string_view>`_.

Below is brief description of the types, their members, and other relevant functions.


The ``basic_string_view`` class template
-----------------------------------------

The signature of the class template is as follows:

.. cpp:class:: basic_string_view<charT, Traits>

    :param charT: The character type.
    :param Traits: The traits class that specify basic operations on the character type.
                   Here, ``Traits`` can be omitted, which is, by default, set to ``std::char_traits<charT>``.

Four typedefs are defined:

.. cpp:type:: basic_string_view<char> string_view
.. cpp:type:: basic_string_view<wchar_t> wstring_view
.. cpp:type:: basic_string_view<char16_t> u16string_view
.. cpp:type:: basic_string_view<char32_t> u32string_view

For ASCII strings with character type ``char``, one should use ``string_view``.


Member types and constants
---------------------------

The class ``basic_string_view<charT, Traits>`` contains a series of member typedefs as follows:

============================= ============================================
 **types**                     **definitions**
----------------------------- --------------------------------------------
``traits_type``                ``Traits``
``value_type``                 ``charT``
``pointer``                    ``const charT*``
``const_pointer``              ``const charT*``
``reference``                  ``const charT&``
``const_reference``            ``const charT&``
``iterator``                   implementing ``RandomAccessIterator``
``const_iterator``             ``iterator``
``reverse_iterator``           ``std::reverse_iterator<iterator>``
``const_reverse_iterator``     ``std::reverse_iterator<const_iterator>``
``size_type``                  ``std::size_t``
``difference_type``            ``std::difference_type``
============================= ============================================

It also has a member constant ``npos``, defined as ``size_t(-1)``, to indicate a certain kind of characters or sub-strings are not found in a finding process. (This is the same as ``std::string``).


Constructors
-------------

The class ``basic_string_view<charT, Traits>`` provides multiple ways to construct a string view.
Below is a brief documentation of the member functions. For conciseness, we take ``string_view`` for example in the following documentation. The same set of constructors and member functions apply to other instantations of the class template similarly.

.. cpp:function:: constexpr string_view() noexcept

    Construct an empty string view

.. cpp:function:: constexpr string_view(const string_view& r) noexcept

    Copy construct a string view from ``r`` (default behavior)

    :note: The copy constructor only sets the size and the base pointer, without copying
           the characters that it refers to.

.. cpp:function:: string_view(const std::string& s) noexcept

    Construct a view of a standard string ``s``.

.. cpp:function:: constexpr string_view(const charT* s, size_type count) noexcept

    Construct a view with the base address `s` and length `count`.

.. cpp:function:: constexpr string_view(const charT* s) noexcept

    Construct a view of a null-terminated C-string.

The ``string_view`` class also has destructor and assignment operators, with default behaviors.


Basic Properties
-----------------

The ``string_view`` class provides member functions to get basic properties:

.. cpp:function:: constexpr bool empty() const noexcept

    Get whether the string view is empty (*i.e.* with zero length).

.. cpp:function:: constexpr size_type length() const noexcept

    Get the length (*i.e.* the number of characters).

.. cpp:function:: constexpr size_type size() const noexcept

    Get the length (the same as ``length()``).

.. cpp:function:: constexpr size_type max_size() const noexcept

    Get the maximum number of characters that a string view can possibly refer to.


Element Access
---------------

.. cpp:function:: constexpr const_reference operator[](size_type pos) const

    Get a const reference to the character at location ``pos``.

    :note: The member function ``operator []`` does not perform bound checking.

.. cpp:function:: const_reference at(size_type pos) const

    Get a const reference to the character at location ``pos``, with bounds checking.

    :throw: an exception of class ``std:out_of_range`` if ``pos >= size()``.

.. cpp:function:: constexpr const_reference front() const

    Get a const reference to the first character in the view.

.. cpp:function:: constexpr const_reference back() const

    Get a const reference to the last character in the view.

.. cpp:function:: constexpr const_pointer data() const noexcept

    Get a const pointer to the base address (*i.e.* to the first character).

    :note: For views constructed with default constructor, this returns a null pointer.


Iterators
----------

.. cpp:function:: constexpr const_iterator cbegin() const noexcept

    Get a const iterator to the beginning.

.. cpp:function:: constexpr const_iterator cend() const noexcept

    Get a const iterator to the end.

.. cpp:function:: constexpr iterator begin() const noexcept

    Get a const iterator to the beginning, equivalent to ``cbegin()``.

.. cpp:function:: constexpr iterator end() const noexcept

    Get a const iterator to the end, equivalent to ``cend()``.

.. cpp:function:: constexpr const_iterator crbegin() const noexcept

    Get a const reverse iterator to the reversed beginning.

.. cpp:function:: constexpr const_iterator crend() const noexcept

    Get a const reverse iterator to the reversed end.

.. cpp:function:: constexpr iterator rbegin() const noexcept

    Get a const reverse iterator to the reversed beginning, equivalent to ``crbegin()``.

.. cpp:function:: constexpr iterator rend() const noexcept

    Get a const reverse iterator to the reversed end, equivalent to ``crend()``.


Modifiers
----------

.. cpp:function:: void clear() noexcept

    Clear the view, resetting the data pointer and the size to ``nullptr`` and ``0`` respectively.

.. cpp:function:: void remove_prefix(size_type n) noexcept

    Exclude the first ``n`` characters from the view.

.. cpp:function:: void remove_suffix(size_type n) noexcept

    Exclude the last ``n`` characters from the view.

.. cpp:function:: void swap(string_view& other) noexcept

    Swap the view with ``other``.

.. note::

    An external ``swap`` function are provided for string views, which invokes the member function ``basic_string_view::swap`` to perform the swapping.


Conversion, Copy, and Sub-string
---------------------------------

.. cpp:function:: explicit operator std::string() const

    Convert the string view to a standard string (by making a copy).

.. cpp:function:: std::string to_string() const

    Convert the string view to a standard string (by making a copy).

.. cpp:function:: size_type copy(charT* s, size_type n, size_type pos = 0) const

    Copy the part starting at ``pos`` to a buffer ``s`` of length ``n``.

    :return: The number of characters actually copied, which is equal to ``min(n, size() - pos)``.

.. cpp:function:: constexpr string_view substr(size_type pos = 0, size_type n = npos) const

    Get a view of a sub-string (with length bounded by ``n``) that begins at ``pos``.

    :return: With ``pos < size()``, it returns a view of a sub-string, whose length is equal to ``min(n, size() - pos)``.

    :throw: an exception of class ``std::out_of_range`` if ``pos >= size()``.

Comparison
-----------

.. cpp:function:: int compare(string_view sv) const noexcept

    Compare with another string view ``sv``.

    :return: ``0`` when it is equal to ``sv``, a negative integer when it is less than ``sv`` (in lexicographical order), or a positive integer when it is greater than ``sv``.

.. cpp:function:: int compare(size_type pos1, size_type n1, string_view sv) const

    Equivalent to ``substr(pos1, n1).compare(sv)``.

.. cpp:function:: int compare(size_type pos1, size_type n1, string_view sv, size_type pos2, size_type n2) const

    Equivalent to ``substr(pos1, n1).compare(sv.substr(pos2, n2))``.

.. cpp:function:: int compare(const charT* s) const

    Compare with a null-terminated C-string ``s``.

.. cpp:function:: int compare(size_type pos1, size_type n1, const charT* s) const

    Equivalent to ``substr(pos1, n1).compare(s)``.

.. cpp:function:: int compare(size_type pos1, size_type n1, const charT* s, size_type n2) const

    Equivalent to ``substr(pos1, n1).compare(string_view(s, n2))``.

.. note::

    These many ``compare`` methods may seem redundant. They are there mainly to be consistent with the interface of ``std::string``.

    In addition to the ``compare`` methods, all comparison operators (including ``==, !=, <, >, <=, >=``) are provided for comparing string views. These operators return values of type ``bool``.


Find Characters
----------------

Similar to ``std::string``, string view classes provide a series of member functions to locate characters or sub-strings. These member functions return the index of the found occurrence or ``string_view::npos`` when the specified character or sub-string is not found within the view (or part of the view).


.. cpp:function:: size_type find(charT c, size_type pos = 0) const noexcept

    Find the first occurrence of a character ``c``, starting from ``pos``.

.. cpp:function:: size_type rfind(charT c, size_type pos = npos) const noexcept

    Find the last occurrence of a character ``c``, in a reverse order, starting from ``pos``, or the end of the string view, if ``pos >= size()``.

.. cpp:function:: size_type find_first_of(charT c, size_type pos = 0) const noexcept

    Find the first occurrence of a character ``c``, starting from ``pos`` (same as ``find(c, pos)``).

.. cpp:function:: size_type find_first_of(string_view s, size_type pos = 0) const noexcept

    Find the first occurrence of a character that is in ``s``, starting from ``pos``.

.. cpp:function:: size_type find_first_of(const charT* s, size_type pos, size_type n) const noexcept

    Equivalent to ``find_first_of(string_view(s, n), pos)``.

.. cpp:function:: size_type find_first_of(const charT* s, size_type pos = 0) const noexcept

    Equivalent to ``find_first_of(string_view(s), pos)``.

.. cpp:function:: size_type find_last_of(charT c, size_type pos = npos) const noexcept

    Find the last occurrence of a character ``c``, in a reverse order, starting from ``pos``, or the end of the string view, if ``pos >= size()`` (same as ``rfind(c, pos)``).

.. cpp:function:: size_type find_last_of(string_view s, size_type pos = npos) const noexcept

    Find the last occurrence of a character that is in ``s``, in a reverse order, starting from ``pos`` (or the end of the string view, if ``pos >= size()``).

.. cpp:function:: size_type find_last_of(const charT* s, size_type pos, size_type n) const noexcept

    Equivalent to ``find_last_of(string_view(s, n), pos)``.

.. cpp:function:: size_type find_last_of(const charT* s, size_type pos = npos) const noexcept

    Equivalent to ``find_last_of(string_view(s), pos)``.

.. cpp:function:: size_type find_first_not_of(charT c, size_type pos = 0) const noexcept

    Find the first occurrence of a character that is not ``c``, starting from ``pos``.

.. cpp:function:: size_type find_first_not_of(string_view s, size_type pos = 0) const noexcept

    Find the first occurrence of a character that is not in ``s``, starting from ``pos``.

.. cpp:function:: size_type find_first_not_of(const charT* s, size_type pos, size_type n) const noexcept

    Equivalent to ``find_first_not_of(string_view(s, n), pos)``.

.. cpp:function:: size_type find_first_not_of(const charT* s, size_type pos = 0) const noexcept

    Equivalent to ``find_first_not_of(string_view(s), pos)``.

.. cpp:function:: size_type find_last_not_of(charT c, size_type pos = npos) const noexcept

    Find the last occurrence of a character that is not ``c``, in a reverse order, starting from ``pos``.

.. cpp:function:: size_type find_last_not_of(string_view s, size_type pos = npos) const noexcept

    Find the first occurrence of a character that is not in ``s``, in a reverse order, starting from ``pos``.

.. cpp:function:: size_type find_last_not_of(const charT* s, size_type pos, size_type n) const noexcept

    Equivalent to ``find_first_not_of(string_view(s, n), pos)``.

.. cpp:function:: size_type find_last_not_of(const charT* s, size_type pos = npos) const noexcept

    Equivalent to ``find_first_not_of(string_view(s), pos)``.


Find Substrings
----------------

.. cpp:function:: size_type find(string_view s, size_type pos = 0) const noexcept

    Find a substring ``s``, starting from ``pos``.

.. cpp:function:: size_type find(const charT* s, size_type pos, size_type n) const noexcept

    Equivalent to ``find(substr(s, n), pos)``.

.. cpp:function:: size_type find(const charT* s, size_type pos = 0) const noexcept

    Equivalent to ``find(substr(s), pos)``.

.. cpp:function:: size_type rfind(string_view s, size_type pos = npos) const noexcept

    Find a substring ``s``, in a reverse order, starting from ``pos``, or the end of the string view if ``pos >= size()``.

    :note: A matched substring is considered as *found* if its starting position precedes ``pos``.

.. cpp:function:: size_type rfind(const charT* s, size_type pos, size_type n) const noexcept

    Equivalent to ``rfind(substr(s, n), pos)``.

.. cpp:function:: size_type rfind(const charT* s, size_type pos = npos) const noexcept

    Equivalent to ``rfind(substr(s), pos)``.


.. note::

    The reason that there are so many ``find_*`` methods in slightly different forms is that string views need to be consistent with ``std::string`` in the interface, so it can serve as a drop-in replacement.
