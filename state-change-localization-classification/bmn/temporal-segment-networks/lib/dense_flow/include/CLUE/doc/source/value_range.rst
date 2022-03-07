Value Range
============

It is a very common pattern in C/C++ programming to write loops that enumerate values within a certain range, such as

.. code-block:: cpp

    for (int i = 0; i < n; ++i) {
        // do something
    }

In C++11, the range for-loop syntax is introduced, which allow concise expression of the looping over elements in a container. However, one has to resort to the old pattern when enumerating values. Here, we provide a class template ``value_range`` that wraps a range of values to a light-weight *container-like* object. Below is an example:

.. code-block:: cpp

    #include <clue/value_range.hpp>

    using namespace clue;

    size_t n = 10

    // enumerate i from 0 to n-1
    for (size_t i: vrange(n)) {
        // do something on i
    }

    double a = 2.0, b = 9.0;
    // enumerate v from 2.0 to 8.0
    for (auto v: vrange(a, b)) {
        // do something on i
    }

    std::vector<int> a{1, 2, 3, 4, 5};
    std::vector<int> b{5, 6, 7, 8, 9};
    std::vector<int> r

    // enumrate i from 0 to a.size() - 1
    for (auto i: indices(a)) {
        r.push_back(a[i] + b[i]);
    }

Documentation of ``value_range`` and relevant functions are given below.

The ``value_range`` and ``stepped_value_range`` class templates
---------------------------------------------------------------

Formally, the class template ``value_range`` is defined as:

.. cpp:class:: value_range<T, D, Traits>

    Classes to represent stepped ranges, such as ``1, 2, 3, 4, ...``.

    :param T: The value type.
    :param D: The difference type. This can be omitted, and it will be, by default, set to ``default_difference<T>::type``.
    :param Traits: A traits class that specifies the behavior of the value type ```T``. This class has to satisfy the *EnumerableValueTraits* concept, which will be explained in the section enumerable_value_traits_. In general, one may omit this, and it will be, by default, set to ``value_type_traits<T, D>``.

.. cpp:class:: default_difference<T>

    It provides a member typedef that indicates the *default* difference type for ``T``.

    In particular, if ``T`` is an unsigned integer type, ``default_difference<T>::type`` is ``std::make_signed<T>::type``. In other cases, ``default_difference<T>::type`` is identical to ``T``.

    To enumerate non-numerical types (*e.g.* dates), one should specialize ``default_difference<T>`` to provide a suitable difference type.

.. cpp:class:: stepped_value_range<T, S, D, Traits>

    Classes to represent stepped ranges, such as ``1, 3, 5, 7, ...``.

    :param T: The value type.
    :param S: The step type.
    :param D: The difference type. By default, it is ``default_difference_type<T>::type``.
    :param Traits: The trait class for ``T``. By default, it is ``value_type_traits<T, D>``.

.. note::

    For ``stepped_value_range<T, S>``, only unsigned integral types for ``T`` and ``S`` are supported at this point.


Member types
-------------

The class ``value_range<T>`` or ``stepped_value_range<T, S>`` contains a series of member typedefs as follows:

============================= ============================================
 **types**                     **definitions**
----------------------------- --------------------------------------------
``value_type``                 ``T``
``difference_type``            ``D``
``step_type``                  ``S``
``traits_type``                ``Traits``
``size_type``                  ``std::size_t``
``pointer``                    ``const T*``
``const_pointer``              ``const T*``
``reference``                  ``const T&``
``const_reference``            ``const T&``
``iterator``                   implementing ``RandomAccessIterator``
``const_iterator``             ``iterator``
============================= ============================================

.. note::

    For ``value_range<T>``, the ``step_type`` is the same as ``size_type``.


Construction
-------------

The ``value_range<T>`` and ``stepped_value_range<T, S>`` classes have simple constructors.

.. cpp:function:: constexpr value_range(const T& vbegin, const T& vend)

    :param vbegin: The beginning value (inclusive).
    :param vend:   The ending value (exclusive).

    For example, ``value_range(0, 3)`` indicates the following sequence ``0, 1, 2``.

.. cpp:function:: stepped_value_range(const T& vbegin, const T& vend, const S& step)

    :param vbegin: The beginning value (inclusive).
    :param vend:   The ending value (exclusive).
    :param step:   The incremental step.

    For example, ``stepped_value_range(0, 2, 5)`` indicates the following sequence ``0, 2, 4``.

.. note::

    These classes also have a copy constructor, an assignment operator, a destructor and a ``swap`` member function, all with default behaviors.

.. note::

    For stepped ranges, the **step must be positive**. Zero or negative step would result in undefined behavior. The size of a stepped range is computed as
    ``(e - b + (s - 1)) / s``.


In addition, convenient constructing functions are provided, with which the user does not need to explictly specify the value type (which would be infered from the arguments):

.. cpp:function:: constexpr value_range<T> vrange(const T& u)

    Equivalent to ``value_range<T>(static_cast<T>(0), u)``.

.. cpp:function:: constexpr value_range<T> vrange(const T& a, const T& b)

    Equivalent to ``value_range<T>(a, b)``.

.. cpp:function:: value_range<Siz> indices(const Container& c)

    Returns a value range that contains indices from ``0`` to ``c.size() - 1``. Here, the value type ``Siz`` is ``Container::size_type``.


Properties and element access
-------------------------------

The ``value_range<T>`` and ``stepped_value_range<T, S>`` classes provide a similar set of member functions that allow access of the basic properties and individual values in the range, as follows.

.. cpp:function:: constexpr size_type size() const noexcept

    Get the size of the range, *i.e.* the number of values contained in the range.

.. cpp:function:: constexpr bool empty() const noexcept

    Get whether the range is empty, *i.e.* contains no values.

.. cpp:function:: constexpr size_type step() const noexcept

    Get the step size.

    :note: For ``value_range<T>``, the step size is always ``1``.

.. cpp:function:: constexpr T front() const noexcept

    Get the first value within the range.

.. cpp:function:: constexpr T back() const noexcept

    Get the last value **within** the range.

.. cpp:function:: constexpr T begin_value() const noexcept

    Get the first value in the range (equivalent to ``front()``).

.. cpp:function:: constexpr T end_value() const noexcept

    Get the value that specifies the end of the value, which is the value next to ``back()``.

.. cpp:function:: constexpr T operator[](size_type pos) const

    Get the value at position ``pos``, withou bounds checking.

.. cpp:function:: constexpr T at(size_type pos) const

    Get the value at position ``pos``, with bounds checking.

    :throw: an exception of class ``std::out_of_range`` if ``pos >= size()``.


Iterators
----------

.. cpp:function:: constexpr const_iterator cbegin() const

    Get a const iterator to the beginning.

.. cpp:function:: constexpr const_iterator cend() const

    Get a const iterator to the end.

.. cpp:function:: constexpr iterator begin() const

    Get a const iterator to the beginning, equivalent to ``cbegin()``.

.. cpp:function:: constexpr iterator end() const

    Get a const iterator to the end, equivalent to ``cend()``.

.. note::

    A value range or stepped value range does not actually store the values in the range. Hence, the iterators are *proxies* that do not refer to an existing location in memory. Instead, ``*iter`` returns the value itself instead of a reference. In spite of this subtle difference from a typical iterator, we find that it works perfectly with most STL algorithms.


.. _enumerable_value_traits:

The *EnumerableValueTraits* concept
------------------------------------

The class template ``value_range`` has a type parameter ``Traits``, which has to satisfy the following concept.

.. code-block:: cpp

    // x, y are values of type T, and n is a value of type D

    Traits::increment(x);       // in-place increment of x
    Traits::decrement(x);       // in-place decrement of x
    Traits::increment(x, n);    // in-place increment of x by n units
    Traits::decrement(x, n);    // in-place decrement of x by n units

    Traits::next(x);        // return the value next to x
    Traits::prev(x);        // return the value that precedes x
    Traits::next(x, n);     // return the value ahead of x by n units
    Traits::prev(x, n);     // return the value behind x by n units

    Traits::eq(x, y);       // whether x is equal to y
    Traits::lt(x, y);       // whether x is less than y
    Traits::le(x, y);       // whether x is less than or equal to y

    Traits::difference(x, y); // the difference between x and y, i.e. x - y

By default, the builtin ``value_range_traits<T, D>`` would be used and users don't have to specify the traits explicitly. However, one can specify a different trait class to provide special behaviors.
