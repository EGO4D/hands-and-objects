Optional (Nullable)
====================

In the `C++ Extensions for Library Fundamentals (N4480) <http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2015/n4480.html>`_, a class template ``optional`` is introduced, which represents objects that may possibly contain a value. Such types are widely provided by modern programming languages (*e.g.* ``Nullable`` in *C#*, ``Maybe`` in *Haskell*, ``Optional`` in *Swift*, and ``Option`` in *Rust*), and have shown their important utility in practice. This library *"backports"* the ``optional`` type to C++11 (within the namespace ``clue``).

Here is a simple example that illustrates the use of the ``optional`` class.

.. code-block:: cpp

    #include <cmath>
    #include <clue/optional.hpp>

    using namespace clue;

    inline optional<double> safe_sqrt(double x) {
        return x >= 0.0 ?
            make_optional(std::sqrt(x)) :
            optional<double>();
    }

    auto u = safe_sqrt(-1.0);  // -> optional<double>()
    (bool)u;          // -> false
    u.value();        // throws an exception
    u.value_or(0.0);  // -> 0.0

    auto v = safe_sqrt(4.0);  // -> optional<double>(2.0)
    (bool)v;          // -> true
    v.value();        // -> 2.0
    v.value_or(0.0);  // -> 2.0


The standard documentation of the ``optional`` type is available `here <http://en.cppreference.com/w/cpp/experimental/optional>`_. Below is a brief description of this type.


Types
------

The class template ``optional`` is declared as:

.. cpp:class:: optional<T>

    :param T: The type of the (possibly) contained value.

.. cpp:class:: optional<T>::value_type

    The ``optional<T>`` has a member typedef ``value_type`` defined as ``T``.

In addition, several helper types are provided:

.. cpp:class:: in_place_t

    A tag type to indicate in-place construction of an optional object. It has a predefined instance ``in_place``.

.. cpp:class:: nullopt_t

    A tag type to indicate an optional object with uninitialized state. It is a predefined instance ``nullopt``.


Constructors
-------------

An optional object can be constructd in different ways:

.. cpp:function:: constexpr optional()

    Constructs an *empty* optional object, which does not contain a value.

.. cpp:function:: constexpr optional(nullopt_t)

    Constructs an *empty* optional object (equivalent to ``optional<T>()``).

.. cpp:function:: optional(const optional&)

    Copy constructor, with default behavior.

.. cpp:function:: optional(optional&&)

    Move constructor, with default behavior.

.. cpp:function:: constexpr optional(const value_type& v)

    Construct an optional object that contains (a copy of) the input value ``v``.

.. cpp:function:: constexpr optional(value_type&& v)

    Construct an optional object that contains the input value ``v`` (moved in).

.. cpp:function:: constexpr optional(in_place_t, Args&&... args)

    Construct an optional object, with the contained value constructed inplace with the initializing arguments ``args``.


Modifiers
----------

After an ``optional`` object is constructed, its value can be re-constructed later using ``swap``, ``emplace``, or the assignment operator.

.. cpp:function:: void swap(optional& other)

    Swap with another optional object ``other``.

.. cpp:function:: void emplace(Args&&... args)

    Re-construct the contained value using the provided arguments ``args``.


Observers
----------

.. note::

    This class provides
    ``operator->`` to allow the access of the contained vlaue in a pointer form, and
    ``operator*`` to allow the access in a dereferenced form. One must use these operators when the ``optional`` object actually contains a value, otherwise it is *undefined behavior*.

    A safer (but slightly less efficient) way to access the contained value is to use ``value`` or ``value_or`` member functions described below.

.. cpp:function:: constexpr explicit operator bool() const noexcept

    Convert the object to a boolean value.

    :return: ``true`` when the object contains a value, or ``false`` otherwise.

.. cpp:function:: constexpr value_type const& value() const

    Get a const reference to the contained value.

    :throw: an exception of class ``bad_optional_access`` when the object is empty.

.. cpp:function:: value_type& value()

    Get a reference to the contained value.

    :throw: an exception of class ``bad_optional_access`` when the object is empty.

.. cpp:function:: constexpr value_type value_or(U&& v) const&

    Get the contained value, or a static convertion of ``v`` to the type ``T`` (when the object is empty).

.. cpp:function:: value_type value_or(U&& v) &&

    Get the contained value, or a static convertion of ``v`` to the type ``T`` (when the object is empty).


Non-member Functions
---------------------

.. cpp:function:: void swap(optional<T>& x, optional<T>& y)

    Swap two optional objects ``x`` and ``y``. Equivalent to ``x.swap(y)``.

.. cpp:function:: constexpr optional<R> make_optional(T&& v)

    Make an optional object that encapsulates a value ``v``.

    :return: An optional object of class ``optional<R>``, where the template parameter ``R`` is defined as ``typename std::decay<T>::type``.


Comparison
-----------

Comparison operators ``==, !=, <, >, <=, >=`` are provided to compare optional objects.

Two optional objects are considered as *equal* if they meet either of the following two conditions:

- they are both empty, or
- they both contain values, and the contained values are equal.

An optional object ``x`` are considered as *lesss than* another optional object ``y``, if either of the following conditions are met:

- ``x`` is empty while ``y`` is not.
- they both contain values, and ``x.value() < y.value()``.

.. note::

    Comparison between an optional object and a value ``v`` of type ``T`` is allowed. In such cases, ``v`` is treated as an optional object that contains a value ``v``, and then the rules above apply.
