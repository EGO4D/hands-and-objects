Meta-types and Meta-functions
===============================

Template meta-programming has become an indispensible part of modern C++. In C++11, new features such as *Variadic template* and *Template alias* makes meta-programming much more efficient and convenient than before. *CLUE++* provides a set of tools to facilitate meta programming, which take full advantage of these new C++ features.

For those who are not familiar with C++ meta-programming, Andrzej has a great `blog <https://akrzemi1.wordpress.com/2012/03/19/meta-functions-in-c11/>`_ that provides an excellent introduction of this topic.

**Important Note:** all meta-programming facilities in *CLUE++* are within the namespace ``clue::meta``.

Basic types
-------------

A set of types to support meta-programming:

.. code-block:: cpp

    // Note: all names below are within the namespace clue::meta

    using std::integral_constant;

    // Indicator of a C++ type
    template<typename T>
    struct type_ {
        using type = T;
    };

    // Extract an encapsulated type
    template<typename A>
    using get_type = typename A::type;

    // Indicator of a nil type (nothing)
    struct nil_{};

    // Static boolean value
    template<bool V>
    using bool_ = integral_constant<bool, V>;

    using true_  = bool_<true>;
    using false_ = bool_<false>;

    // Static integral values
    template<char  V> using char_  = integral_constant<char,  V>;
    template<int   V> using int_   = integral_constant<int,   V>;
    template<long  V> using long_  = integral_constant<long,  V>;
    template<short V> using short_ = integral_constant<short, V>;

    template<unsigned char  V> using uchar_  = integral_constant<unsigned char,  V>;
    template<unsigned int   V> using uint_   = integral_constant<unsigned int,   V>;
    template<unsigned long  V> using ulong_  = integral_constant<unsigned long,  V>;
    template<unsigned short V> using ushort_ = integral_constant<unsigned short, V>;

    // Static size value
    template<size_t V> using size_ = integral_constant<size_t, V>;

    // Extract the value type of a static value.
    template<class A>
    using value_type_of = typename A::value_type;


Sometimes, it is useful to combine two types. For this purpose, we provide a ``pair_`` type to express a pair of types, as well as meta-functions ``first`` and ``second`` to retrieve them.

.. code-block:: cpp

    // Note: all names below are within the namespace clue::meta

    template<typename T1, typename T2>
    struct pair_ {
        using first_type = T1;
        using second_type = T2;
    };

    template<typename T1, typename T2>
    struct first<pair_<T1, T2>> {
        using type = T1;
    };

    template<typename T1, typename T2>
    struct second<pair_<T1, T2>> {
        using type = T2;
    };

    template<class A> using first_t  = typename first<A>::type;
    template<class A> using second_t = typename second<A>::type;

.. note::

    The meta-functions ``first`` and ``second`` are also specialized for other meta data structures, such as the *meta sequence*.


Static Index Sequence
-----------------------

The library provides useful facilities to construct static index sequence, which is useful for splatting elements of a tuples as arguments.

.. code-block:: cpp

    // index_seq can be used to represent a static sequence of indexes
    template<size_t... Inds>
    struct index_seq{};

    // make_index_seq<N> constructs index_seq<0, ..., N-1>

    make_index_seq<0>;  // -> index_seq<>
    make_index_seq<1>;  // -> index_seq<1>
    make_index_seq<4>;  // -> index_seq<0, 1, 2, 3>

The following example shows how one can leverage ``make_index_seq`` to splat tuple arguments.

.. code-block:: cpp

    // suppose you have a function join can accepts arbitrary number of arguments
    template<class... Args>
    void join(const Args&... args) { /* ... */ }

    // the join_tup function can splat elements of a tuple

    template<class... Args, size_t... I>
    void join_tup_impl(const std::tuple<Args...>& tup, clue::meta::index_seq<I...>) {
        join(std::get<I>(tup)...);
    }

    template<class... Args>
    void join_tup(const std::tuple<Args...>& tup) {
        join_tup_impl(tup, clue::meta::make_index_seq<sizeof...(Args)>{});
    }

    join_tup(std::make_tuple("abc", "xyz", 123));


Basic functions
----------------

The library also has a series of meta-functions to work with types or static values.

Arithmetic functions
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: cpp

    // Note: all names below are within the namespace clue::meta

    template<typename A>
    using negate = integral_constant<value_type_of<A>, -A::value>;

    template<typename A>
    using next = integral_constant<value_type_of<A>, A::value+1>;

    template<typename A>
    using prev = integral_constant<value_type_of<A>, A::value-1>;

    template<typename A, typename B>
    using plus = integral_constant<value_type_of<A>, A::value + B::value>;

    template<typename A, typename B>
    using minus = integral_constant<value_type_of<A>, A::value - B::value>;

    template<typename A, typename B>
    using mul = integral_constant<value_type_of<A>, A::value * B::value>;

    template<typename A, typename B>
    using div = integral_constant<value_type_of<A>, A::value / B::value>;

    template<typename A, typename B>
    using mod = integral_constant<value_type_of<A>, A::value % B::value>;

    // aliases, to cover the names in <functional>
    template<typename A, typename B> using multiplies = mul<A, B>;
    template<typename A, typename B> using divides = div<A, B>;
    template<typename A, typename B> using modulo = mod<A, B>;

Comparison functions
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: cpp

    // Note: all names below are within the namespace clue::meta

    template<typename A, typename B> using eq = bool_<(A::value == B::value)>;
    template<typename A, typename B> using ne = bool_<(A::value != B::value)>;
    template<typename A, typename B> using gt = bool_<(A::value >  B::value)>;
    template<typename A, typename B> using ge = bool_<(A::value >= B::value)>;
    template<typename A, typename B> using lt = bool_<(A::value <  B::value)>;
    template<typename A, typename B> using le = bool_<(A::value <= B::value)>;

    // aliases, to cover the names in <functional>
    template<typename A, typename B> using equal_to      = eq<A, B>;
    template<typename A, typename B> using not_equal_to  = ne<A, B>;
    template<typename A, typename B> using greater       = gt<A, B>;
    template<typename A, typename B> using greater_equal = ge<A, B>;
    template<typename A, typename B> using less          = lt<A, B>;
    template<typename A, typename B> using less_equal    = le<A, B>;

Logical functions
~~~~~~~~~~~~~~~~~~

.. cpp:class:: not_<A>

    The member constant ``not_<A>::value`` is equal to ``!A::value``.

.. cpp:class:: and_<A, B>

    The member constant ``and_<A, B>::value`` is ``true`` iff both ``A::value`` and ``B::value`` is true.

.. cpp:class:: or_<A, B>

    The member constant ``or_<A, B>::value`` is ``true`` iff either ``A::value`` or ``B::value`` is true.

.. note::

    The meta-functions ``and_<A, B>`` and ``or_<A, B>`` implement the *short-circuit behavior*. In particular, when ``A::value == false``, ``and_<A, B>::value`` is set to ``false``  without examining the internals of ``B``.
    Likewise, when ``A::value == true``, ``or_<A, B>::value`` is set to ``true`` without examining the internals of ``B``.

Select
-------

C++11 provides ``std::conditional`` for static dispatch based on a condition. However, using this type in practice, especially in the cases with multiple branches, is very cumbersome. Below is an example that uses ``std::conditional`` to map a numeric value to a signed value type.

.. code-block:: cpp

    #include <type_traits>

    template<typename T>
    using signed_type =
        typename std::conditional<
            std::is_integral<T>::value,
            typename std::conditional<std::is_unsigned<T>::value,
                typename std::make_signed<T>::type,
                T
            >::type,
            typename std::conditional<std::is_floating_point<T>::value,
                T,
                nil_t
            >::type
        >::type;

With the meta-function ``select`` and the helper alias ``select_t``, this can be expressed in a much more elegant and concise way:

.. code-block:: cpp

    #include <clue/meta.hpp>

    using namespace clue;

    template<typename T>
    using signed_type =
        meta::select_t<
            std::is_unsigned<T>,       std::make_signed<T>,
            std::is_signed<T>,         meta::type_<T>,
            std::is_floating_point<T>, meta::type_<T>,
            meta::type_<nil_t> >;

Specifically, ``meta::select`` is a variadic class template, described as follows:

- ``select<C1, A1, R>`` has a member typedef ``type`` which is equal to ``A1::type`` when ``C1::value`` is true, or ``R::type`` otherwise. This meta-function can accept arbitrary odd number of arguments.
- Generally, ``select<C1, A1, C2, A2, ..., Cm, Am, R>`` has a member typedef ``type`` which is equal to ``A1::type`` when ``C1::value`` is true, otherwise, it is equal to ``A2::type`` if ``C2::value`` is true, and so on. If no conditions are met, it is set to ``R::type``.

A helper alias ``select_t`` is provided to further simplify the use:

.. code-block:: cpp

    template<typename... Args>
    using select_t = typename select<Args...>::type;

.. note::

    The meta-function ``select`` implements a *short-circuit behavior*. It examines the conditions sequentially, and once it finds a condition that is ``true``, it extracts the next type, and will not continue to examine following conditions.

Variadic Reduction
-------------------

A set of variadic meta-functions are provided to perform reduction over static values.

.. cpp:class:: meta::sum<Args...>

    With a member constant ``value`` that equals the sum of argument's member values.

.. cpp:class:: meta::prod<Args...>

    With a member constant ``value`` that equals the product of argument's member values.

.. cpp:class:: meta::maximum<Args...>

    With a member constant ``value`` that equals the maximum of argument's member values.

.. cpp:class:: meta::minimum<Args...>

    With a member constant ``value`` that equals the minimum of argument's member values.

.. cpp:class:: meta::all<Args...>

    With a member constant ``value``, which equals ``true`` if all argument's member values are ``true``, or ``false`` otherwise.

    :note: ``all<>::value == true``.

.. cpp:class:: meta::any<Args...>

    With a member constant ``value``, which equals ``true`` if any of the argument's member value is ``true``, or ``false`` otherwise.

    :note: ``any<>::value == false``.

.. cpp:class:: meta::count_true<Args...>

    With a member constant ``value``, which equals the number of arguments whose member value is ``true``.

.. cpp:class:: meta::count_false<Args...>

    With a member constant ``value``, which equals the number of arguments whose member value is ``false``.

.. cpp:class:: meta::all_same<Args...>

    With a member constant ``value``, which indicates whether all argument types are the same.

.. note::

    The meta-functions ``all`` and ``any`` both implement the *short-circuit behaviors*. They won't look further once the resultant value can be determined.
