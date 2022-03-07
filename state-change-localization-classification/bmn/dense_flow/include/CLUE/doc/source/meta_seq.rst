Meta-sequence: Sequence of Types
==================================

In meta-programming, it is sometimes useful to process a list of types, in a way that is similar to ``std::vector``. *CLUE++* provides facilities to support such operations in *compile-time*. Like other meta-programming tools, all these facilities are also in the namespace ``clue::meta``.

Meta-sequence
--------------

In *CLUE++*, we use a variadic class template ``meta::seq_``, defined below, to indicate a sequence of types.

.. code-block:: cpp

    // Within the namespace clue::meta:
    template<typename... Elems> struct seq_;

We provide a series of meta-functions that emulate the ``std::vector`` API to work with such a sequence of types.
Below is an example that illustrates the use of ``seq_`` and some of the meta-functions working with it.

.. code-block:: cpp

    using namespace clue::meta;
    using meta::seq_;
    using i1 = meta::int_<1>;
    using i2 = meta::int_<2>;
    using i3 = meta::int_<3>;
    using i4 = meta::int_<4>;

    // define a sequence of static values
    using s = seq_<i1, i2, i3>;

    constexpr size_t n = meta::size<s>::value // n = 3;

    using xf = meta::front_t<s>;  // xf is i1
    using xb = meta::back_t<s>;   // xb is i3
    using x0 = meta::at_t<s, 0>;  // x0 is i1
    using x1 = meta::at_t<s, 1>;  // x1 is i2

    using r1 = meta::push_back_t<s, i4>;
    // r1 is seq_<i1, i2, i3, i4>

    using r2 = meta::push_front_t<s, i4>;
    // r2 is seq_<i4, i1, i2, i3>

    using r3 = meta::pop_front_t<s>;
    // r3 is seq_<i2, i3>

    using r4 = meta::pop_back_t<s>;
    // r4 is seq_<i1, i2>

    using rr = meta::reverse<s>;
    // rr is seq_<i3, i2, i1>

    using rt = meta::transform<meta::next, s>;
    // rt is seq_<i2, i3, i4>

    constexpr int v1 = meta::sum<s>::value; // v1 = 6;
    constexpr int v2 = meta::max<s>::value; // v2 = 3;
    constexpr int v3 = meta::min<s>::value; // v3 = 1;


Basic properties
-----------------

.. cpp:class:: meta::size<seq_<Elems...>>

    With a member constant ``value`` that equals to the number of element types.

.. cpp:class:: meta::empty<seq_<Elems...>>

    With a member constant ``value`` that is ``true`` when the number of element types is zero, or ``false`` otherwise.


Element type access
---------------------

.. cpp:class:: meta::front<seq_<Elems...>>

    With a member typedef ``type`` corresponding to the first element type.

.. cpp:class:: meta::back<seq_<Elems...>>

    With a member typedef ``type`` corresponding to the last element type in the sequence.

.. cpp:class:: meta::at<seq_<Elems...>, N>

    With a member typedef ``type`` corresponding to the ```N``-th element type of the sequence.

.. cpp:class:: meta::first<seq_<Elems...>>

    With a member typedef ``type`` corresponding to the first element type. (Equivalent to using ``meta::front``).

.. cpp:class:: meta::second<seq_<Elems...>>

    With a member typedef ``type`` corresponding to the second element type.

Helper aliases are provided for all these meta functions:

.. code-block:: cpp

    // Within the namespace clue::meta:

    template<class Seq> using front_t  = typename front<Seq>::type;
    template<class Seq> using back_t   = typename back<Seq>::type;
    template<class Seq> using first_t  = typename first<Seq>::type;
    template<class Seq> using second_t = typename second<Seq>::type;

    template<class Seq, size_t N>
    using at_t = typename at<Seq, N>::type;

Modifiers
----------

.. cpp:class:: meta::clear<seq_<Elems...>>

    With a member typedef ``type = meta::seq_<>``.

.. cpp:class:: meta::pop_front<seq_<Elems...>>

    With a member typedef ``type`` which is a meta sequence with the first element type excluded.

.. cpp:class:: meta::pop_back<seq_<Elems...>>

    With a member typedef ``type`` which is a meta sequence with the last element type excluded.

.. cpp:class:: meta::push_front<seq_<Elems...>, X>

    With a member typedef ``type`` which prepends a type ``X`` to the front of the input meta sequence.

.. cpp:class:: meta::push_back<seq_<Elems...>, X>

    With a member typedef ``type`` which appends a type ``X`` to the back of the input meta sequence.

Helper aliases are provided for all these meta functions:

.. code-block:: cpp

    // Within the namespace clue::meta:

    template<class Seq> using clear_t = typename clear<Seq>::type;
    template<class Seq> using pop_front_t = typename pop_front<Seq>::type;
    template<class Seq> using pop_back_t  = typename pop_back<Seq>::type;

    template<class Seq, typename X>
    using push_front_t = typename push_front<Seq, X>::type;

    template<class Seq, typename X>
    using push_back_t = typename push_back<Seq, X>::type;


Sequence reduction
--------------------

All variadic reduction functions are specialized to perform reduction over a sequence, as

.. code-block:: cpp

    template<typename... Elems>
    struct sum<seq_<Elems...>> : public sum<Elems...> {};

    template<typename... Elems>
    struct prod<seq_<Elems...>> : public prod<Elems...> {};

    template<typename... Elems>
    struct max<seq_<Elems...>> : public max<Elems...> {};

    template<typename... Elems>
    struct min<seq_<Elems...>> : public min<Elems...> {};

    template<typename... Elems>
    struct all<seq_<Elems...>> : public all<Elems...> {};

    template<typename... Elems>
    struct any<seq_<Elems...>> : public any<Elems...> {};

    template<typename... Elems>
    struct count_true<seq_<Elems...>> : public count_true<Elems...> {};

    template<typename... Elems>
    struct count_false<seq_<Elems...>> : public count_false<Elems...> {};


Algorithms
-----------

We also implement a collection of algorithms to work with meta sequences.

.. cpp:class:: meta::cat<S1, S2>

    With a member typedef ``type`` that is a concatenation of two meta sequences ``S1`` and ``S2``.

.. cpp:class:: meta::zip<S1, S2>

    With a member typedef ``type`` that zips two meta sequences ``S1`` and ``S2`` of the same length.

**Example:**

.. code-block:: cpp

    using namespace clue;
    using S1 = meta::seq_<char, int>;
    using S2 = meta::seq_<float, double>;

    using R = typename zip<S1, S2>::type;
    // meta::seq_<
    //   meta::pair_<char, float>,
    //   meta::pair_<int,  double>
    // >

.. cpp:class:: meta::repeat<X, N>

    With a member typedef ``type`` which is a meta sequence that repeats the type ``X`` for ``N`` times.

    :example: ``meta::repeat<int, 3>::type`` is ``meta::seq_<int, int, int>``.

.. cpp:class:: meta::reverse<S>

    With a member typedef ``type`` which is a reversed meta sequence.

    :example: ``meta::reverse<meta::seq_<char, short, int>>::type`` is ``meta::seq_<int, short, char>``.

.. cpp:class:: meta::transform<F, S>

    With a member typedef ``type`` which is the transformed sequence obtained by applying a meta-function ``F`` to each element type of ``S``.

.. cpp:class:: meta::transform2<F, S1, S2>

    With a member typedef ``type`` which is the transformed sequence obtained by applying a meta-function ``F`` to each element type of ``S1`` and that of ``S2``.

**Examples:**

.. code-block:: cpp

    using namespace clue;
    using meta::int_;
    using meta::seq_;

    using S1 = seq_<int_<1>, int_<2>, int_<3>>;
    using S2 = seq_<int_<4>, int_<5>, int_<6>>;

    using U = typename meta::transform<meta::next, S1>::type;
    // U is seq_<int_<2>, int_<3>, int_<4>>

    using V = typename meta::transform2<meta::plus, S1, S2>::type;
    // V is seq_<int_<5>, int_<7>, int_<9>>

.. cpp:class:: meta::filter<Pred, S>

    With a member typedef ``type`` which is the filtered sequence by retaining the element types ``X`` in ``S`` for which ``Pred<X>::value`` is ``true``.

**Examples:**

.. code-block:: cpp

    using namespace clue;
    using meta::int_;
    using meta::seq_;

    using S = seq_<int_<1>, int_<2>, int_<3>>;

    template<class A>
    struct is_odd : public bool_<(A::value % 2 == 1)> {};

    using R = typename meta::filter<is_odd, S>::type;
    // R is seq_<int_<1>, int_<3>>;

.. cpp:class:: exists<X, S>

    With a member constant ``value`` that indicates whether the type ``X`` exists as an element type of ``S``.

.. cpp:class:: exists_if<Pred, S>

    With a member constant ``value`` which is ``true`` if there exist element types ``X`` of ``S`` such that ``Pred<X>::value`` is ``true``.

.. cpp:class:: count<X, S>

    With a member constant ``value`` which is equal to the number of occurrences of a type ``X`` in the sequence ``S``.

.. cpp:class:: count_if<X, S>

    With a member constant ``value`` which is equal to the number of element types ``X`` in ``S`` that satisfy the condition ``Pred<X>::value`` is ``true``.


Helper aliases are provided for all algorithms that transform types:

.. code-block:: cpp

    template<class S1, class S2>   using cat_t    = typename cat<S1, S2>::type;
    template<class S1, class S2>   using zip_t    = typename zip<S1, S2>::type;
    template<typename X, size_t N> using repeat_t = typename repeat<X, N>::type;

    template<class Seq> using reverse_t = typename reverse<Seq>::type;

    template<template<typename X> class F, class Seq>
    using transform_t = typename transform<F, Seq>::type;

    template<template<typename X, typename Y> class F, class S1, class S2>
    using transform2_t = typename transform2<F, S1, S2>::type;

    template<template<typename X> class Pred, class Seq>
    using filter_t = typename filter<Pred, Seq>::type;
