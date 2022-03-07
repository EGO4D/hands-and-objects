.. _predicates:

Predicates
============

*CLUE++* provides a series of higher-order functions for generating predicates (functors that returns ``bool``), in the header ``<clue/predicates.hpp>``.
These predicates can be very useful in programming for expressing certain conditions.

Take a look of the following example, where we want to determine whether all elements are positive.
With C++11, this can be accomplished as:

.. code-block:: cpp

    std::all_of(s.begin(), s.end(), [](int x){ return x > 0; });

This is convenient enough. However, still expressing simple conditions like positiveness using a full-fledged lambda expression remains cumbersome, especially when there are many conditions to express. *CLUE* provides a higher-order function ``gt``, with which ``[](int x){ return x > 0; }`` above can be simplified as ``gt(0)``, and consequently
the code above can be rewritten as:

    std::all_of(s.begin(), s.end(), gt(0));

Generic predicates
--------------------

The following table lists the predicates provided by *CLUE*. Let ``x`` be the value to be tested by the predicates.
Note that all these predicates are in the namespace ``clue``.

===================== ===========================================================================
 functors               conditions
--------------------- ---------------------------------------------------------------------------
``eq(v)``               ``x == v``
``ne(v)``               ``x != v``
``gt(v)``               ``x > v``
``ge(v)``               ``x >= v``
``lt(v)``               ``x < v``
``le(v)``               ``x <= v``
``in(s)``               ``x`` is in ``s``, *i.e.* ``x`` is equal to one of the elements of ``s``
===================== ===========================================================================

.. note::

    For ``in(s)``, ``s`` can be a C-string. In this case, the generated predicate returns ``true``,
    when the input character ``x`` equals one of the character in ``s``.

*CLUE* also provides ``and_`` and ``or_`` to combine conditions.

.. cpp:function:: and_(p1, p2, ...)

    Return a predicate, which returns ``true`` for an argument ``x`` when ``p1(x) && p2(x) && ...``.

    **Example:** to express the condition like ``a < x < b``, one can write ``and_(gt(a), lt(b))``, or
    if it is a closed interval as ``a <= x <= b``, then one can write ``and_(ge(a), le(b))``.

.. cpp:function:: or_(p1, p2, ...)

    Return a predicate, which returns ``true`` for an argument ``x`` when ``p1(x) || p2(x) || ...``.

    **Example:** ``or_(eq(a), eq(b), eq(c))`` expresses the condition that ``x`` is equal to either
    ``a``, ``b``, or ``c``.

Char predicates
-----------------

*CLUE* provides several predicates for testing characters (of type ``char`` or ``wchar_t``) within the namespace ``clue::chars``, as follows.
These functors can be very useful in text parsing.

===================== ========================
 functors               conditions
--------------------- ------------------------
``chars::is_space``    ``std::isspace(x)``
``chars::is_blank``    ``std::isblank(x)``
``chars::is_digit``    ``std::isdigit(x)``
``chars::is_xdigit``   ``std::isxdigit(x)``
``chars::is_alpha``    ``std::isalpha(x)``
``chars::is_alnum``    ``std::isalnum(x)``
``chars::is_punct``    ``std::ispunct(x)``
``chars::is_upper``    ``std::isupper(x)``
``chars::is_lower``    ``std::islower(x)``
===================== ========================

.. note::

    All these ``is_space`` etc are typed functors. Unlike the C-function such as ``isspace``, these functors are likely to be inlined
    when passed to higher-level algorithms (*e.g.* ``std::all_of``, ``std::find``, etc). Also these functors work with both ``char``
    and ``wchar_t``. For example, ``char::is_space(c)`` calls ``std::iswspace`` internally when ``c`` is of type ``wchar_t``.

Float predicates
-----------------

*CLUE* also provides predicates for testing floating point numbers, within the namespace ``clue::floats``.

===================== ========================
 functors               conditions
--------------------- ------------------------
``floats::is_finite``   ``std::isfinite(x)``
``floats::is_inf``      ``std::isinf(x)``
``floats::is_nan``      ``std::isnan(x)``
===================== ========================

.. note::

    These functors work with ``float``, ``double``, and ``long double``.
