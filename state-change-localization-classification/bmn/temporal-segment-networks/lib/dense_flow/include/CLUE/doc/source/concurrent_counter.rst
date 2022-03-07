Concurrent Counter
====================

In concurrent programming, it is not uncommon that some function is triggered by a certain condition (*e.g.* a number grow beyond certain threshold).
*CLUE* provides a class ``concurrent_counter`` to implement this. This class in the header file ``<clue/concurrent_counter.hpp>``.

.. cpp:class:: concurrent_counter

    Concurrent counter.

    It has a default constructor that initializes the count to zero.
    There's another constructor that accepts an initial count.

    A concurrent counter is not copyable and not movable.


This class has the following member functions:

.. cpp:function:: long get()

    Get the current value of the counter;

.. cpp:function:: void set(long v)

    Set a new count value, and notify all waiting threads.

.. cpp:function:: void inc(long x = 1)

    Increment the count by ``x`` (default is ``1``), and notify all waiting threads.

.. cpp:function:: void dec(long x = 1)

    Decrement the count by ``x`` (default is ``1``), and notify all waiting threads.

.. cpp:function:: void reset()

    Reset the count value to zero, and notify all waiting threads.
    Equivalent to ``set(0)``.

.. cpp:function:: void wait(Pred&& pred)

    Waits until the count meets the specified condition (``pred(count)`` returns ``true``).

    .. note::

        *CLUE* has provided a series of predicates that can be useful here. (Refer to :ref:`predicates` for details).
        For example, if you want to wait until when the count value goes above a certain threshold ``m``, then
        you may write ``wait( clue::ge(m) )`` (or ``wait( ge(m) )`` when the namespace ``clue`` is being used).

.. cpp:function:: void wait(v)

    Waits until the count hits the given value ``v``.
    Equivalent to ``wait( clue::eq(v) )``.

.. cpp:function:: bool wait_for(Pred&& pred, const std::chrono::duration& dur)

    Waits until the count meets the specified condition or the duration ``dur`` elapses,
    whichever comes first.

    It returns whether the count meets the condition upon returning.

.. cpp:function:: bool wait_for(long v, const std::chrono::duration& dur)

    Equivalent to ``wait_for(clue::eq(v), dur)``.

.. cpp:function:: bool wait_until(Pred&& pred, const std::chrono::time_point& t)

    Waits until the count meets the specified condition or the time-out ``t``,
    whichever comes first.

    It returns whether the count meets the condition upon returning.

.. cpp:function:: bool wait_until(long v, const std::chrono::time_point& t)

    Equivalent to ``wait_until(clue::eq(v), t)``.


**Examples:** The following example shows how a concurrent counter can be used in practice. In this example, a message will be printed when the accumulated value exceeds *100*.

.. code-block:: cpp

    clue::concurrent_counter accum_val(0);

    std::thread worker([&](){
        for (size_t i = 0; i < 100; ++i) {
            accum_val.inc(static_cast<long>(i + 1));
        }
    });

    std::thread listener([&](){
        accum_val.wait( clue::gt(100) );
        std::printf("accum_val goes beyond 100!\n");
    });

    worker.join();
    listener.join();

The source file ``examples/ex_cccounter.cpp`` provides another example.
