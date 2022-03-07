Concurrent Queue
=================

Concurrent queue is very useful in concurrent programming. For example, task queue can be considered as a special kind of concurrent queue.
*CLUE* implements a concurrent queue class, in header file ``<clue/concurrent_queue.hpp>``.

.. cpp:class:: concurrent_queue<T>

    Concurrent queue class. ``T`` is the element type.

This class has a default constructor, but it is not copyable or movable. The class provides the following member functions:

.. cpp:function:: size_t size() const

    Get the number of elements in the queue (at the point this method is being called).

.. cpp:function:: bool empty() const

    Get whether the queue is empty (contains no elements).

.. cpp:function:: void synchronize()

    Block until all updating (*e.g.* push or pop) are done.

.. cpp:function:: void clear()

    Clear the queue (pop all remaining elements).

.. cpp:function:: void push(const T& x)

    Push an element ``x`` to the back of the queue.

.. cpp:function:: void push(T&& x)

    Push an element ``x`` (by moving) to the back of the queue.

.. cpp:function:: void emplace(Args&&... args)

    Construct an element using the given arguments and push it to the back of the queue.

.. cpp:function:: bool try_pop(T& dst)

    If the queue is not empty, pop the element at the front, store it to ``dst``, and return ``true``.
    Otherwise, return ``false`` immediately.

.. cpp:function:: T wait_pop()

    Wait until the queue is non-empty, and pop the element at the front and return it.

    If the queue is already non-empty, it pops the front element and returns it immediately.

.. note::

    All updating methods, including ``push``, ``emplace``, ``try_pop``, and ``wait_pop``, are thread-safe.
    It is safe to call these methods in concurrent threads.


**Example:** The following example shows how to use ``concurrent_queue`` to implement a task queue.
In this example, multiple concurrent producers generate items to be processed, and a consumer fetches them from a queue and process.

.. code-block:: cpp

    #include <clue/concurrent_queue.hpp>
    #include <vector>
    #include <thread>
    #include <cstdio>

    inline void process_item(double v) {
        std::printf("process item %g\n", v);
    }

    int main() {
        const size_t M = 2;  // # producers
        const size_t k = 10;  // # items per producer
        size_t remain_nitems = M * k;

        clue::concurrent_queue<double> Q;
        std::vector<std::thread> producers;

        // producers: generate items to be processed
        for (size_t t = 0; t < M; ++t) {
            producers.emplace_back([&Q,t,k](){
                for (size_t i = 0; i < k; ++i) {
                    double v = i + 1;
                    Q.push(v);
                }
            });
        }

        // consumer: process the items
        std::thread consumer([&](){
            while (remain_nitems > 0) {
                process_item(Q.wait_pop());
                -- remain_nitems;
            }
        });

        // wait for all threads to complete
        for (auto& th: producers) th.join();
        consumer.join();
    }

.. note::

    To emulate a typical task queue, one may also push functions as elements, and let the consumer invokes each function that it acquires from the queue.
