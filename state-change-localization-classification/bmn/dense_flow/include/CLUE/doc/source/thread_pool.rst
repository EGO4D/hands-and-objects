Thread Pool
============

`Thread pool <https://en.wikipedia.org/wiki/Thread_pool>`_ is a very important pattern in concurrent programming. It maps multiple tasks to a smaller number of threads.
This is generally more efficient than spawning one thread for each task, especially when the number of tasks is large.
*CLUE* provides a ``thread_pool`` class in the header file ``<clue/thread_pool.hpp>``.

.. cpp:class:: thread_pool

    A thread pool class.

    By default, ``thread_pool()`` constructs a thread pool with zero threads. ``thread_pool(n)`` constructs a thread pool with ``n`` threads.
    One can modify the number of threads using the ``resize()`` method later.

    A thread pool is not copyable and not movable.

The ``thread_pool`` class provides the following member functions:

.. cpp:function:: bool empty() const noexcept

    Return whether the pool is empty (contains no threads).

.. cpp:function:: size_t size() const noexcept

    Get the number of threads maintained by the pool.

.. cpp:function:: std::thread& get_thread(size_t i)

    Get a reference to the ``i``-th thread.

.. cpp:function:: const std::thread& get_thread(size_t i) const

    Get a const-reference to the ``i``-th thread.

.. cpp:function:: size_t num_scheduled_tasks() const noexcept

    Get the total number of scheduled tasks (all the tasks that have ever been pushed to the queue).

.. cpp:function:: size_t num_completed_tasks() const noexcept

    Get the total number of tasks that have been completed.

.. cpp:function:: bool stopped() const noexcept

    Get whether the thread pool has been stopped (by calling ``stop()``).

.. cpp:function:: bool done() const noexcept

    Get whether all scheduled tasks have been done.

.. cpp:function:: void resize(n)

    Resize the pool to ``n`` threads.

    .. note::

        When ``n`` is less than ``size()``, the pool will be shrinked, trailing threads will be terminated and detached.

.. cpp:function:: std::future<R> schedule(F&& f)

    Schedule a task.

    Here, ``f`` should be a functor/function that accepts a thread index of type ``size_t`` as an argument.
    This function returns a future of class ``std::future<R>``, where ``R`` is the return type of ``f``.

    This function would wrap ``f`` into a ``packaged_task`` and push it to the internal task queue. When a thread is available,
    it will try to get a task from the front of the internal task queue and execute it.

    .. note::

        It is straightforward to push a function that accepts more arguments. One can just wrap it into a closure using C++11's lambda function.

.. cpp:function:: void synchronize()

    Block until all current tasks have been completed.

    This function does not close the thread pool or stop any threads. After synchronization, one can continue to schedule new tasks.

    .. note::

        Multiple threads can synchronize a thread pool at the same time.
        However, it is not allowed to schedule a task while some one is synchronizing.

.. cpp:function:: void close(bool stop_cmd=false)

    Close the queue, so that no new tasks can be scheduled.

    If ``stop_cmd`` is explicitly set to ``true``, it also sends a stopping command to all threads.

    .. note::

        This function returns immediately after closing the queue (and optionally sending the stopping command).
        It won't wait for the threads to finish (for this purpose, one can call ``join()``).

.. cpp:function:: void close_and_stop()

    Equivalent to ``close(true)``.

.. cpp:function:: void join()

    Block until all threads finish.

    A thread will finish when the current task is completed and then no task can be acquired (the queue is closed and empty) or when it is stopped explicitly by the stopping command.

    .. note::

        The thread pool can only be joined when it is closed. Otherwise a runtime error will be raised.
        Also, when all threads finish, the function, this function will clear the thread pool, resizing it
        to ``0`` threads. However, one can call ``resize(n)`` to reinstantiate a new set of threads.

.. cpp:function:: void wait_done()

    Block until all tasks are completed.
    Equivalent to ``close(); join();``.

.. cpp:function:: void stop_and_wait()

    Block until all active tasks (those being run) are completed. Tasks that have been scheduled but have not been launched will remain in the queue (but won't be run by threads).

    This is equivalent to ``close_and_stop(); join();``.

    One can later call ``resize()`` to re-instate a new set of threads to complete the remaining tasks or call
    ``clear_tasks()`` to clear all remaining tasks.

.. cpp:function:: void clear_tasks()

    Clear all tasks that remain in the queue. This function won't affect those tasks that are being executed.



**Example:** The following example shows how to schedule tasks and wait until when they are all done.

.. code-block:: cpp

    #include <clue/thread_pool.hpp>

    void my_task(double arg) {
        // some processing ...
    }

    int main() {
        // construct a thread pool with 4 threads
        clue::thread_pool P(4);

        size_t n = 20;
        for (size_t i = 0; i < n; ++i) {
            double a = // get an argument;

            // tid is the index of the thread
            P.schedule([](size_t tid){ my_task(a); });
        }

        // wait until all tasks are completed
        P.wait_done();
    }
