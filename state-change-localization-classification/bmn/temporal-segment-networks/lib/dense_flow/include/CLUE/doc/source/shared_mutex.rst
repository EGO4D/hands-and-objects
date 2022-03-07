Shared Mutex (Read/write lock)
================================

In C++14/C++17, a new kind of *mutex*, called *shared mutex*, is introduced.

Unlike other mutex types, a *shared mutex* has two levels of access:

- **shared:** several threads can share ownership of the same mutex.
- **exclusive:** only one thread can own the mutex.

This is useful in situations where we may allow multiple parallel readers or one writer to operate on a block of data.

As it is part of the future C++ standard, cppreference already has `detailed documentation <http://en.cppreference.com/w/cpp/header/shared_mutex>`_. Below is just a brief summary of the relevant classes and functions. (In *CLUE++*, we provide a C++11 implementation for these types, and the names are in the namespace ``clue``, instead of ``std``).


Here is an example of how shared_mutex can be used in practice.

.. code-block:: cpp

    using namespace clue;

    class MyData {
        std::vector<double> data_;
        mutable shared_mutex mut_;   // the mutex to protect data_;

    public:
        void write() {
            unique_lock<shared_mutex> lk(mut_);
            // ... write to data_ ...
        }

        void read() const {
            shared_lock<shared_mutex> lk(mut_);
            // ... read the data ...
        }
    };

    // --- main program ---

    MyData a;

    std::thread t_write([&](){
        a.write();
        sleep_for_a_while();
    });

    std::thread t_read1([&](){
        a.read();
    });

    std::thread t_read2([&](){
        a.read();
    });

    // t_read1 and t_read2 can simultaneously read a,
    // while t_write is not writing



Class ``shared_mutex``
------------------------

.. cpp:class:: shared_mutex

    A mutex class that allows multiple thread to maintain shared ownership at the same time, or a thread to maintain exclusive ownership.

    :note: It is a default constructor and a destructor, while the copy constructor and assignment operator are deleted.

    :note: This class is accepted to the C++17 standard.

The table below lists its member functions:

.. cpp:function:: void lock()

    Locks the mutex (acquires exclusive ownership), blocks if the mutex is not available.

.. cpp:function:: bool try_lock()

    Tries to lock the mutex.

    Returns immediately. On successful lock acquisition returns true, otherwise returns false.

.. cpp:function:: void unlock()

    Unlocks the mutex.

    :note: The mutex must be locked by the current thread of execution, otherwise, the behavior is undefined.

.. cpp:function:: void lock_shared()

    Acquires shared ownership of the mutex.

    If another thread is holding the mutex in exclusive ownership, a call to lock_shared will block execution until shared ownership can be acquired.

.. cpp:function:: bool try_lock_shared()

    Tries to lock the mutex in shared mode. Returns immediately. On successful lock acquisition returns true, otherwise returns false.

.. cpp:function:: void unlock_shared()

    Releases the mutex from shared ownership by the calling thread.

    :note: The mutex must be locked by the current thread of execution in shared mode, otherwise, the behavior is undefined.


Class ``shared_timed_mutex``
-----------------------------

.. cpp:class:: shared_time_mutex

    Similar to ``shared_mutex``, ``shared_timed_mutex`` allows multiple shared ownership or one exclusive ownership. In addition, it provides the ability to try to acquire the exclusive or shared ownership with a timeout.

    :note: This class is introduced in C++14.

The class ``shared_timed_mutex`` provides all the member funtions as in ``shared_mutex``. In addition, it provides the following members:

.. cpp:function:: bool try_lock_for(const std::chrono::duration<Rep,Period>& duration)

    Tries to lock the mutex (acquire exclusive ownership).

    Blocks until specified ``duration`` has elapsed or the lock is acquired, whichever comes first. On successful lock acquisition returns true, otherwise returns false.

.. cpp:function:: bool try_lock_until( const std::chrono::time_point<Clock,Duration>& t)

    Tries to lock the mutex (acquire exclusive ownership).

    Blocks until specified due time ``t`` has been reached or the lock is acquired, whichever comes first. On successful lock acquisition returns true, otherwise returns false.

.. cpp:function:: bool try_lock_shared_for(const std::chrono::duration<Rep,Period>& duration)

    Tries to lock the mutex in shared mode (acquire shared ownership).

    Blocks until specified ``duration`` has elapsed or the lock is acquired, whichever comes first. On successful lock acquisition returns true, otherwise returns false.

.. cpp:function:: bool try_lock_shared_until( const std::chrono::time_point<Clock,Duration>& t)

    Tries to lock the mutex in shared mode (acquire shared ownership).

    Blocks until specified due time ``t`` has been reached or the lock is acquired, whichever comes first. On successful lock acquisition returns true, otherwise returns false.


Class ``shared_lock``
-----------------------

.. cpp:class:: shared_lock<Mutex>

    The class shared_lock is a general-purpose shared mutex ownership wrapper allowing deferred locking, timed locking and transfer of lock ownership.

    The shared_lock locks the associated shared mutex in shared mode (to lock it in exclusive mode, ``std::unique_lock`` can be used)
