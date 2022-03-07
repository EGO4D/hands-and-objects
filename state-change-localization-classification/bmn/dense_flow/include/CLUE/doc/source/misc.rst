Miscellaneous Utilities
========================

*CLUE* also provides some utilities that are handy in programming practice. These utilities are provided by the header ``<clue/misc.hpp>``.

.. cpp:function:: make_unique<T>(args...)

    Constructs an object of type ``T`` and wraps it in a unique pointer of class ``std::unique_ptr<T>``.

    Here, ``args`` are the arguments to be forwarded to the constructor.

    It is equivalent to ``unique_ptr<T>(new T(std::forward<Args>(args)...))``.

.. cpp:function:: pass(args...)

    Accepts arbitrary arguments and does nothing.

    The purpose of this function is mainly to trigger the execution of all the argument expressions in a variadic context.

.. cpp:class:: temporary_buffer<T>

    Tempoary buffer.

    An object of this class invokes ``std::get_temporary_buffer`` on construction, and ``std::return_temporary_buffer`` on destruction.

    .. note::

        A temporary buffer is supposed to be used locally, and it is not copyable or movable.

    **Examples:**

    .. code-block:: cpp

        #include <clue/misc.hpp>

        void myfun(size_t n) {
            // calls std::get_temporary_buffer to acquire a buffer
            // that can host at least n integers.
            temporary_buffer<int> buf(n);

            size_t cap = buf.capacity();  // get the size that are actually allocated
            int *p = buf.data();  // get the memory address

            // do somthing with buf ...

            // upon exit, the buffer will be returned by calling
            // std::return_temporary_buffer
        }
