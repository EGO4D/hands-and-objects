Array View
===========

An array view is a light-weight *container-like* wrapper of a pointer and a size. It provides a convenient way to turn a contiguous memory region into a container-like object. In practice, such an object maintains the efficiency of a raw pointer while providing richer API to work with memory regions. Below is an example to illustrate this.

.. code-block:: cpp

    #include <clue/array_view.hpp>

    using namespace clue;

    int a[] = {1, 2, 3, 4, 5};

    for (int& v: aview(a, 5)) {
        std::cout << v << std::endl;
        v += 1;
    }

In practice, it is not uncommon that you maintain a vector in your object and would like to expose the elements to the users (without allowing the users to refer to the vector directly). In such cases, it would be a good idea to return an array view.

.. code-block:: cpp

    using namespace clue;

    class A {
    public:
        // ...

        array_view<const T> elements() const {
            return aview(elems_.data(), elems_.size());
        }

    private:
        std::vector<T> elems_;
    };

    // client code

    A a( /* ... */ );

    std::cout << "# elems = " << a.elements().size() << std::endl;
    for (const T& v: a.elements()) {
        std::cout << v << std::endl;
    }


The ``array_view`` class template
-----------------------------------

.. cpp:class:: array_view<T>

    :param T: The element type.

.. note::

    In general, ``array_view<T>`` allows modification of the elements, *e.g* ``a[i] = x``. To provide a readonly view, one can use ``array_view<const T>``.


Member types
-------------

The class ``array_view<T>`` contains a series of member typedefs as follows:

============================= ============================================
 **types**                     **definitions**
----------------------------- --------------------------------------------
``value_type``                 ``std::remove_cv<T>::type``
``size_type``                  ``std::size_t``
``difference_type``            ``std::ptrdiff_t``
``pointer``                    ``T*``
``const_pointer``              ``const T*``
``reference``                  ``T&``
``const_reference``            ``const T&``
``iterator``                   implementing ``RandomAccessIterator``
``const_iterator``             ``iterator``
``reverse_iterator``           ``std::reverse_iterator<iterator>``
``const_reverse_iterator``     ``std::reverse_iterator<const_iterator>``
============================= ============================================


Construction
-------------

.. cpp:function:: constexpr array_view() noexcept

    Construct an empty view, with null data pointer.

.. cpp:function:: constexpr array_view(pointer data, size_type len) noexcept

    Construct an array view, with data pointer ``data`` and size ``len``.

.. note::

    It also has a copy constructor, an assignment operator, a destructor and a ``swap`` member function, all with default behaviors. It is worth noting that the copy construction/assignment of a view is *shallow*, meaning that only the pointer and the size value are copied, the underlying content remains there.

A convenient function ``aview`` is provided for constructing array views without the need of explicitly articulating the value type.

.. cpp:function:: constexpr array_view<T> aview(T* p, size_t n) noexcept

    Construct an array view, with data pointer ``p`` and size ``n``.

    :note: If ``p`` is of type ``T*``, it returns a view of class ``array_view<T>``, and if ``p`` is a const pointer of type ``const T*``, it returns a view of class ``array_view<const T>``, which is a read-only view.


Basic properties and element access
------------------------------------

.. cpp:function:: constexpr size_type size() const noexcept

    Get the size of the range, *i.e.* the number of elements referred to by the view.

.. cpp:function:: constexpr bool empty() const noexcept

    Get whether the view is empty, *i.e.* refers to no elements.

.. cpp:function:: constexpr const_pointer data() const noexcept

    Get a const pointer to the base address.

.. cpp:function:: pointer data() noexcept

    Get a pointer to the base address.

.. cpp:function:: constexpr const_reference front() const

    Get a const reference to the first element within the view.

.. cpp:function:: reference front()

    Get a reference to the first element within the view.

.. cpp:function:: constexpr const_reference back() const

    Get a const reference to the last element within the view.

.. cpp:function:: reference back()

    Get a reference to the last element within the view.

.. cpp:function:: constexpr const_reference operator[](size_type pos) const

    Get a const reference to the element at position ``pos``, without bounds checking.

.. cpp:function:: reference operator[](size_type pos)

    Get a reference to the element at position ``pos``, without bounds checking.

.. cpp:function:: constexpr const_reference at(size_type pos) const

    Get a const reference to the element at position ``pos``, with bounds checking.

    :throw: an exception of class ``std::out_of_range`` if ``pos >= size()``.

.. cpp:function:: reference at(size_type pos)

    Get a reference to the element at position ``pos``, with bounds checking.

    :throw: an exception of class ``std::out_of_range`` if ``pos >= size()``.


Iterators
----------

.. cpp:function:: constexpr const_iterator cbegin() const

    Get a const iterator to the beginning.

.. cpp:function:: constexpr const_iterator cend() const

    Get a const iterator to the end.

.. cpp:function:: constexpr const_iterator begin() const

    Get a const iterator to the beginning, equivalent to ``cbegin()``.

.. cpp:function:: constexpr const_iterator end() const

    Get a const iterator to the end, equivalent to ``cend()``.

.. cpp:function:: iterator begin()

    Get an iterator to the beginning.

.. cpp:function:: iterator end()

    Get an iterator to the end.

.. cpp:function:: constexpr const_iterator crbegin() const

    Get a const reverse iterator to the reversed beginning.

.. cpp:function:: constexpr const_iterator crend() const

    Get a const reverse iterator to the reversed end.

.. cpp:function:: constexpr iterator rbegin() const

    Get a const reverse iterator to the reversed beginning, equivalent to ``crbegin()``.

.. cpp:function:: constexpr iterator rend() const

    Get a const reverse iterator to the reversed end, equivalent to ``crend()``.

.. cpp:function:: iterator rbegin()

    Get a reverse iterator to the reversed beginning.

.. cpp:function:: iterator rend()

    Get a reverse iterator to the reversed end.
