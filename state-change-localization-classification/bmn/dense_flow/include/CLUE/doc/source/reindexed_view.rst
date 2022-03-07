Reindexed View
===============

In practice, people often want to work on a selected subset of elements of a sequence. A typical approach is to copy those elements to another container, as

.. code-block:: cpp

    std::vector<int> source{1, 2, 3, 4, 5, 6};
    std::vector<size_t> selected_inds{5, 1, 4};

    std::vector<int> selected;
    selected.reserve(selected_inds.size());
    for (size_t i: selected_inds) {
        selected.push_back(source[i]);
    }

This approach is cumbersome and inefficient. *CLUE++* introduces a class template ``reindexed_view`` to tackle this problem. An reindexed view is an object that *refers to* the selected elements while providing container-like API to work with them. Below is an example:

.. code-block:: cpp

    #include <clue/reindexed_view.hpp>

    using namespace clue;

    std::vector<int> source{1, 2, 3, 4, 5, 6};
    std::vector<size_t> selected_inds{5, 1, 4};

    for (auto v: reindexed(source, selected_inds)) {
        // do something on v
    }

Below is the documentation of this class template and relevant functions.


The ``reindexed_view`` class template
--------------------------------------

.. cpp:class:: reindexed_view<Container, Indices>

    :param Container: The type of the element container.
    :param Indices:   The type of the indices container.

    Both ``Container`` and ``Indices`` need to be random access containers.

.. note::

    Here, the ``Container`` type can be a const type, *e.g.* ``const vector<int>``. Using a constant
    container type as the template argument would lead to a read-only view, which are often very useful.

Member types
-------------

The class ``reindexed_view<Container, Indices>`` contains a series of member typedefs as follows:

============================= ============================================
 **types**                     **definitions**
----------------------------- --------------------------------------------
``container_type``             ``std::remove_cv<Container>::type``
``indices_type``               ``std::remove_cv<Indices>::type``
``value_type``                 ``container_type::value_type``
``size_type``                  ``indices_type::size_type``
``difference_type``            ``indices_type::difference_type``
``const_reference``            ``container_type::const_reference``
``const_pointer``              ``container_type::const_pointer``
``const_iterator``             ``container_type::const_iterator``
============================= ============================================

There are also other member typedefs, whose definitions depend on the *constness* of ``Container``.

.. cpp:type:: reference

    Defined as ``container_type::const_reference`` when ``Container`` is a const type,
    or ``container_type::reference`` otherwise.

.. cpp:type:: pointer

    Defined as ``container_type::const_pointer`` when ``Container`` is a const type, or ``container_type::pointer`` otherwise.

.. cpp:type:: iterator

    Defined as ``container_type::const_iterator`` when ``Container`` is a const type, or ``container_type::iterator`` otherwise.


Construction
-------------

.. cpp:function:: constexpr reindexed_view(Container& container, Indices& indices) noexcept

    Construct a reindexed view, with the given source container and index sequence.

.. note::

    A reindexed view only maintains references to ``container`` and ``indices``. It is the caller's responsibility to ensure that the ``container`` and ``indices`` remain valid while using the view. Otherwise, undefined behaviors may result.

A convenient function ``reindexed`` is provided for creating reindexed views, without requiring the user to explicitly specify the container type and the indices type:

.. cpp:function:: constexpr reindexed_view<Container, Indices> reindexed(Container& c, Indices& inds)

    Construct a reindexed view, with the given source container and index sequence, where the types ``Container`` and ``Indices`` are deduced from arguments.

    :note: If ``c`` is a const reference, then ``Container`` will be deduced to a const type. The same also applies to ``indices``.


Basic properties and element access
-------------------------------------

.. cpp:function:: constexpr bool empty() const noexcept

    Get whether the view is empty (*i.e.* contains no selected elements). It is equal to ``indices.empty()``.

.. cpp:function:: constexpr size_type size() const noexcept

    Get the number of *selected* elements. It is equal to ``indices.size()``.

.. cpp:function:: constexpr size_type max_size() const noexcept

    Get the maximum number of elements that a view can possibly refer to.

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

.. cpp:function:: reference at(size_type pos)

    Get a reference to the element at position ``pos``, with bounds checking.

Iterators
---------

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
