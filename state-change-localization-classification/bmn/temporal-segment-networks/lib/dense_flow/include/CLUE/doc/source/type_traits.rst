Extensions of Type Traits
==========================

In C++11, a collection of type traits have been introduced into the standard library (in the header ``<type_traits>``). While they are very useful, using these type traits in practice is sometimes cumbersome. For example, to add a const qualifier to a type, one has to write

.. code-block:: cpp

    using const_type = typename std::add_const<my_type>::type;

The need to use ``typename`` and ``::type`` introduces unnecessary noise to the code. In C++14, a set of helpers are introduced, such as

.. code-block:: cpp

    template<class T>
    using add_const_t = typename add_const<T>::type;

This makes the codes that transform types more concise. In particular, with ``add_const_t``, one can write:

.. code-block:: cpp

    using const_type = add_const_t<my_type>;

In *CLUE++*, we define all these helpers in the header ``<clue/type_traits.hpp>``, so that they can be used within C++11 environment. In particular, the following helpers are provided. All these *backported* helpers are within the namespace ``clue``.

**Note:** Below is just a list. For detailed descriptions of these type traits, please refer to the `standard documentation <http://en.cppreference.com/w/cpp/header/type_traits>`_.

.. code-block:: cpp

    // for const-volatility specifiers

    template<class T>
    using remove_cv_t = typename ::std::remove_cv<T>::type;
    template<class T>
    using remove_const_t = typename ::std::remove_const<T>::type;
    template<class T>
    using remove_volatile_t = typename ::std::remove_volatile<T>::type;

    template<class T>
    using add_cv_t = typename ::std::add_cv<T>::type;
    template<class T>
    using add_const_t = typename ::std::add_const<T>::type;
    template<class T>
    using add_volatile_t = typename ::std::add_volatile<T>::type;

    // for references

    template<class T>
    using remove_reference_t = typename ::std::remove_reference<T>::type;
    template<class T>
    using add_lvalue_reference_t = typename ::std::add_lvalue_reference<T>::type;
    template<class T>
    using add_rvalue_reference_t = typename ::std::add_rvalue_reference<T>::type;

    // for pointers
    template<class T>
    using remove_pointer_t = typename ::std::remove_pointer<T>::type;
    template<class T>
    using add_pointer_t = typename ::std::add_pointer<T>::type;

    // for sign modifiers

    template<class T>
    using make_signed_t = typename ::std::make_signed<T>::type;
    template<class T>
    using make_unsigned_t = typename ::std::make_unsigned<T>::type;

    // for arrays

    template<class T>
    using remove_extent_t = typename ::std::remove_extent<T>::type;
    template<class T>
    using remove_all_extents_t = typename ::std::remove_all_extents<T>::type;

    // static conditions

    template<bool B, class T = void>
    using enable_if_t = typename ::std::enable_if<B,T>::type;
    template<bool B, class T, class F>
    using conditional_t = typename ::std::conditional<B,T,F>::type;

    // other transformations

    template<class T>
    using decay_t = typename ::std::decay<T>::type;
    template<class... T>
    using common_type_t = typename ::std::common_type<T...>::type;
    template<class T>
    using underlying_type_t = typename ::std::underlying_type<T>::type;
    template<class T>
    using result_of_t = typename ::std::result_of<T>::type;
