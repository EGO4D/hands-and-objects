/**
 * @file type_traits.hpp
 *
 * Some additional type traits facilities introduced in C++14 or 17.
 */

#ifndef CLUE_TYPE_TRAITS__
#define CLUE_TYPE_TRAITS__

#include <clue/common.hpp>

namespace clue {

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

template<class T>
using remove_reference_t = typename ::std::remove_reference<T>::type;

template<class T>
using add_lvalue_reference_t = typename ::std::add_lvalue_reference<T>::type;

template<class T>
using add_rvalue_reference_t = typename ::std::add_rvalue_reference<T>::type;

template<class T>
using make_signed_t = typename ::std::make_signed<T>::type;

template<class T>
using make_unsigned_t = typename ::std::make_unsigned<T>::type;

template<class T>
using remove_extent_t = typename ::std::remove_extent<T>::type;

template<class T>
using remove_all_extents_t = typename ::std::remove_all_extents<T>::type;

template<class T>
using remove_pointer_t = typename ::std::remove_pointer<T>::type;

template<class T>
using add_pointer_t = typename ::std::add_pointer<T>::type;

template<class T>
using decay_t = typename ::std::decay<T>::type;

template<bool B, class T = void>
using enable_if_t = typename ::std::enable_if<B,T>::type;

template<bool B, class T, class F>
using conditional_t = typename ::std::conditional<B,T,F>::type;

template<class... T>
using common_type_t = typename ::std::common_type<T...>::type;

template<class T>
using underlying_type_t = typename ::std::underlying_type<T>::type;

template<class T>
using result_of_t = typename ::std::result_of<T>::type;

}

#endif
