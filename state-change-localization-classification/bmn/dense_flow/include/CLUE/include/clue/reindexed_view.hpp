/**
 * @file reindexed_view.hpp
 *
 * The class reindexed_view.
 */

#ifndef CLUE_REINDEXED_VIEW__
#define CLUE_REINDEXED_VIEW__

#include <clue/container_common.hpp>

namespace clue {

namespace details {

template<class Container, class Iter>
class reindexed_iterator {
    static_assert(::std::is_same<
            typename ::std::iterator_traits<Iter>::iterator_category,
            ::std::random_access_iterator_tag>::value,
            "Iter must be a random-access-iterator type.");

private:
    typedef typename ::std::remove_cv<Container>::type container_type;

public:
    typedef typename ::std::random_access_iterator_tag iterator_category;

    typedef typename ::std::iterator_traits<Iter>::difference_type difference_type;

    typedef typename container_type::value_type value_type;

    typedef typename ::std::conditional<::std::is_const<Container>::value,
            typename container_type::const_reference,
            typename container_type::reference>::type reference;

    typedef typename ::std::conditional<::std::is_const<Container>::value,
            typename container_type::const_pointer,
            typename container_type::pointer>::type pointer;

private:
    Container& container_;
    Iter iter_;

public:
    // constructor
    constexpr reindexed_iterator(Container& container, const Iter& iter) noexcept :
        container_(container), iter_(iter) {}

    // comparison
    constexpr bool operator <  (reindexed_iterator r) const noexcept { return iter_ <  r.iter_; }
    constexpr bool operator <= (reindexed_iterator r) const noexcept { return iter_ <= r.iter_; }
    constexpr bool operator >  (reindexed_iterator r) const noexcept { return iter_ >  r.iter_; }
    constexpr bool operator >= (reindexed_iterator r) const noexcept { return iter_ >= r.iter_; }
    constexpr bool operator == (reindexed_iterator r) const noexcept { return iter_ == r.iter_; }
    constexpr bool operator != (reindexed_iterator r) const noexcept { return iter_ != r.iter_; }

    // dereference
    constexpr reference operator* () const {
        return container_[*iter_];
    }

    constexpr pointer operator->() const {
        return &(container_[*iter_]);
    }

    constexpr reference operator[](difference_type n) const {
        return container_[iter_[n]];
    }

    // increment & decrement
    reindexed_iterator& operator++() { ++iter_; return *this; }
    reindexed_iterator& operator--() { --iter_; return *this; }
    reindexed_iterator  operator++(int) { return reindexed_iterator(container_, iter_++); }
    reindexed_iterator  operator--(int) { return reindexed_iterator(container_, iter_--); }

    // arithmetics
    constexpr reindexed_iterator operator + (difference_type n) const {
        return reindexed_iterator(container_, iter_ + n);
    }

    constexpr reindexed_iterator operator - (difference_type n) const {
        return reindexed_iterator(container_, iter_ - n);
    }

    reindexed_iterator& operator += (difference_type n) {
        iter_ += n;
        return *this;
    }

    reindexed_iterator& operator -= (difference_type n) {
        iter_ -= n;
        return *this;
    }

    constexpr difference_type operator - (reindexed_iterator r) const {
        return iter_ - r.iter_;
    }

}; // end class reindexed_iterator

} // end namespace details


template<class Container, class Indices>
class reindexed_view {
    static_assert(::std::is_object<Container>::value,
            "reindexed_view<Container, Indices>: Container must be an object type.");
    static_assert(::std::is_object<Indices>::value,
            "reindexed_view<Container, Indices>: Indices must be an object type.");

public:
    // types
    typedef typename ::std::remove_cv<Container>::type container_type;
    typedef typename ::std::remove_cv<Indices>::type indices_type;
    typedef typename container_type::value_type value_type;
    typedef typename indices_type::size_type size_type;
    typedef typename indices_type::difference_type difference_type;

    typedef typename container_type::const_reference const_reference;
    typedef typename ::std::conditional<::std::is_const<Container>::value,
            typename container_type::const_reference,
            typename container_type::reference>::type reference;

    typedef typename container_type::const_pointer const_pointer;
    typedef typename ::std::conditional<::std::is_const<Container>::value,
            typename container_type::const_pointer,
            typename container_type::pointer>::type pointer;

    typedef details::reindexed_iterator<
            typename ::std::add_const<Container>::type,
            typename indices_type::const_iterator> const_iterator;
    typedef details::reindexed_iterator<
            Container, typename indices_type::const_iterator> iterator;

private:
    Container& container_;
    Indices& indices_;

public:
    // constructors and destructor

    constexpr reindexed_view(Container& container, Indices& indices) noexcept :
        container_(container), indices_(indices) {}

    reindexed_view(const reindexed_view&) = default;

    ~reindexed_view() noexcept = default;

    // size related

    constexpr bool empty() const noexcept(noexcept(indices_.empty())) {
        return indices_.empty();
    }

    constexpr size_type size() const noexcept(noexcept(indices_.size())) {
        return indices_.size();
    }

    constexpr size_type max_size() const noexcept(noexcept(indices_.max_size())) {
        return indices_.max_size();
    }

    // element access

    reference at(size_type pos) {
        return container_.at(indices_.at(pos));
    }

    constexpr const_reference at(size_type pos) const {
        return container_.at(indices_.at(pos));
    }

    reference operator[](size_type pos) {
        return container_[indices_[pos]];
    }

    constexpr const_reference operator[](size_type pos) const {
        return container_[indices_[pos]];
    }

    reference front() {
        return container_[indices_.front()];
    }

    constexpr const_reference front() const {
        return container_[indices_.front()];
    }

    reference back() {
        return container_[indices_.back()];
    }

    constexpr const_reference back() const {
        return container_[indices_.back()];
    }

    // iterators

    iterator begin() noexcept(noexcept(indices_.begin())) {
        return iterator(container_, indices_.begin());
    }

    iterator end() noexcept(noexcept(indices_.end())) {
        return iterator(container_, indices_.end());
    }

    constexpr const_iterator begin() const noexcept(noexcept(indices_.begin())) {
        return const_iterator(container_, indices_.begin());
    }

    constexpr const_iterator end() const noexcept(noexcept(indices_.end())) {
        return const_iterator(container_, indices_.end());
    }

    constexpr const_iterator cbegin() const noexcept(noexcept(indices_.begin())) {
        return begin();
    }

    constexpr const_iterator cend() const noexcept(noexcept(indices_.end())) {
        return end();
    }

}; // end class reindexed_view

template<class Container, class Indices>
constexpr reindexed_view<Container, Indices> reindexed(Container& c, Indices& inds) {
    return reindexed_view<Container, Indices>(c, inds);
}


}

#endif
