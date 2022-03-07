Welcome to CLUE++'s documentation!
===================================

*CLUE++* is a light-weight extensions to the C++ standard library. It provides a collection of components that are widely useful in practical development. These components include a subset of functions and classes introduced in C++14 or later technical specifications (but are also useful for C++11), and some new facilities that we feel are useful in practice.

All components provided by this library are implemented in a way that closely follow the convention of the standard library. In particular, for those components that are *backported* from C++14 or new technical specifications, we strictly follow the standard specification whenever possible (with in the capability of C++11).

Contents:
---------

Basic utilites
~~~~~~~~~~~~~~~~

.. toctree::
   :maxdepth: 1

   optional.rst
   timing.rst
   value_range.rst
   array_view.rst
   reindexed_view.rst
   predicates.rst
   type_name.rst
   misc.rst

String and text processing
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. toctree::
   :maxdepth: 1

   string_view.rst
   stringex.rst
   sformat.rst
   stemplate.rst
   textio.rst

Meta-programming tools
~~~~~~~~~~~~~~~~~~~~~~~

.. toctree::
   :maxdepth: 1

   type_traits.rst
   meta.rst
   meta_seq.rst

Concurrent programming
~~~~~~~~~~~~~~~~~~~~~~~

.. toctree::
   :maxdepth: 1

   shared_mutex.rst
   concurrent_counter.rst
   concurrent_queue.rst
   thread_pool.rst
