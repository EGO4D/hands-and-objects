String Template
==================

*CLUE* provides a light-weight string template engine, by the class ``stemplate``.

This template engine uses ``{{ name }}`` to indicate the terms to be interpolated, and accepts a dictionary-like object for interpolation.

.. code-block:: cpp

    clue::stemplate st("{{a}} + {{b}} = {{c}}");

    std::unordered_map<std::string, int> dct;
    dct["a"] = 123;
    dct["b"] = 456;
    dct["c"] = 579;

    std::cout << st.with(dct);  // directly write to the output stream
    st.with(dct).str();         // return a rendered string

.. note::

    Here, ``st.with(dct)`` returns a light-weight wrapper that maintains const references to both the template ``st`` and the value dictionary ``dct``.
    When inserted to an output stream, the result is directly written to the output stream.
    One may also call the ``str()`` member function of the wrapper, which would return the rendered string, an object of class ``std::string``.
