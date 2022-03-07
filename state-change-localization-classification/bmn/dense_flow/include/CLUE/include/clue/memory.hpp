#ifndef CLUE_MEMORY__
#define CLUE_MEMORY__

#include <clue/common.hpp>

#if (defined(_WIN32) || defined(_WIN64)) && defined(_MSC_VER)
#include <malloc.h>
#else
#include <stdlib.h>
#endif

namespace clue {

#if (defined(_WIN32) || defined(_WIN64)) && defined(_MSC_VER)

inline void* aligned_alloc(size_t nbytes, unsigned int alignment)
{
    void* p = ::_aligned_malloc(nbytes, alignment);
    if (!p)
    {
        throw std::bad_alloc();
    }
    return p;
}

inline void aligned_free(void *p)
{
    ::_aligned_free(p);
}

#else

inline void* aligned_alloc(size_t nbytes, unsigned int alignment)
{
    char* p = 0;
    if (::posix_memalign((void**)(&p), alignment, nbytes) != 0)
    {
        throw std::bad_alloc();
    }
    return p;
}

inline void aligned_free(void *p)
{
    ::free(p);
}

#endif

}

#endif
