#ifndef CLUE_TESTIO__
#define CLUE_TESTIO__

#include <clue/common.hpp>
#include <clue/stringex.hpp>
#include <cstring>
#include <fstream>
#include <stdexcept>

namespace clue {

// read file content entirely into a string

inline std::string read_file_content(const char *filename) {
    std::ifstream in(filename, std::ios::in | std::ios::binary);
    if (!in) throw
        std::runtime_error(std::string("Failed to open file ") + filename);

    std::string str;
    in.seekg(0, std::ios::end);
    str.resize(in.tellg());
    in.seekg(0, std::ios::beg);
    in.read(const_cast<char*>(str.data()), str.size());
    in.close();
    return str;
}

inline std::string read_file_content(const std::string& filename) {
    return read_file_content(filename.c_str());
}


// turn a multiline string to a stream of lines

class line_stream {
public:
    class iterator {
    public:
        typedef string_view value_type;
        typedef string_view reference;
        typedef const string_view* pointer;
        typedef std::ptrdiff_t difference_type;
        typedef std::forward_iterator_tag iterator_category;

    private:
        const char *text_;
        size_t len_;
        size_t beg_;
        size_t end_;

    public:
        iterator(const char *text, size_t len, size_t b, size_t e) noexcept
            : text_(text)
            , len_(len)
            , beg_(b), end_(e) {}

        bool operator==(const iterator& r) const noexcept {
            return beg_ == r.beg_;
        }

        bool operator!=(const iterator& r) const noexcept {
            return !(operator==(r));
        }

        string_view operator* () const noexcept {
            return string_view(text_ + beg_, end_ - beg_);
        }

        iterator& operator++() noexcept {
            next_();
            return *this;
        }

        iterator operator++(int) noexcept {
            iterator tmp(*this);
            next_();
            return tmp;
        }

    private:
        void next_() noexcept {
            if (beg_ < len_) {
                beg_ = end_;
                while (end_ < len_ && !is_line_delim(end_)) end_++;
                if (end_ < len_) end_++;
            }
        }

        bool is_line_delim(size_t i) const noexcept {
            return text_[i] == '\n';
        }
    };

    typedef iterator const_iterator;

private:
    const char *text_;
    size_t len_;

public:
    line_stream(const char* text, size_t len)
        : text_(text), len_(len) {}

    explicit line_stream(const char* text)
        : line_stream(text, std::strlen(text)) {}

    explicit line_stream(const std::string& str)
        : text_(str.c_str()), len_(str.size()) {}

    iterator begin() const {
        iterator it(text_, len_, 0, 0);
        return ++it;
    }

    iterator end() const {
        return iterator(text_, len_, len_, len_);
    }

    iterator cbegin() const {
        return begin();
    }

    iterator cend() const {
        return end();
    }
};


}

#endif
