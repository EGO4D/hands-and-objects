#ifndef CLUE_STEMPLATE__
#define CLUE_STEMPLATE__

#include <clue/common.hpp>
#include <clue/stringex.hpp>
#include <sstream>
#include <vector>

namespace clue {

// forward declaration
class stemplate;


// stemplate_wrap

template<class Dict>
struct stemplate_wrap {
    const stemplate& templ;
    const Dict& dict;

    std::string str() const;
};

// stemplate

class stemplate {
private:
    enum class PartType {
        Text,
        Term
    };

    struct Part {
        PartType type;
        std::string s;
    };

    std::vector<Part> _parts;

public:
    stemplate(string_view templ) {
        if (!templ.empty()) {
            _build(templ);
        }
    }

    stemplate(const char *templ) {
        if (templ) {
            _build(string_view(templ));
        }
    }

    stemplate(const std::string& templ) {
        if (!templ.empty()) {
            _build(string_view(templ));
        }
    }

    template<class Dict>
    void render(std::ostream& out, const Dict& dict) const {
        for (const Part& part: _parts) {
            if (part.type == PartType::Term) {
                out << dict.at(part.s);
            } else {
                out << part.s;
            }
        }
    }

    template<class Dict>
    stemplate_wrap<Dict> with(const Dict& dict) const {
        return stemplate_wrap<Dict>{*this, dict};
    }

private:
    void _build(string_view templ) {
        string_view sv = templ;
        for(;;) {
            size_t i = sv.find("{{", 0, 2);
            if (i > 0) {
                _add_part(PartType::Text, sv.substr(0, i));
            }
            if (i < sv.size()) {
                size_t j = i + 2;
                size_t r = sv.find("}}", j, 2);
                if (r == sv.npos) {
                    throw std::invalid_argument(
                        "stemplate: invalid template, closing brackets missing.");
                }
                if (r <= j) {
                    throw std::invalid_argument(
                        "stemplate: invalid template, empty term.");
                }
                _add_part(PartType::Term, clue::trim(sv.substr(j, r-j)));
                sv = sv.substr(r+2);
            } else {
                return;
            }
        }
    }

    void _add_part(PartType pty, string_view sv) {
        _parts.push_back(Part{pty, sv.to_string()});
    }
}; // end class stemplate


template<class Dict>
inline std::ostream& operator << (std::ostream& out, const stemplate_wrap<Dict>& w) {
    w.templ.render(out, w.dict);
    return out;
}

template<class Dict>
inline std::string stemplate_wrap<Dict>::str() const {
    std::stringstream ss;
    ss << *this;
    return ss.str();
}


}

#endif
