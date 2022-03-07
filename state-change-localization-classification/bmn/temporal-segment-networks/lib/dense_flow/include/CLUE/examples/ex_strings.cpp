// This shows how to use string_views and tokenizers for parsing

#include <clue/stringex.hpp>
#include <iostream>
#include <sstream>
#include <vector>
#include <string>
#include <stdexcept>

using namespace clue;

const char *TEXT = R"(
# This is a list of attribues
# The symbol `#` is to indicate comments

bar = 100, 20, 3
foo = 13, 568, 24
xyz = 75, 62, 39, 18
)";

struct Record {
    std::string name;
    std::vector<int> nums;

    Record(const std::string& name) :
        name(name) {}

    void add(int v) {
        nums.push_back(v);
    }
};

inline std::ostream& operator << (std::ostream& os, const Record& r) {
    os << r.name << ": ";
    for (int v: r.nums) {
        os << v << ' ';
    }
    return os;
}

int main() {
    // This is to emulate an input file stream
    std::istringstream ss(TEXT);

    // get first line
    char buf[256];
    ss.getline(buf, 256);

    while (ss) {
        // construct a string view out of buffer,
        // and trim leading and trailing spaces
        auto sv = trim(string_view(buf));

        // process each line
        // ignoring empty lines or comments
        if (!sv.empty() && !starts_with(sv, '#')) {
            // format "<name> = <value>"
            // locate '='
            size_t ieq = sv.find('=');

            // note: sub-string of a string view remains a view
            // no copying is done here
            auto name = trim(sv.substr(0, ieq));
            auto rhs = trim(sv.substr(ieq + 1));

            Record record(name.to_string());

            // parse the each term of right-hand-side
            // by tokenizing
            foreach_token_of(rhs, ", ", [&](const char *p, size_t n){
                int v = 0;
                if (try_parse(string_view(p, n), v)) {
                    record.add(v);
                } else {
                    throw std::runtime_error("Invalid integer number.");
                }
                return true;
            });

            std::cout << record << std::endl;
        }

        // get next line
        ss.getline(buf, 256);
    }

    return 0;
}
