/*
 * These test units are adapted from akrzemi1/Optional
 *
 * URL: https://github.com/akrzemi1/Optional/blob/master/test_optional.cpp
 */

#include <clue/optional.hpp>
#include <unordered_set>
#include <gtest/gtest.h>

namespace stdx = clue;

enum State {
    sDefaultConstructed,
    sValueCopyConstructed,
    sValueMoveConstructed,
    sCopyConstructed,
    sMoveConstructed,
    sMoveAssigned,
    sCopyAssigned,
    sValueCopyAssigned,
    sValueMoveAssigned,
    sMovedFrom,
    sValueConstructed
};

struct OracleVal {
    State s;
    int i;
    OracleVal(int i = 0) : s(sValueConstructed), i(i) {}
};

struct Oracle {
    State s;
    OracleVal val;

    Oracle() : s(sDefaultConstructed) {}
    Oracle(const OracleVal& v) : s(sValueCopyConstructed), val(v) {}
    Oracle(OracleVal&& v) : s(sValueMoveConstructed), val(std::move(v)) {v.s = sMovedFrom;}
    Oracle(const Oracle& o) : s(sCopyConstructed), val(o.val) {}
    Oracle(Oracle&& o) : s(sMoveConstructed), val(std::move(o.val)) {o.s = sMovedFrom;}

    Oracle& operator=(const OracleVal& v) { s = sValueCopyConstructed; val = v; return *this; }
    Oracle& operator=(OracleVal&& v) { s = sValueMoveConstructed; val = std::move(v); v.s = sMovedFrom; return *this; }
    Oracle& operator=(const Oracle& o) { s = sCopyConstructed; val = o.val; return *this; }
    Oracle& operator=(Oracle&& o) { s = sMoveConstructed; val = std::move(o.val); o.s = sMovedFrom; return *this; }
};
bool operator==( Oracle const& a, Oracle const& b ) { return a.val.i == b.val.i; }
bool operator!=( Oracle const& a, Oracle const& b ) { return a.val.i != b.val.i; }


TEST(Optional, NilCtor) {
    stdx::optional<int> o1;
    ASSERT_TRUE(!o1);
    ASSERT_FALSE(bool(o1));

    stdx::optional<int> o2 = stdx::nullopt;
    ASSERT_TRUE(!o2);
    ASSERT_FALSE(bool(o2));

    stdx::optional<int> o3 = o2;
    ASSERT_TRUE(!o3);
    ASSERT_FALSE(bool(o3));

    ASSERT_TRUE(o1 == stdx::nullopt);
    ASSERT_TRUE(o1 == stdx::optional<int>{});

    ASSERT_TRUE (o2 == stdx::nullopt);
    ASSERT_TRUE (o2 == stdx::optional<int>{});

    ASSERT_TRUE (o3 == stdx::nullopt);
    ASSERT_TRUE (o3 == stdx::optional<int>{});

    ASSERT_TRUE (o1 == o2);
    ASSERT_TRUE (o2 == o1);
    ASSERT_TRUE (o1 == o3);
    ASSERT_TRUE (o3 == o1);
    ASSERT_TRUE (o2 == o3);
    ASSERT_TRUE (o3 == o2);
}

TEST(Optional, ValueCtor) {
    OracleVal v;
    stdx::optional<Oracle> oo1(v);
    ASSERT_TRUE(oo1 != stdx::nullopt);
    ASSERT_TRUE(oo1 != stdx::optional<Oracle>{});
    ASSERT_TRUE(oo1 == stdx::optional<Oracle>{ v });
    ASSERT_TRUE(!!oo1);
    ASSERT_TRUE(bool(oo1));
    // NA: ASSERT_TRUE (oo1->s == sValueCopyConstructed);
    ASSERT_TRUE(oo1->s == sMoveConstructed);
    ASSERT_TRUE(v.s == sValueConstructed);

    stdx::optional<Oracle> oo2(std::move(v));
    ASSERT_TRUE(oo2 != stdx::nullopt);
    ASSERT_TRUE(oo2 != stdx::optional<Oracle>{});
    ASSERT_TRUE(oo2 == oo1);
    ASSERT_TRUE(!!oo2);
    ASSERT_TRUE(bool(oo2));
    ASSERT_TRUE(oo2->s == sMoveConstructed);
    ASSERT_TRUE(v.s == sMovedFrom);

    {
        OracleVal v;
        stdx::optional<Oracle> oo1 { stdx::in_place, v };
        ASSERT_TRUE(oo1 != stdx::nullopt);
        ASSERT_TRUE(oo1 != stdx::optional<Oracle>{});
        ASSERT_TRUE(oo1 == stdx::optional<Oracle>{ v });
        ASSERT_TRUE(!!oo1);
        ASSERT_TRUE(bool(oo1));
        ASSERT_TRUE(oo1->s == sValueCopyConstructed);
        ASSERT_TRUE(v.s == sValueConstructed);

        stdx::optional<Oracle> oo2 { stdx::in_place, std::move(v) };
        ASSERT_TRUE(oo2 != stdx::nullopt);
        ASSERT_TRUE(oo2 != stdx::optional<Oracle>{});
        ASSERT_TRUE(oo2 == oo1);
        ASSERT_TRUE(!!oo2);
        ASSERT_TRUE(bool(oo2));
        ASSERT_TRUE(oo2->s == sValueMoveConstructed);
        ASSERT_TRUE(v.s == sMovedFrom);
    }
}


TEST(Optional, SimpleCopyMoveCtor) {
    stdx::optional<int> oi;
    stdx::optional<int> oj = oi;

    assert(!oj);
    assert(oj == oi);
    assert(oj == stdx::nullopt);
    assert(!bool(oj));

    oi = 1;
    stdx::optional<int> ok = oi;
    assert(!!ok);
    assert(bool(ok));
    assert(ok == oi);
    assert(ok != oj);
    assert(*ok == 1);

    stdx::optional<int> ol = std::move(oi);
    assert(!!ol);
    assert(bool(ol));
    assert(ol == oi);
    assert(ol != oj);
    assert(*ol == 1);
}


TEST(Optional, OptionalCtor)
{
    stdx::optional<stdx::optional<int>> oi1 = stdx::nullopt;
    assert(oi1 == stdx::nullopt);
    assert(!oi1);

    {
        stdx::optional<stdx::optional<int>> oi2 { stdx::in_place };
        assert(oi2 != stdx::nullopt);
        assert(bool(oi2));
        assert(*oi2 == stdx::nullopt);
    }

    {
        stdx::optional<stdx::optional<int>> oi2 { stdx::in_place, stdx::nullopt };
        assert(oi2 != stdx::nullopt);
        assert(bool(oi2));
        assert(*oi2 == stdx::nullopt);
        assert(!*oi2);
    }

    {
        stdx::optional<stdx::optional<int>> oi2 { stdx::optional<int> { } };
        assert(oi2 != stdx::nullopt);
        assert(bool(oi2));
        assert(*oi2 == stdx::nullopt);
        assert(!*oi2);
    }

    stdx::optional<int> oi;
    auto ooi = stdx::make_optional(oi);
    static_assert( std::is_same<stdx::optional<stdx::optional<int>>, decltype(ooi)>::value, "");
}


TEST(Optional, Assignment) {
    stdx::optional<int> oi;
    oi = stdx::optional<int>{1};
    assert (*oi == 1);

    oi = stdx::nullopt;
    assert (!oi);

    oi = 2;
    assert (*oi == 2);

    oi = {};
    assert (!oi);
};


template<class T>
struct MoveAware {
    T val;
    bool moved;
    MoveAware(T val) :
        val(val), moved(false) {
    }
    MoveAware(MoveAware const&) = delete;
    MoveAware(MoveAware&& rhs) :
        val(rhs.val), moved(rhs.moved) {
        rhs.moved = true;
    }
    MoveAware& operator=(MoveAware const&) = delete;
    MoveAware& operator=(MoveAware&& rhs) {
        val = (rhs.val);
        moved = (rhs.moved);
        rhs.moved = true;
        return *this;
    }
};

TEST(Optional, Move) {
    // first, test mock:
    MoveAware<int> i{ 1 }, j{ 2 };
    ASSERT_TRUE(i.val == 1);
    ASSERT_TRUE(!i.moved);
    ASSERT_TRUE(j.val == 2);
    ASSERT_TRUE(!j.moved);

    MoveAware<int> k = std::move(i);
    ASSERT_TRUE(k.val == 1);
    ASSERT_TRUE(!k.moved);
    ASSERT_TRUE(i.val == 1);
    ASSERT_TRUE(i.moved);

    k = std::move(j);
    ASSERT_TRUE(k.val == 2);
    ASSERT_TRUE(!k.moved);
    ASSERT_TRUE(j.val == 2);
    ASSERT_TRUE(j.moved);

    // now, test optional
    stdx::optional<MoveAware<int>> oi{ 1 }, oj{ 2 };
    ASSERT_TRUE(bool(oi));
    ASSERT_TRUE(!oi->moved);
    ASSERT_TRUE(bool(oj));
    ASSERT_TRUE(!oj->moved);

    stdx::optional<MoveAware<int>> ok = std::move(oi);
    ASSERT_TRUE(bool(ok));
    ASSERT_TRUE(!ok->moved);
    ASSERT_TRUE(bool(oi));
    ASSERT_TRUE(oi->moved);

    ok = std::move(oj);
    ASSERT_TRUE(bool(ok));
    ASSERT_TRUE(!ok->moved);
    ASSERT_TRUE(bool(oj));
    ASSERT_TRUE(oj->moved);
}


// Guard is non-copyable (and non-moveable)
struct Guard {
    std::string val;
    Guard() : val{} {}
    explicit Guard(std::string s, int = 0) : val(s) {}
    Guard(const Guard&) = delete;
    Guard(Guard&&) = delete;
    void operator=(const Guard&) = delete;
    void operator=(Guard&&) = delete;
};

TEST(Optional, WithGuard) {
    // empty
    stdx::optional<Guard> oga;

    // initializes the contained value with "res1"
    stdx::optional<Guard> ogb(stdx::in_place, "res1");
    assert(bool(ogb));
    assert(ogb->val == "res1");

    // default-constructs the contained value
    stdx::optional<Guard> ogc(stdx::in_place);
    assert(bool(ogc));
    assert(ogc->val == "");

    // initialize the contained value with "res1"
    oga.emplace("res1");
    assert(bool(oga));
    assert(oga->val == "res1");

    // destroys the contained value and default-construct a new one
    oga.emplace();
    assert(bool(oga));
    assert(oga->val == "");

    // clear the value
    oga = stdx::nullopt;
    assert(!(oga));
}


bool foo(std::string , stdx::optional<int> oi = stdx::nullopt) {
  return bool(oi);
}

TEST(Optional, Converting) {
    ASSERT_TRUE(foo("dog", 2));
    ASSERT_FALSE(foo("dog"));
    ASSERT_FALSE(foo("dog", stdx::nullopt));
}


TEST(Optional, Value) {
    stdx::optional<int> oi = 1;
    ASSERT_EQ(1, oi.value());

    oi = stdx::nullopt;
    ASSERT_THROW(oi.value(), stdx::bad_optional_access);

    stdx::optional<std::string> os{"AAA"};
    ASSERT_EQ("AAA", os.value());
    os = {};
    ASSERT_THROW(os.value(), stdx::bad_optional_access);
}


TEST(Optional, ValueOr) {
    stdx::optional<int> oi = 1;
    int i = oi.value_or(0);
    ASSERT_EQ(1, i);

    oi = stdx::nullopt;
    ASSERT_EQ(3, oi.value_or(3));

    stdx::optional<std::string> os{"AAA"};
    ASSERT_EQ("AAA", os.value_or("BBB"));
    os = {};
    ASSERT_EQ("BBB", os.value_or("BBB"));
}


TEST(Optional, Comparison) {
    stdx::optional<int> oN { stdx::nullopt };
    stdx::optional<int> o0 { 0 };
    stdx::optional<int> o1 { 1 };

    ASSERT_TRUE((oN < 0));
    ASSERT_TRUE((oN < 1));
    ASSERT_TRUE(!(o0 < 0));
    ASSERT_TRUE((o0 < 1));
    ASSERT_TRUE(!(o1 < 0));
    ASSERT_TRUE(!(o1 < 1));

    ASSERT_TRUE(!(oN >= 0));
    ASSERT_TRUE(!(oN >= 1));
    ASSERT_TRUE((o0 >= 0));
    ASSERT_TRUE(!(o0 >= 1));
    ASSERT_TRUE((o1 >= 0));
    ASSERT_TRUE((o1 >= 1));

    ASSERT_TRUE(!(oN > 0));
    ASSERT_TRUE(!(oN > 1));
    ASSERT_TRUE(!(o0 > 0));
    ASSERT_TRUE(!(o0 > 1));
    ASSERT_TRUE((o1 > 0));
    ASSERT_TRUE(!(o1 > 1));

    ASSERT_TRUE((oN <= 0));
    ASSERT_TRUE((oN <= 1));
    ASSERT_TRUE((o0 <= 0));
    ASSERT_TRUE((o0 <= 1));
    ASSERT_TRUE(!(o1 <= 0));
    ASSERT_TRUE((o1 <= 1));

    ASSERT_TRUE((0 > oN));
    ASSERT_TRUE((1 > oN));
    ASSERT_TRUE(!(0 > o0));
    ASSERT_TRUE((1 > o0));
    ASSERT_TRUE(!(0 > o1));
    ASSERT_TRUE(!(1 > o1));

    ASSERT_TRUE(!(0 <= oN));
    ASSERT_TRUE(!(1 <= oN));
    ASSERT_TRUE((0 <= o0));
    ASSERT_TRUE(!(1 <= o0));
    ASSERT_TRUE((0 <= o1));
    ASSERT_TRUE((1 <= o1));

    ASSERT_TRUE(!(0 < oN));
    ASSERT_TRUE(!(1 < oN));
    ASSERT_TRUE(!(0 < o0));
    ASSERT_TRUE(!(1 < o0));
    ASSERT_TRUE((0 < o1));
    ASSERT_TRUE(!(1 < o1));

    ASSERT_TRUE((0 >= oN));
    ASSERT_TRUE((1 >= oN));
    ASSERT_TRUE((0 >= o0));
    ASSERT_TRUE((1 >= o0));
    ASSERT_TRUE(!(0 >= o1));
    ASSERT_TRUE((1 >= o1));
}


TEST(Optional, Equality) {
    ASSERT_TRUE(stdx::make_optional(0) == 0);
    ASSERT_TRUE(stdx::make_optional(1) == 1);
    ASSERT_TRUE(stdx::make_optional(0) != 1);
    ASSERT_TRUE(stdx::make_optional(1) != 0);

    stdx::optional<int> oN { stdx::nullopt };
    stdx::optional<int> o0 { 0 };
    stdx::optional<int> o1 { 1 };

    ASSERT_TRUE(o0 == 0);
    ASSERT_TRUE(0 == o0);
    ASSERT_TRUE(o1 == 1);
    ASSERT_TRUE(1 == o1);
    ASSERT_TRUE(o1 != 0);
    ASSERT_TRUE(0 != o1);
    ASSERT_TRUE(o0 != 1);
    ASSERT_TRUE(1 != o0);

    ASSERT_TRUE(1 != oN);
    ASSERT_TRUE(0 != oN);
    ASSERT_TRUE(oN != 1);
    ASSERT_TRUE(oN != 0);
    ASSERT_TRUE(!(1 == oN));
    ASSERT_TRUE(!(0 == oN));
    ASSERT_TRUE(!(oN == 1));
    ASSERT_TRUE(!(oN == 0));

    std::string cat { "cat" }, dog { "dog" };
    stdx::optional<std::string> oNil { }, oDog { "dog" }, oCat { "cat" };

    ASSERT_TRUE(oCat == cat);
    ASSERT_TRUE(cat == oCat);
    ASSERT_TRUE(oDog == dog);
    ASSERT_TRUE(dog == oDog);
    ASSERT_TRUE(oDog != cat);
    ASSERT_TRUE(cat != oDog);
    ASSERT_TRUE(oCat != dog);
    ASSERT_TRUE(dog != oCat);

    ASSERT_TRUE(dog != oNil);
    ASSERT_TRUE(cat != oNil);
    ASSERT_TRUE(oNil != dog);
    ASSERT_TRUE(oNil != cat);
    ASSERT_TRUE(!(dog == oNil));
    ASSERT_TRUE(!(cat == oNil));
    ASSERT_TRUE(!(oNil == dog));
    ASSERT_TRUE(!(oNil == cat));
}

TEST(Optional, ConstPropagation) {
    stdx::optional<int> mmi { 0 };
    ASSERT_TRUE((std::is_same<decltype(*mmi), int&>::value));

    const stdx::optional<int> cmi { 0 };
    ASSERT_TRUE((std::is_same<decltype(*cmi), const int&>::value));

    stdx::optional<const int> mci { 0 };
    ASSERT_TRUE((std::is_same<decltype(*mci), const int&>::value));

    stdx::optional<const int> cci { 0 };
    ASSERT_TRUE((std::is_same<decltype(*cci), const int&>::value));
}


TEST(Optional, Hashing) {
    using std::string;

    std::hash<int> hi;
    std::hash<stdx::optional<int>> hoi;
    std::hash<string> hs;
    std::hash<stdx::optional<string>> hos;

    ASSERT_EQ(hi(0), hoi(stdx::optional<int>{0}));
    ASSERT_EQ(hi(1), hoi(stdx::optional<int>{1}));
    ASSERT_EQ(hi(3198), hoi(stdx::optional<int>{3198}));

    ASSERT_EQ(hs(""), hos(stdx::optional<string>{""}));
    ASSERT_EQ(hs("0"), hos(stdx::optional<string>{"0"}));
    ASSERT_EQ(hs("Qa1#"), hos(stdx::optional<string>{"Qa1#"}));

    std::unordered_set<stdx::optional<string>> set;
    ASSERT_TRUE(set.find({"Qa1#"}) == set.end());

    set.insert({"0"});
    ASSERT_TRUE(set.find({"Qa1#"}) == set.end());

    set.insert({"Qa1#"});
    ASSERT_TRUE(set.find({"Qa1#"}) != set.end());
};

