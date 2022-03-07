#ifndef CLUE_PREPROC__
#define CLUE_PREPROC__

#include <clue/common.hpp>

#define CLUE_TERMLIST_1(G) G(1)
#define CLUE_TERMLIST_2(G) G(1), G(2)
#define CLUE_TERMLIST_3(G) G(1), G(2), G(3)
#define CLUE_TERMLIST_4(G) G(1), G(2), G(3), G(4)
#define CLUE_TERMLIST_5(G) G(1), G(2), G(3), G(4), G(5)
#define CLUE_TERMLIST_6(G) G(1), G(2), G(3), G(4), G(5), G(6)
#define CLUE_TERMLIST_7(G) G(1), G(2), G(3), G(4), G(5), G(6), G(7)
#define CLUE_TERMLIST_8(G) G(1), G(2), G(3), G(4), G(5), G(6), G(7), G(8)
#define CLUE_TERMLIST_9(G) G(1), G(2), G(3), G(4), G(5), G(6), G(7), G(8), G(9)

#define CLUE_TERMLIST_10(G) G(1), G(2), G(3), G(4), G(5), G(6), G(7), G(8), G(9), G(10)
#define CLUE_TERMLIST_11(G) G(1), G(2), G(3), G(4), G(5), G(6), G(7), G(8), G(9), G(10), G(11)
#define CLUE_TERMLIST_12(G) G(1), G(2), G(3), G(4), G(5), G(6), G(7), G(8), G(9), G(10), G(11), G(12)
#define CLUE_TERMLIST_13(G) G(1), G(2), G(3), G(4), G(5), G(6), G(7), G(8), G(9), G(10), G(11), G(12), G(13)
#define CLUE_TERMLIST_14(G) G(1), G(2), G(3), G(4), G(5), G(6), G(7), G(8), G(9), G(10), G(11), G(12), G(13), G(14)
#define CLUE_TERMLIST_15(G) G(1), G(2), G(3), G(4), G(5), G(6), G(7), G(8), G(9), G(10), G(11), G(12), G(13), G(14), G(15)
#define CLUE_TERMLIST_16(G) G(1), G(2), G(3), G(4), G(5), G(6), G(7), G(8), G(9), G(10), G(11), G(12), G(13), G(14), G(15), G(16)


#define CLUE_GENEXPR_1(G, Op) G(1)
#define CLUE_GENEXPR_2(G, Op) G(1) Op G(2)
#define CLUE_GENEXPR_3(G, Op) G(1) Op G(2) Op G(3)
#define CLUE_GENEXPR_4(G, Op) G(1) Op G(2) Op G(3) Op G(4)
#define CLUE_GENEXPR_5(G, Op) G(1) Op G(2) Op G(3) Op G(4) Op G(5)
#define CLUE_GENEXPR_6(G, Op) G(1) Op G(2) Op G(3) Op G(4) Op G(5) Op G(6)
#define CLUE_GENEXPR_7(G, Op) G(1) Op G(2) Op G(3) Op G(4) Op G(5) Op G(6) Op G(7)
#define CLUE_GENEXPR_8(G, Op) G(1) Op G(2) Op G(3) Op G(4) Op G(5) Op G(6) Op G(7) Op G(8)
#define CLUE_GENEXPR_9(G, Op) G(1) Op G(2) Op G(3) Op G(4) Op G(5) Op G(6) Op G(7) Op G(8) Op G(9)

#define CLUE_GENEXPR_10(G, Op) G(1) Op G(2) Op G(3) Op G(4) Op G(5) Op G(6) Op G(7) Op G(8) Op G(9) Op G(10)
#define CLUE_GENEXPR_11(G, Op) G(1) Op G(2) Op G(3) Op G(4) Op G(5) Op G(6) Op G(7) Op G(8) Op G(9) Op G(10) Op G(11)
#define CLUE_GENEXPR_12(G, Op) G(1) Op G(2) Op G(3) Op G(4) Op G(5) Op G(6) Op G(7) Op G(8) Op G(9) Op G(10) Op G(11) Op G(12)
#define CLUE_GENEXPR_13(G, Op) G(1) Op G(2) Op G(3) Op G(4) Op G(5) Op G(6) Op G(7) Op G(8) Op G(9) Op G(10) Op G(11) Op G(12) Op G(13)
#define CLUE_GENEXPR_14(G, Op) G(1) Op G(2) Op G(3) Op G(4) Op G(5) Op G(6) Op G(7) Op G(8) Op G(9) Op G(10) Op G(11) Op G(12) Op G(13) Op G(14)
#define CLUE_GENEXPR_15(G, Op) G(1) Op G(2) Op G(3) Op G(4) Op G(5) Op G(6) Op G(7) Op G(8) Op G(9) Op G(10) Op G(11) Op G(12) Op G(13) Op G(14) Op G(15)
#define CLUE_GENEXPR_16(G, Op) G(1) Op G(2) Op G(3) Op G(4) Op G(5) Op G(6) Op G(7) Op G(8) Op G(9) Op G(10) Op G(11) Op G(12) Op G(13) Op G(14) Op G(15) Op G(16)


#define CLUE_STMTLIST_1(G) G(1);

#define CLUE_STMTLIST_2(G) \
    G(1); \
    G(2);

#define CLUE_STMTLIST_3(G) \
    G(1); \
    G(2); \
    G(3);

#define CLUE_STMTLIST_4(G) \
    G(1); \
    G(2); \
    G(3); \
    G(4);

#define CLUE_STMTLIST_5(G) \
    G(1); \
    G(2); \
    G(3); \
    G(4); \
    G(5);

#define CLUE_STMTLIST_6(G) \
    G(1); \
    G(2); \
    G(3); \
    G(4); \
    G(5); \
    G(6);

#define CLUE_STMTLIST_7(G) \
    G(1); \
    G(2); \
    G(3); \
    G(4); \
    G(5); \
    G(6); \
    G(7);

#define CLUE_STMTLIST_8(G) \
    G(1); \
    G(2); \
    G(3); \
    G(4); \
    G(5); \
    G(6); \
    G(7); \
    G(8);

#define CLUE_STMTLIST_9(G) \
    G(1); \
    G(2); \
    G(3); \
    G(4); \
    G(5); \
    G(6); \
    G(7); \
    G(8); \
    G(9);

#define CLUE_STMTLIST_10(G) \
    G(1); \
    G(2); \
    G(3); \
    G(4); \
    G(5); \
    G(6); \
    G(7); \
    G(8); \
    G(9); \
    G(10);

#define CLUE_STMTLIST_11(G) \
    G(1); \
    G(2); \
    G(3); \
    G(4); \
    G(5); \
    G(6); \
    G(7); \
    G(8); \
    G(9); \
    G(10); \
    G(11);

#define CLUE_STMTLIST_12(G) \
    G(1); \
    G(2); \
    G(3); \
    G(4); \
    G(5); \
    G(6); \
    G(7); \
    G(8); \
    G(9); \
    G(10); \
    G(11); \
    G(12);

#define CLUE_STMTLIST_13(G) \
    G(1); \
    G(2); \
    G(3); \
    G(4); \
    G(5); \
    G(6); \
    G(7); \
    G(8); \
    G(9); \
    G(10); \
    G(11); \
    G(12); \
    G(13);

#define CLUE_STMTLIST_14(G) \
    G(1); \
    G(2); \
    G(3); \
    G(4); \
    G(5); \
    G(6); \
    G(7); \
    G(8); \
    G(9); \
    G(10); \
    G(11); \
    G(12); \
    G(13); \
    G(14);

#define CLUE_STMTLIST_15(G) \
    G(1); \
    G(2); \
    G(3); \
    G(4); \
    G(5); \
    G(6); \
    G(7); \
    G(8); \
    G(9); \
    G(10); \
    G(11); \
    G(12); \
    G(13); \
    G(14); \
    G(15);

#define CLUE_STMTLIST_16(G) \
    G(1); \
    G(2); \
    G(3); \
    G(4); \
    G(5); \
    G(6); \
    G(7); \
    G(8); \
    G(9); \
    G(10); \
    G(11); \
    G(12); \
    G(13); \
    G(14); \
    G(15); \
    G(16);

#endif
