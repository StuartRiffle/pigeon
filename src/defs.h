// defs.h - PIGEON CHESS ENGINE (c) 2012-2016 Stuart Riffle

namespace Pigeon {
#ifndef PIGEON_DEFS_H__
#define PIGEON_DEFS_H__

enum
{
    H1, G1, F1, E1, D1, C1, B1, A1,
    H2, G2, F2, E2, D2, C2, B2, A2,
    H3, G3, F3, E3, D3, C3, B3, A3,
    H4, G4, F4, E4, D4, C4, B4, A4,
    H5, G5, F5, E5, D5, C5, B5, A5,
    H6, G6, F6, E6, D6, C6, B6, A6,
    H7, G7, F7, E7, D7, C7, B7, A7,
    H8, G8, F8, E8, D8, C8, B8, A8,
};

enum
{
    SHIFT_N    =  8,
    SHIFT_NW   =  9,
    SHIFT_W    =  1,
    SHIFT_SW   = -7,
    SHIFT_S    = -8,
    SHIFT_SE   = -9,
    SHIFT_E    = -1,
    SHIFT_NE   =  7,
};

enum 
{
    PAWN,
    KNIGHT,
    BISHOP,
    ROOK,
    QUEEN,
    KING,

    NO_PROMOTION = PAWN
};

enum 
{
    MOVE,
    CAPTURE_LOSING,
    CAPTURE_EQUAL,
    CAPTURE_WINNING,
    PROMOTE_KNIGHT,
    PROMOTE_BISHOP,
    PROMOTE_ROOK,
    PROMOTE_QUEEN,
    CAPTURE_PROMOTE_KNIGHT,
    CAPTURE_PROMOTE_BISHOP,
    CAPTURE_PROMOTE_ROOK,
    CAPTURE_PROMOTE_QUEEN,
    TT_BEST_MOVE,
    PRINCIPAL_VARIATION
};



typedef i16 EvalTerm;

const int       TT_MEGS_DEFAULT     = 64;
const int       QUIET_SEARCH_LIMIT  = 32;
const int       MAX_METRICS_DEPTH   = 64;
const int       MAX_MOVE_LIST       = 218;
const int       MAX_FEN_LENGTH      = 96;
const int       MAX_MOVETEXT        = 6;
const int       MIN_TIME_SLICE      = 20;
const int       MAX_TIME_SLICE      = 200;
const int       LAG_SAFETY_BUFFER   = 500;
const int       NO_TIME_LIMIT       = -1;
const int       PERFT_PARALLEL_MAX  = 5;
const int       PIGEON_VER_MAJ      = 1;
const int       PIGEON_VER_MIN      = 31;

const EvalTerm  EVAL_SEARCH_ABORTED = 0x7FFF;
const EvalTerm  EVAL_MAX            = 0x7F00;
const EvalTerm  EVAL_MIN            = -EVAL_MAX;
const EvalTerm  EVAL_NO_MOVES       = EVAL_MIN + 1;
const int       EVAL_OPENING_PLIES  = 12;
const int       EVAL_MIDGAME_BLEND  = 8;

const u64       HASH_SEED0          = 0xF59C66FB26DCF319ULL;
const u64       HASH_SEED1          = 0xABCC5167CCAD925FULL;
const u64       HASH_SEED2          = 0x5121CE64774FBE32ULL;
const u64       HASH_SEED3          = 0x69852DFD09072166ULL;
const u64       HASH_UNCALCULATED   = 0;

const u64       SQUARE_A1           = 1ULL << A1;
const u64       SQUARE_A8           = 1ULL << A8;
const u64       SQUARE_B1           = 1ULL << B1;
const u64       SQUARE_B8           = 1ULL << B8;
const u64       SQUARE_C1           = 1ULL << C1;
const u64       SQUARE_C8           = 1ULL << C8;
const u64       SQUARE_D1           = 1ULL << D1;
const u64       SQUARE_D8           = 1ULL << D8;
const u64       SQUARE_E1           = 1ULL << E1;
const u64       SQUARE_E8           = 1ULL << E8;
const u64       SQUARE_F1           = 1ULL << F1;
const u64       SQUARE_F8           = 1ULL << F8;
const u64       SQUARE_G1           = 1ULL << G1;
const u64       SQUARE_G8           = 1ULL << G8;
const u64       SQUARE_H1           = 1ULL << H1;
const u64       SQUARE_H8           = 1ULL << H8;

const u64       FILE_A              = 0x8080808080808080ULL;
const u64       FILE_B              = 0x4040404040404040ULL;
const u64       FILE_C              = 0x2020202020202020ULL;
const u64       FILE_D              = 0x1010101010101010ULL;
const u64       FILE_E              = 0x0808080808080808ULL;
const u64       FILE_F              = 0x0404040404040404ULL;
const u64       FILE_G              = 0x0202020202020202ULL;
const u64       FILE_H              = 0x0101010101010101ULL;

const u64       RANK_1              = 0x00000000000000FFULL;
const u64       RANK_2              = 0x000000000000FF00ULL;
const u64       RANK_3              = 0x0000000000FF0000ULL;
const u64       RANK_4              = 0x00000000FF000000ULL;
const u64       RANK_5              = 0x000000FF00000000ULL;
const u64       RANK_6              = 0x0000FF0000000000ULL;
const u64       RANK_7              = 0x00FF000000000000ULL;
const u64       RANK_8              = 0xFF00000000000000ULL;

const u64       LIGHT_SQUARES       = 0x55AA55AA55AA55AAULL;
const u64       DARK_SQUARES        = 0xAA55AA55AA55AA55ULL;
const u64       ALL_SQUARES         = 0xFFFFFFFFFFFFFFFFULL;
const u64       CASTLE_ROOKS        = SQUARE_A1 | SQUARE_H1 | SQUARE_A8 | SQUARE_H8;
const u64       EP_SQUARES          = RANK_3 | RANK_6;
const u64       EDGE_SQUARES        = FILE_A | FILE_H | RANK_1 | RANK_8;
const u64       CENTER_SQUARES      = (FILE_C | FILE_D | FILE_E | FILE_F) & (RANK_3 | RANK_4 | RANK_5 | RANK_6);


template< typename T > INLINE   int     SimdWidth()															{ return( 1 ); }
template< typename T > INLINE   bool    SimdSupported()														{ return( false ); }
template< typename T > INLINE   T       MaskAllClear()                                                      { return(  T( 0 ) ); }
template< typename T > INLINE   T       MaskAllSet()                                                        { return( ~T( 0 ) ); }
template< typename T > INLINE   T       MaskOut( const T& val, const T& bitsToClear )                       { return( val & ~bitsToClear ); }
template< typename T > INLINE   T       SelectIfNotZero( const T& val, const T& a )                         { return( val? a : 0 ); }
template< typename T > INLINE   T       SelectIfNotZero( const T& val, const T& a, const T& b )             { return( val? a : b ); }
template< typename T > INLINE   T       SelectIfZero(    const T& val, const T& a )                         { return( val? 0 : a ); }
template< typename T > INLINE   T       SelectIfZero(    const T& val, const T& a, const T& b )             { return( val? b : a ); }
template< typename T > INLINE   T       SelectWithMask(  const T& mask, const T& a, const T& b )            { return( b ^ (mask & (a ^ b)) ); } 
template< typename T > INLINE   T       CmpEqual( const T& a, const T& b )                                  { return( (a == b)? MaskAllSet< T >() : MaskAllClear< T >() ); }
template< typename T > INLINE   T       ByteSwap( const T& val )                                            { return PlatByteSwap64( val ); }
template< typename T > INLINE   T       CountBits( const T& val )                                           { return PlatCountBits64( val ); }
template< typename T > INLINE   T       MulLow32( const T& val, u32 scale )                                 { return( val * scale ); }

template< typename T > INLINE   T       Min( const T& a, const T& b )                                       { return( (a < b)? a : b ); }
template< typename T > INLINE   T       Max( const T& a, const T& b )                                       { return( (b > a)? b : a ); }
template< typename T > INLINE   T       SignOrZero( const T& val )                                          { return( (val > 0) - (val < 0) ); }
template< typename T > INLINE   T       SquareBit( const T& idx )                                           { return( T( 1 ) << idx ); }
template< typename T > INLINE   T       LowestBit( const T& val )                                           { return( val & -val ); }
template< typename T > INLINE   T       ClearLowestBit( const T& val )                                      { return( val & (val - 1) ); }
template< typename T > INLINE   T       FlipSquareIndex( const T& idx )                                     { return( ((T( 63 ) - idx) & 0x38) | (idx & 0x7) ); }
template< typename T > INLINE   T       XorShiftA( T n )                                                    { return( n ^= (n << 18), n ^= (n >> 31), n ^= (n << 11), n ); }    
template< typename T > INLINE   T       XorShiftB( T n )                                                    { return( n ^= (n << 19), n ^= (n >> 29), n ^= (n <<  8), n ); }    
template< typename T > INLINE   T       XorShiftC( T n )                                                    { return( n ^= (n <<  8), n ^= (n >> 29), n ^= (n << 19), n ); }    
template< typename T > INLINE   T       XorShiftD( T n )                                                    { return( n ^= (n << 11), n ^= (n >> 31), n ^= (n << 18), n ); }    
template< typename T > INLINE   T       ClearBitIndex( const T& val, const T& idx )                         { return( val & ~SquareBit( idx ) ); }
template< typename T > INLINE   T       LowestBitIndex( const T& val )                                      { return PlatLowestBitIndex64( val ); }
template< typename T > INLINE   T       ConsumeLowestBitIndex( T& val )                                     { T idx = LowestBitIndex( val ); val = ClearLowestBit( val ); return( idx ); }
template< typename T > INLINE   void    Exchange( T& a, T& b )                                              { T temp = a; a = b; b = temp; }





template< typename T > struct MoveSpecT;
template< typename T > struct MoveMapT;
template< typename T > struct PositionT;

typedef MoveSpecT< u8 >     MoveSpec;
typedef MoveMapT<  u64 >    MoveMap;
typedef PositionT< u64 >    Position;

#endif // PIGEON_DEFS_H__
};
