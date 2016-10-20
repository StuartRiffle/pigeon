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

    PIECE_TYPES,
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
    CAPTURE_PROMOTE_QUEEN
};

enum
{
    FLAG_TT_BEST_MOVE           = (1 << 0),
    FLAG_PRINCIPAL_VARIATION    = (1 << 1)
};

enum
{
    DISABLE_POPCNT,
    ENABLE_POPCNT
};


typedef i16     EvalTerm;
typedef i32     EvalWeight;

const PDECL int       PIGEON_VER_MAJOR    = 1;
const PDECL int       PIGEON_VER_MINOR    = 6;
const PDECL int       PIGEON_VER_PATCH    = 0;
         
const PDECL int       TT_MEGS_DEFAULT     = 512;
const PDECL int       TT_SAMPLE_SIZE      = 128;
const PDECL int       MAX_SEARCH_DEPTH    = 64;
const PDECL int       METRICS_DEPTH       = 64;
const PDECL int       METRICS_MOVES       = 20;
const PDECL int       MAX_MOVE_LIST       = 218;
const PDECL int       MAX_FEN_LENGTH      = 96;
const PDECL int       MAX_MOVETEXT        = 6;
const PDECL int       MIN_TIME_SLICE      = 10;
const PDECL int       MAX_TIME_SLICE      = 200;
const PDECL int       LAG_SAFETY_BUFFER   = 200;
const PDECL int       NO_TIME_LIMIT       = -1;
const PDECL int       PERFT_PARALLEL_MAX  = 5;
const PDECL int       LAST_QUIET_LEVEL    = -4;
const PDECL int       WEIGHT_SHIFT        = 16;
const PDECL float     WEIGHT_SCALE        = (1 << WEIGHT_SHIFT);
const PDECL bool      OWNBOOK_DEFAULT     = true;
         
const PDECL EvalTerm  EVAL_SEARCH_ABORTED = 0x7FFF;
const PDECL EvalTerm  EVAL_MAX            = 0x7F00;
const PDECL EvalTerm  EVAL_MIN            = -EVAL_MAX;
const PDECL EvalTerm  EVAL_CHECKMATE      = EVAL_MIN + 1;
const PDECL EvalTerm  EVAL_STALEMATE      = 0;
const PDECL int       EVAL_OPENING_PLIES  = 10;
const PDECL EvalTerm  ALLOW_REP_SCORE     = 190;
         
const PDECL u64       HASH_SEED0          = 0xF59C66FB26DCF319ULL;
const PDECL u64       HASH_SEED1          = 0xABCC5167CCAD925FULL;
const PDECL u64       HASH_SEED2          = 0x5121CE64774FBE32ULL;
const PDECL u64       HASH_SEED3          = 0x69852DFD09072166ULL;
const PDECL u64       HASH_UNCALCULATED   = 0;
         
const PDECL u64       SQUARE_A1           = 1ULL << A1;
const PDECL u64       SQUARE_A8           = 1ULL << A8;
const PDECL u64       SQUARE_B1           = 1ULL << B1;
const PDECL u64       SQUARE_B8           = 1ULL << B8;
const PDECL u64       SQUARE_C1           = 1ULL << C1;
const PDECL u64       SQUARE_C8           = 1ULL << C8;
const PDECL u64       SQUARE_D1           = 1ULL << D1;
const PDECL u64       SQUARE_D8           = 1ULL << D8;
const PDECL u64       SQUARE_E1           = 1ULL << E1;
const PDECL u64       SQUARE_E8           = 1ULL << E8;
const PDECL u64       SQUARE_F1           = 1ULL << F1;
const PDECL u64       SQUARE_F8           = 1ULL << F8;
const PDECL u64       SQUARE_G1           = 1ULL << G1;
const PDECL u64       SQUARE_G8           = 1ULL << G8;
const PDECL u64       SQUARE_H1           = 1ULL << H1;
const PDECL u64       SQUARE_H8           = 1ULL << H8;
         
const PDECL u64       FILE_A              = 0x8080808080808080ULL;
const PDECL u64       FILE_B              = 0x4040404040404040ULL;
const PDECL u64       FILE_C              = 0x2020202020202020ULL;
const PDECL u64       FILE_D              = 0x1010101010101010ULL;
const PDECL u64       FILE_E              = 0x0808080808080808ULL;
const PDECL u64       FILE_F              = 0x0404040404040404ULL;
const PDECL u64       FILE_G              = 0x0202020202020202ULL;
const PDECL u64       FILE_H              = 0x0101010101010101ULL;
         
const PDECL u64       RANK_1              = 0x00000000000000FFULL;
const PDECL u64       RANK_2              = 0x000000000000FF00ULL;
const PDECL u64       RANK_3              = 0x0000000000FF0000ULL;
const PDECL u64       RANK_4              = 0x00000000FF000000ULL;
const PDECL u64       RANK_5              = 0x000000FF00000000ULL;
const PDECL u64       RANK_6              = 0x0000FF0000000000ULL;
const PDECL u64       RANK_7              = 0x00FF000000000000ULL;
const PDECL u64       RANK_8              = 0xFF00000000000000ULL;

const PDECL u64       CENTER_DIST_0       = 0x0000001818000000ULL;
const PDECL u64       CENTER_DIST_1       = 0x0000182424180000ULL;
const PDECL u64       CENTER_DIST_2       = 0x0018244242241800ULL;
const PDECL u64       CENTER_DIST_3       = 0x1824428181422418ULL;
const PDECL u64       CENTER_DIST_4       = 0x2442810000814224ULL;
const PDECL u64       CENTER_DIST_5       = 0x4281000000008142ULL;
const PDECL u64       CENTER_DIST_6       = 0x8100000000000081ULL;

const PDECL u64       CENTER_RING_3       = (FILE_A | FILE_H | RANK_1 | RANK_8);
const PDECL u64       CENTER_RING_2       = (FILE_B | FILE_G | RANK_2 | RANK_7) & ~(CENTER_RING_3);
const PDECL u64       CENTER_RING_1       = (FILE_C | FILE_F | RANK_3 | RANK_6) & ~(CENTER_RING_3 | CENTER_RING_2);
const PDECL u64       CENTER_RING_0       = (FILE_D | FILE_E | RANK_4 | RANK_5) & ~(CENTER_RING_3 | CENTER_RING_2 | CENTER_RING_1);
         
const PDECL u64       LIGHT_SQUARES       = 0x55AA55AA55AA55AAULL;
const PDECL u64       DARK_SQUARES        = 0xAA55AA55AA55AA55ULL;
const PDECL u64       ALL_SQUARES         = 0xFFFFFFFFFFFFFFFFULL;
const PDECL u64       CASTLE_ROOKS        = SQUARE_A1 | SQUARE_H1 | SQUARE_A8 | SQUARE_H8;
const PDECL u64       EP_SQUARES          = RANK_3 | RANK_6;
const PDECL u64       CENTER_SQUARES      = (FILE_C | FILE_D | FILE_E | FILE_F) & (RANK_3 | RANK_4 | RANK_5 | RANK_6);
const PDECL u64       CORNER_SQUARES      = (FILE_A | FILE_H) & (RANK_1 | RANK_8);
const PDECL u64       RIM_SQUARES         = (FILE_A | FILE_H | RANK_1 | RANK_8);
const PDECL u64       EDGE_SQUARES        = (FILE_A | FILE_H | RANK_1 | RANK_8) & ~CORNER_SQUARES;


template< int POPCNT, typename T >
INLINE PDECL T CountBits( const T& val )
{ 
    return PlatCountBits64< POPCNT >( val ); 
}

template< typename T > INLINE PDECL T       MaskAllClear()                                                      { return(  T( 0 ) ); }
template< typename T > INLINE PDECL T       MaskAllSet()                                                        { return( ~T( 0 ) ); }
template< typename T > INLINE PDECL T       MaskOut( const T& val, const T& bitsToClear )                       { return( val & ~bitsToClear ); }
template< typename T > INLINE PDECL T       SelectIfNotZero( const T& val, const T& a )                         { return( val? a : 0 ); }
template< typename T > INLINE PDECL T       SelectIfNotZero( const T& val, const T& a, const T& b )             { return( val? a : b ); }
template< typename T > INLINE PDECL T       SelectIfZero(    const T& val, const T& a )                         { return( val? 0 : a ); }
template< typename T > INLINE PDECL T       SelectIfZero(    const T& val, const T& a, const T& b )             { return( val? b : a ); }
template< typename T > INLINE PDECL T       SelectWithMask(  const T& mask, const T& a, const T& b )            { return( b ^ (mask & (a ^ b)) ); } 
template< typename T > INLINE PDECL T       CmpEqual( const T& a, const T& b )                                  { return( (a == b)? MaskAllSet< T >() : MaskAllClear< T >() ); }
template< typename T > INLINE PDECL T       ByteSwap( const T& val )                                            { return PlatByteSwap64( val ); }
template< typename T > INLINE PDECL T       MulLow32( const T& val, u32 scale )                                 { return( val * scale ); }
template< typename T > INLINE PDECL T       SubClampZero( const T& a, const T& b )                              { return( (a > b)? (a - b) : 0 ); }
template< typename T > INLINE PDECL T       LoadIndirect32( const i32* ptr, const T& ofs )                      { return( ((i64) ptr[ofs]) ); }
template< typename T > INLINE PDECL T       LoadIndirectMasked32( const i32* ptr, const T& ofs, const T& mask ) { return( mask? (((i64) ptr[ofs]) & mask) : 0); }
                                
template< typename T > INLINE PDECL T       Min( const T& a, const T& b )                                       { return( (a < b)? a : b ); }
template< typename T > INLINE PDECL T       Max( const T& a, const T& b )                                       { return( (b > a)? b : a ); }
template< typename T > INLINE PDECL T       SignOrZero( const T& val )                                          { return( (val > 0) - (val < 0) ); }
template< typename T > INLINE PDECL T       SquareBit( const T& idx )                                           { return( T( 1 ) << idx ); }
template< typename T > INLINE PDECL T       LowestBit( const T& val )                                           { return( val & -val ); }
template< typename T > INLINE PDECL T       ClearLowestBit( const T& val )                                      { return( val & (val - 1) ); }
template< typename T > INLINE PDECL T       FlipSquareIndex( const T& idx )                                     { return( ((T( 63 ) - idx) & 0x38) | (idx & 0x7) ); }
template< typename T > INLINE PDECL T       XorShiftA( const T& val )                                           { T n = val; return( n ^= (n << 18), n ^= (n >> 31), n ^= (n << 11), n ); }    
template< typename T > INLINE PDECL T       XorShiftB( const T& val )                                           { T n = val; return( n ^= (n << 19), n ^= (n >> 29), n ^= (n <<  8), n ); }    
template< typename T > INLINE PDECL T       XorShiftC( const T& val )                                           { T n = val; return( n ^= (n <<  8), n ^= (n >> 29), n ^= (n << 19), n ); }    
template< typename T > INLINE PDECL T       XorShiftD( const T& val )                                           { T n = val; return( n ^= (n << 11), n ^= (n >> 31), n ^= (n << 18), n ); }    
template< typename T > INLINE PDECL T       ClearBitIndex( const T& val, const T& idx )                         { return( val & ~SquareBit( idx ) ); }
template< typename T > INLINE PDECL T       LowestBitIndex( const T& val )                                      { return PlatLowestBitIndex64( val ); }
template< typename T > INLINE PDECL T       ConsumeLowestBitIndex( T& val )                                     { T idx = LowestBitIndex( val ); val = ClearLowestBit( val ); return( idx ); }
template< typename T > INLINE PDECL void    Exchange( T& a, T& b )                                              { T temp = a; a = b; b = temp; }
template< typename T > INLINE PDECL void    Transpose( const T* src, int srcStep, T* dest, int destStep )       { *dest = *src; }
template< typename T > INLINE PDECL void    SimdInsert( T& dest, u64 val, int lane )                            { dest = val; }
                                        

template< typename T > struct   MoveSpecT;
template< typename T > struct   MoveMapT;
template< typename T > struct   PositionT;

typedef MoveSpecT< u8 >         MoveSpec;
typedef MoveMapT<  u64 >        MoveMap;
typedef PositionT< u64 >        Position;


template< typename T >
struct SimdWidth
{
    enum { LANES = 1 };
};


#endif // PIGEON_DEFS_H__
};
