// cpu-sse4.h - PIGEON CHESS ENGINE (c) 2012-2016 Stuart Riffle

namespace Pigeon {
#ifndef PIGEON_CPU_SSE4_H__
#define PIGEON_CPU_SSE4_H__

#if PIGEON_ENABLE_SSE4

INLINE __m128i _mm_popcnt_epi64_sse4( const __m128i& v )
{
    static const __m128i nibbleBits = _mm_setr_epi8( 0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4 );

    __m128i loNib = _mm_shuffle_epi8( nibbleBits, _mm_and_si128( v,                      _mm_set1_epi8( 0x0F ) ) );
    __m128i hiNib = _mm_shuffle_epi8( nibbleBits, _mm_and_si128( _mm_srli_epi16( v, 4 ), _mm_set1_epi8( 0x0F ) ) );
    __m128i pop8  = _mm_add_epi8( loNib, hiNib );
    __m128i pop64 = _mm_sad_epu8( pop8, _mm_setzero_si128() );

    return( pop64 );
}

INLINE __m128i _mm_bswap_epi64_sse4( const __m128i& v )
{
    static const __m128i perm = _mm_setr_epi8( 7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8 );

    return( _mm_shuffle_epi8( v, perm ) );
}

struct simd2_sse4
{
    __m128i vec;

    INLINE simd2_sse4()                         {}
    INLINE simd2_sse4( u64 s )                  { vec = _mm_set1_epi64x( s ); }
    INLINE simd2_sse4( const __m128i& v )       { vec = v; }
    INLINE simd2_sse4( const simd2_sse4& v )    { vec = v.vec; }

    INLINE              operator __m128i()                  const { return( vec ); }
    INLINE simd2_sse4   operator~  ()                       const { return( _mm_xor_si128(   vec, _mm_set1_epi8(  ~0 ) ) ); }
    INLINE simd2_sse4   operator-  ( u64 s )                const { return( _mm_sub_epi64(   vec, _mm_set1_epi64x( s ) ) ); }
    INLINE simd2_sse4   operator+  ( u64 s )                const { return( _mm_add_epi64(   vec, _mm_set1_epi64x( s ) ) ); }
    INLINE simd2_sse4   operator&  ( u64 s )                const { return( _mm_and_si128(   vec, _mm_set1_epi64x( s ) ) ); }
    INLINE simd2_sse4   operator|  ( u64 s )                const { return( _mm_or_si128(    vec, _mm_set1_epi64x( s ) ) ); }
    INLINE simd2_sse4   operator^  ( u64 s )                const { return( _mm_xor_si128(   vec, _mm_set1_epi64x( s ) ) ); }
    INLINE simd2_sse4   operator<< ( int c )                const { return( _mm_slli_epi64(  vec, c ) ); }
    INLINE simd2_sse4   operator>> ( int c )                const { return( _mm_srli_epi64(  vec, c ) ); }
    INLINE simd2_sse4   operator<< ( const simd2_sse4& v )  const { return( _mm_sllv_epi64x( vec, v.vec ) ); }
    INLINE simd2_sse4   operator-  ( const simd2_sse4& v )  const { return( _mm_sub_epi64(   vec, v.vec ) ); }
    INLINE simd2_sse4   operator+  ( const simd2_sse4& v )  const { return( _mm_add_epi64(   vec, v.vec ) ); }
    INLINE simd2_sse4   operator&  ( const simd2_sse4& v )  const { return( _mm_and_si128(   vec, v.vec ) ); }
    INLINE simd2_sse4   operator|  ( const simd2_sse4& v )  const { return( _mm_or_si128(    vec, v.vec ) ); }
    INLINE simd2_sse4   operator^  ( const simd2_sse4& v )  const { return( _mm_xor_si128(   vec, v.vec ) ); }
    INLINE simd2_sse4&  operator+= ( const simd2_sse4& v )        { return( vec = _mm_add_epi64( vec, v.vec ), *this ); }
    INLINE simd2_sse4&  operator&= ( const simd2_sse4& v )        { return( vec = _mm_and_si128( vec, v.vec ), *this ); }
    INLINE simd2_sse4&  operator|= ( const simd2_sse4& v )        { return( vec = _mm_or_si128(  vec, v.vec ), *this ); }
    INLINE simd2_sse4&  operator^= ( const simd2_sse4& v )        { return( vec = _mm_xor_si128( vec, v.vec ), *this ); }
};             

template<> INLINE simd2_sse4    CountBits< DISABLE_POPCNT, simd2_sse4 >( const simd2_sse4& val )                                    { return( _mm_popcnt_epi64_sse4( val.vec ) ); }
template<> INLINE simd2_sse4    CountBits< ENABLE_POPCNT,  simd2_sse4 >( const simd2_sse4& val )                                    { return( _mm_popcnt_epi64_sse4( val.vec ) ); }

template<> INLINE simd2_sse4    MaskAllClear<    simd2_sse4 >()                                                                     { return( _mm_setzero_si128() ); } 
template<> INLINE simd2_sse4    MaskAllSet<      simd2_sse4 >()                                                                     { return( _mm_set1_epi8( ~0 ) ); } 
template<> INLINE simd2_sse4    ByteSwap<        simd2_sse4 >( const simd2_sse4& val )                                              { return( _mm_bswap_epi64_sse4( val.vec ) ); }
template<> INLINE simd2_sse4    MulLow32<        simd2_sse4 >( const simd2_sse4& val,  u32 scale )                                  { return( _mm_mul_epu32( val.vec, _mm_set1_epi64x( scale ) ) ); }
template<> INLINE simd2_sse4    MaskOut<         simd2_sse4 >( const simd2_sse4& val,  const simd2_sse4& bitsToClear )              { return( _mm_andnot_si128( bitsToClear.vec, val.vec ) ); }
template<> INLINE simd2_sse4    CmpEqual<        simd2_sse4 >( const simd2_sse4& a,    const simd2_sse4& b )                        { return( _mm_cmpeq_epi64( a.vec, b.vec ) ); }
template<> INLINE simd2_sse4    SelectIfZero<    simd2_sse4 >( const simd2_sse4& val,  const simd2_sse4& a )                        { return( _mm_and_si128( a.vec, _mm_cmpeq_epi64( val.vec, _mm_setzero_si128() ) ) ); }
template<> INLINE simd2_sse4    SelectIfZero<    simd2_sse4 >( const simd2_sse4& val,  const simd2_sse4& a, const simd2_sse4& b )   { return( _mm_select( b.vec, a.vec, _mm_cmpeq_epi64( val.vec, _mm_setzero_si128() ) ) ); }
template<> INLINE simd2_sse4    SelectIfNotZero< simd2_sse4 >( const simd2_sse4& val,  const simd2_sse4& a )                        { return( _mm_andnot_si128( _mm_cmpeq_epi64( val.vec, _mm_setzero_si128() ), a.vec ) ); }
template<> INLINE simd2_sse4    SelectIfNotZero< simd2_sse4 >( const simd2_sse4& val,  const simd2_sse4& a, const simd2_sse4& b )   { return( _mm_select( a.vec, b.vec, _mm_cmpeq_epi64( val.vec, _mm_setzero_si128() ) ) ); }
template<> INLINE simd2_sse4    SelectWithMask<  simd2_sse4 >( const simd2_sse4& mask, const simd2_sse4& a, const simd2_sse4& b )   { return( _mm_select( b.vec, a.vec, mask.vec ) ); }
template<> INLINE simd2_sse4    SubClampZero<    simd2_sse4 >( const simd2_sse4& a,    const simd2_sse4& b )                        { return( _mm_select( _mm_setzero_si128(), _mm_sub_epi64( a.vec, b.vec ), _mm_cmplt_epi32( b.vec, a.vec ) ) ); }

template<>
struct SimdWidth< simd2_sse4 >
{
    enum { LANES = 2 };
};

template<>
void SimdInsert< simd2_sse4 >( simd2_sse4& dest, u64 val, int lane )
{
    dest.vec = (lane == 0)? _mm_insert_epi64( dest.vec, val, 0 ) : _mm_insert_epi64( dest.vec, val, 1 );
}

template<> 
INLINE void Transpose< simd2_sse4 >( const simd2_sse4* src, int srcStep, simd2_sse4* dest, int destStep )
{
    const simd2_sse4* RESTRICT  src_r  = src;
    simd2_sse4* RESTRICT        dest_r = dest;

    dest_r[0]         = _mm_unpacklo_epi64( src_r[0], src_r[srcStep] );
    dest_r[destStep]  = _mm_unpackhi_epi64( src_r[0], src_r[srcStep] );
}

template<>
INLINE simd2_sse4 LoadIndirect32< simd2_sse4 >( const i32* ptr, const simd2_sse4& ofs )
{
    return( _mm_i64gather_epi32_sse2( ptr, ofs ) );
}

template<>
INLINE simd2_sse4 LoadIndirectMasked32< simd2_sse4 >( const i32* ptr, const simd2_sse4& ofs, const simd2_sse4& mask )
{
    return( _mm_mask_i64gather_epi32_sse2( ptr, ofs, mask ) );
}


#endif // PIGEON_ENABLE_SSE4
#endif // PIGEON_CPU_SSE4_H__
};
