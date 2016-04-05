// simd.h - PIGEON CHESS ENGINE (c) 2012-2016 Stuart Riffle

namespace Pigeon {
#ifndef PIGEON_SIMD_H__
#define PIGEON_SIMD_H__





// SSE2

INLINE __m128i _mm_select( const __m128i& a, const __m128i& b, const __m128i& mask )
{          
    return _mm_xor_si128( a, _mm_and_si128( mask, _mm_xor_si128( b, a ) ) ); // mask? b : a
}

INLINE __m128i _mm_cmpeq_epi64_sse2( const __m128i& a, const __m128i& b )
{
    __m128i eq32 = _mm_cmpeq_epi32( a, b );
    __m128i eq64 = _mm_and_si128( eq32, _mm_shuffle_epi32( eq32, _MM_SHUFFLE( 2, 3, 0, 1 ) ) );

    return( eq64 );
}

INLINE __m128i _mm_popcnt_epi64_sse2( __m128i v )
{
    // BROKEN?

    __m128i mask    = _mm_set1_epi8( 0x77 );
    __m128i n1      = _mm_and_si128( mask, _mm_srli_epi64( v,  1 ) );
    __m128i n2      = _mm_and_si128( mask, _mm_srli_epi64( n1, 1 ) );
    __m128i n3      = _mm_and_si128( mask, _mm_srli_epi64( n2, 1 ) );

    v = _mm_sub_epi8( _mm_sub_epi8( _mm_sub_epi8( v, n1 ), n2 ), n3 );
    v = _mm_add_epi8( v, _mm_srli_epi16( v, 4 ) );
    v = _mm_and_si128( _mm_set1_epi8( 0x0F ), v );
    v = _mm_sad_epu8( v, _mm_setzero_si128() );

    return( v );
}

INLINE __m128i _mm_bswap_epi64_sse2( const __m128i& v )
{
    __m128i swap16  = _mm_or_si128( _mm_slli_epi16( v, 8 ), _mm_srli_epi16( v, 8 ) );
    __m128i shuflo  = _mm_shufflelo_epi16( swap16, _MM_SHUFFLE( 0, 1, 2, 3 ) );
    __m128i result  = _mm_shufflehi_epi16( shuflo, _MM_SHUFFLE( 0, 1, 2, 3 ) );

    return( result );
}
                                    
struct simd2_sse2
{
    __m128i vec;

    INLINE simd2_sse2() {}
    INLINE simd2_sse2( u64 s )                              : vec( _mm_set1_epi64x( s ) ) {}
    INLINE simd2_sse2( const __m128i& v )                   : vec( v ) {}
    INLINE simd2_sse2( const simd2_sse2& v )                : vec( v.vec ) {}

    INLINE simd2_sse2   operator~  ()                       const { return( _mm_xor_si128(  vec, _mm_set1_epi8(  ~0 ) ) ); }
    INLINE simd2_sse2   operator-  ( u64 s )                const { return( _mm_sub_epi64(  vec, _mm_set1_epi64x( s ) ) ); }
    INLINE simd2_sse2   operator+  ( u64 s )                const { return( _mm_add_epi64(  vec, _mm_set1_epi64x( s ) ) ); }
    INLINE simd2_sse2   operator&  ( u64 s )                const { return( _mm_and_si128(  vec, _mm_set1_epi64x( s ) ) ); }
    INLINE simd2_sse2   operator|  ( u64 s )                const { return( _mm_or_si128(   vec, _mm_set1_epi64x( s ) ) ); }
    INLINE simd2_sse2   operator^  ( u64 s )                const { return( _mm_xor_si128(  vec, _mm_set1_epi64x( s ) ) ); }
    INLINE simd2_sse2   operator<< ( int c )                const { return( _mm_slli_epi64( vec, c ) ); }
    INLINE simd2_sse2   operator>> ( int c )                const { return( _mm_srli_epi64( vec, c ) ); }
    INLINE simd2_sse2   operator-  ( const simd2_sse2& v )  const { return( _mm_sub_epi64(  vec, v.vec ) ); }
    INLINE simd2_sse2   operator+  ( const simd2_sse2& v )  const { return( _mm_add_epi64(  vec, v.vec ) ); }
    INLINE simd2_sse2   operator&  ( const simd2_sse2& v )  const { return( _mm_and_si128(  vec, v.vec ) ); }
    INLINE simd2_sse2   operator|  ( const simd2_sse2& v )  const { return( _mm_or_si128(   vec, v.vec ) ); }
    INLINE simd2_sse2   operator^  ( const simd2_sse2& v )  const { return( _mm_xor_si128(  vec, v.vec ) ); }
    INLINE              operator __m128i()                  const { return( vec ); }
};             

template<> INLINE int           SimdWidth<       simd2_sse2 >()                                                                     { return( 2 ); }
template<> INLINE bool          SimdSupported<   simd2_sse2 >()                                                                     { return( true ); }//(CpuIdEx( 1, 3 ) & (1 << 26)) != 0 ); }
template<> INLINE simd2_sse2    MaskAllClear<    simd2_sse2 >()                                                                     { return( _mm_setzero_si128() ); } 
template<> INLINE simd2_sse2    MaskAllSet<      simd2_sse2 >()                                                                     { return( _mm_set1_epi8( ~0 ) ); } 
template<> INLINE simd2_sse2    CountBits<       simd2_sse2 >( const simd2_sse2& val )                                              { return( _mm_popcnt_epi64_sse2( val.vec ) ); }
template<> INLINE simd2_sse2    ByteSwap<        simd2_sse2 >( const simd2_sse2& val )                                              { return( _mm_bswap_epi64_sse2( val.vec ) ); }
template<> INLINE simd2_sse2    MulLow32<        simd2_sse2 >( const simd2_sse2& val,  u32 scale )                                  { return( _mm_mul_epu32( val.vec, _mm_set1_epi64x( scale ) ) ); }
template<> INLINE simd2_sse2    MaskOut<         simd2_sse2 >( const simd2_sse2& val,  const simd2_sse2& bitsToClear )              { return( _mm_andnot_si128( bitsToClear.vec, val.vec ) ); }
template<> INLINE simd2_sse2    CmpEqual<        simd2_sse2 >( const simd2_sse2& a,    const simd2_sse2& b )                        { return( _mm_cmpeq_epi64_sse2( a.vec, b.vec ) ); }
template<> INLINE simd2_sse2    SelectIfZero<    simd2_sse2 >( const simd2_sse2& val,  const simd2_sse2& a )                        { return( _mm_and_si128( a.vec, _mm_cmpeq_epi64_sse2( val.vec, _mm_setzero_si128() ) ) ); }
template<> INLINE simd2_sse2    SelectIfZero<    simd2_sse2 >( const simd2_sse2& val,  const simd2_sse2& a, const simd2_sse2& b )   { return( _mm_select( b.vec, a.vec, _mm_cmpeq_epi64_sse2( val.vec, _mm_setzero_si128() ) ) ); }
template<> INLINE simd2_sse2    SelectIfNotZero< simd2_sse2 >( const simd2_sse2& val,  const simd2_sse2& a )                        { return( _mm_andnot_si128( _mm_cmpeq_epi64_sse2( val.vec, _mm_setzero_si128() ), a.vec ) ); }
template<> INLINE simd2_sse2    SelectIfNotZero< simd2_sse2 >( const simd2_sse2& val,  const simd2_sse2& a, const simd2_sse2& b )   { return( _mm_select( a.vec, b.vec, _mm_cmpeq_epi64_sse2( val.vec, _mm_setzero_si128() ) ) ); }
template<> INLINE simd2_sse2    SelectWithMask<  simd2_sse2 >( const simd2_sse2& mask, const simd2_sse2& a, const simd2_sse2& b )   { return( _mm_select( b.vec, a.vec, mask.vec ) ); }


// SSE4

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

struct simd2_sse4 : public simd2_sse2 
{
    INLINE simd2_sse4( const __m128i& v ) : simd2_sse2( v ) {}
};

template<> INLINE int           SimdWidth<       simd2_sse4 >()                                                                     { return( 2 ); }
template<> INLINE bool          SimdSupported<   simd2_sse4 >()                                                                     { return( true ); }//(CpuIdEx( 1, 3 ) & (1 << 26)) != 0 ); }
template<> INLINE simd2_sse4    CountBits<       simd2_sse4 >( const simd2_sse4& val )                                              { return( _mm_popcnt_epi64_sse4( val.vec ) ); }
template<> INLINE simd2_sse4    ByteSwap<        simd2_sse4 >( const simd2_sse4& val )                                              { return( _mm_bswap_epi64_sse4( val.vec ) ); }
template<> INLINE simd2_sse4    CmpEqual<        simd2_sse4 >( const simd2_sse4& a,    const simd2_sse4& b )                        { return( _mm_cmpeq_epi64( a.vec, b.vec ) ); }
template<> INLINE simd2_sse4    SelectIfZero<    simd2_sse4 >( const simd2_sse4& val,  const simd2_sse4& a )                        { return( _mm_and_si128( a.vec, _mm_cmpeq_epi64( val.vec, _mm_setzero_si128() ) ) ); }
template<> INLINE simd2_sse4    SelectIfZero<    simd2_sse4 >( const simd2_sse4& val,  const simd2_sse4& a, const simd2_sse4& b )   { return( _mm_select( b.vec, a.vec, _mm_cmpeq_epi64( val.vec, _mm_setzero_si128() ) ) ); }
template<> INLINE simd2_sse4    SelectIfNotZero< simd2_sse4 >( const simd2_sse4& val,  const simd2_sse4& a )                        { return( _mm_andnot_si128( _mm_cmpeq_epi64( val.vec, _mm_setzero_si128() ), a.vec ) ); }
template<> INLINE simd2_sse4    SelectIfNotZero< simd2_sse4 >( const simd2_sse4& val,  const simd2_sse4& a, const simd2_sse4& b )   { return( _mm_select( a.vec, b.vec, _mm_cmpeq_epi64( val.vec, _mm_setzero_si128() ) ) ); }


// AVX2

INLINE __m256i _mm256_popcnt_epi64_avx2( const __m256i& v )
{
    return( _mm256_setzero_si256() ); // FIXME
}

INLINE __m256i _mm256_bswap_epi64_avx2( const __m256i& v )
{
    return( _mm256_setzero_si256() ); // FIXME
}

INLINE __m256i _mm256_select( const __m256i& a, const __m256i& b, const __m256i& mask )
{          
    return _mm256_blendv_epi8( a, b, mask ); // mask? b : a
}


struct simd4_avx2
{
    __m256i vec;

    INLINE simd4_avx2() {}
    INLINE simd4_avx2( u64 s )                               : vec( _mm256_set1_epi64x( s ) ) {}
    INLINE simd4_avx2( const __m256i& v )                    : vec( v ) {}
    INLINE simd4_avx2( const simd4_avx2& v )                 : vec( v.vec ) {}

    INLINE simd4_avx2    operator~  ()                       const { return( _mm256_xor_si256(  vec, _mm256_set1_epi8(  ~0 ) ) ); }
    INLINE simd4_avx2    operator-  ( u64 s )                const { return( _mm256_sub_epi64(  vec, _mm256_set1_epi64x( s ) ) ); }
    INLINE simd4_avx2    operator+  ( u64 s )                const { return( _mm256_add_epi64(  vec, _mm256_set1_epi64x( s ) ) ); }
    INLINE simd4_avx2    operator&  ( u64 s )                const { return( _mm256_and_si256(  vec, _mm256_set1_epi64x( s ) ) ); }
    INLINE simd4_avx2    operator|  ( u64 s )                const { return( _mm256_or_si256(   vec, _mm256_set1_epi64x( s ) ) ); }
    INLINE simd4_avx2    operator^  ( u64 s )                const { return( _mm256_xor_si256(  vec, _mm256_set1_epi64x( s ) ) ); }
    INLINE simd4_avx2    operator<< ( int c )                const { return( _mm256_slli_epi64( vec, c ) ); }
    INLINE simd4_avx2    operator>> ( int c )                const { return( _mm256_srli_epi64( vec, c ) ); }
    INLINE simd4_avx2    operator-  ( const simd4_avx2& v )  const { return( _mm256_sub_epi64(  vec, v.vec ) ); }
    INLINE simd4_avx2    operator+  ( const simd4_avx2& v )  const { return( _mm256_add_epi64(  vec, v.vec ) ); }
    INLINE simd4_avx2    operator&  ( const simd4_avx2& v )  const { return( _mm256_and_si256(  vec, v.vec ) ); }
    INLINE simd4_avx2    operator|  ( const simd4_avx2& v )  const { return( _mm256_or_si256(   vec, v.vec ) ); }
    INLINE simd4_avx2    operator^  ( const simd4_avx2& v )  const { return( _mm256_xor_si256(  vec, v.vec ) ); }
    INLINE               operator   __m256i()                const { return( vec ); }
};               
            

template<> INLINE int            SimdWidth<       simd4_avx2 >()                                                                    { return( 4 ); }
template<> INLINE bool           SimdSupported<   simd4_avx2 >()                                                                    { return( true ); }//(CpuIdEx( 7, 1 ) & (1 << 5)) != 0 ); }
template<> INLINE simd4_avx2     MaskAllClear<    simd4_avx2 >()                                                                    { return( _mm256_setzero_si256() ); } 
template<> INLINE simd4_avx2     MaskAllSet<      simd4_avx2 >()                                                                    { return( _mm256_set1_epi8( ~0 ) ); } 
template<> INLINE simd4_avx2     CountBits<       simd4_avx2 >( const simd4_avx2& val )                                             { return( _mm256_popcnt_epi64_avx2( val.vec ) ); }
template<> INLINE simd4_avx2     ByteSwap<        simd4_avx2 >( const simd4_avx2& val )                                             { return( _mm256_bswap_epi64_avx2( val.vec ) ); }
template<> INLINE simd4_avx2     MulLow32<        simd4_avx2 >( const simd4_avx2& val,  u32 scale )                                 { return( _mm256_mul_epu32( val.vec, _mm256_set1_epi64x( scale ) ) ); }
template<> INLINE simd4_avx2     MaskOut<         simd4_avx2 >( const simd4_avx2& val,  const simd4_avx2& bitsToClear )             { return( _mm256_andnot_si256( bitsToClear.vec, val.vec ) ); }
template<> INLINE simd4_avx2     CmpEqual<        simd4_avx2 >( const simd4_avx2& a,    const simd4_avx2& b )                       { return( _mm256_cmpeq_epi64( a.vec, b.vec ) ); }
template<> INLINE simd4_avx2     SelectIfZero<    simd4_avx2 >( const simd4_avx2& val,  const simd4_avx2& a )                       { return( _mm256_and_si256( a.vec, _mm256_cmpeq_epi64( val.vec, _mm256_setzero_si256() ) ) ); }
template<> INLINE simd4_avx2     SelectIfZero<    simd4_avx2 >( const simd4_avx2& val,  const simd4_avx2& a, const simd4_avx2& b )	{ return( _mm256_select( b.vec, a.vec, _mm256_cmpeq_epi64( val.vec, _mm256_setzero_si256() ) ) ); }
template<> INLINE simd4_avx2     SelectIfNotZero< simd4_avx2 >( const simd4_avx2& val,  const simd4_avx2& a )                       { return( _mm256_andnot_si256( _mm256_cmpeq_epi64( val.vec, _mm256_setzero_si256() ), a.vec ) ); }
template<> INLINE simd4_avx2     SelectIfNotZero< simd4_avx2 >( const simd4_avx2& val,  const simd4_avx2& a, const simd4_avx2& b )  { return( _mm256_select( a.vec, b.vec, _mm256_cmpeq_epi64( val.vec, _mm256_setzero_si256() ) ) ); }
template<> INLINE simd4_avx2     SelectWithMask<  simd4_avx2 >( const simd4_avx2& mask, const simd4_avx2& a, const simd4_avx2& b )  { return( _mm256_select( b.vec, a.vec, mask.vec ) ); }


// AVX3 

#if PIGEON_ENABLE_AVX3

struct simd8_avx3
{
    __m512i vec;

    INLINE simd8_avx3() {}
    INLINE simd8_avx3( u64 s )                               : vec( _mm512_set1_epi64x( s ) ) {}
    INLINE simd8_avx3( const __m512i& v )                    : vec( v ) {}
    INLINE simd8_avx3( const simd8_avx3& v )                 : vec( v.vec ) {}

    INLINE simd8_avx3    operator~  ()                       const { return( _mm512_xor_si512(  vec, _mm512_set1_epi8(  ~0 ) ) ); }
    INLINE simd8_avx3    operator-  ( u64 s )                const { return( _mm512_sub_epi64(  vec, _mm512_set1_epi64x( s ) ) ); }
    INLINE simd8_avx3    operator+  ( u64 s )                const { return( _mm512_add_epi64(  vec, _mm512_set1_epi64x( s ) ) ); }
    INLINE simd8_avx3    operator&  ( u64 s )                const { return( _mm512_and_si512(  vec, _mm512_set1_epi64x( s ) ) ); }
    INLINE simd8_avx3    operator|  ( u64 s )                const { return( _mm512_or_si512(   vec, _mm512_set1_epi64x( s ) ) ); }
    INLINE simd8_avx3    operator^  ( u64 s )                const { return( _mm512_xor_si512(  vec, _mm512_set1_epi64x( s ) ) ); }
    INLINE simd8_avx3    operator<< ( int c )                const { return( _mm512_slli_epi64( vec, c ) ); }
    INLINE simd8_avx3    operator>> ( int c )                const { return( _mm512_srli_epi64( vec, c ) ); }
    INLINE simd8_avx3    operator-  ( const simd8_avx3& v )  const { return( _mm512_sub_epi64(  vec, v.vec ) ); }
    INLINE simd8_avx3    operator+  ( const simd8_avx3& v )  const { return( _mm512_add_epi64(  vec, v.vec ) ); }
    INLINE simd8_avx3    operator&  ( const simd8_avx3& v )  const { return( _mm512_and_si512(  vec, v.vec ) ); }
    INLINE simd8_avx3    operator|  ( const simd8_avx3& v )  const { return( _mm512_or_si512(   vec, v.vec ) ); }
    INLINE simd8_avx3    operator^  ( const simd8_avx3& v )  const { return( _mm512_xor_si512(  vec, v.vec ) ); }
    INLINE explicit      operator __m512i()                  const { return( vec ); }
};                                                                                                  

template<> INLINE int            SimdWidth<       simd8_avx3 >()                                                                    { return( 8 ); }
template<> INLINE bool           SimdSupported<   simd8_avx3 >()                                                                    { return( true ); }//(CpuIdEx( 7, 1 ) & (1 << 16)) != 0 ); }
template<> INLINE simd8_avx3     MaskAllClear<    simd8_avx3 >()                                                                    { return( _mm512_setzero_si512() ); } 
template<> INLINE simd8_avx3     MaskAllSet<      simd8_avx3 >()                                                                    { return( _mm512_set1_epi8( ~0 ) ); } 
template<> INLINE simd8_avx3     CountBits<       simd8_avx3 >( const simd8_avx3& val )                                             { return( _mm512_popcnt_epi64_avx3( val.vec ) ); }
template<> INLINE simd8_avx3     ByteSwap<        simd8_avx3 >( const simd8_avx3& val )                                             { return( _mm512_bswap_epi64_avx3( val.vec ) ); }
template<> INLINE simd8_avx3     MulLow32<        simd8_avx3 >( const simd8_avx3& val,  u32 scale )                                 { return( _mm512_mul_epu32( val, _mm512_set1_epi64x( scale ) ) ); }
template<> INLINE simd8_avx3     MaskOut<         simd8_avx3 >( const simd8_avx3& val,  const simd8_avx3& bitsToClear )             { return( _mm512_andnot_si512( bitsToClear.vec, val.vec ) ); }
template<> INLINE simd8_avx3     CmpEqual<        simd8_avx3 >( const simd8_avx3& a,    const simd8_avx3& b )                       { return( _mm512_cmpeq_epi64( a.vec, b.vec ) ); }
template<> INLINE simd8_avx3     SelectIfZero<    simd8_avx3 >( const simd8_avx3& val,  const simd8_avx3& a )                       { return( _mm512_and_si512( a.vec, _mm512_cmpeq_epi64( val.vec, _mm512_setzero_si512() ) ) ); }
template<> INLINE simd8_avx3     SelectIfZero<    simd8_avx3 >( const simd8_avx3& val,  const simd8_avx3& a, const simd8_avx3& b )	{ return( _mm512_select( b.vec, a.vec, _mm512_cmpeq_epi64( val.vec, _mm512_setzero_si512() ) ) ); }
template<> INLINE simd8_avx3     SelectIfNotZero< simd8_avx3 >( const simd8_avx3& val,  const simd8_avx3& a )                       { return( _mm512_andnot_si512( _mm512_cmpeq_epi64( val.vec, _mm512_setzero_si512() ), a.vec ) ); }
template<> INLINE simd8_avx3     SelectIfNotZero< simd8_avx3 >( const simd8_avx3& val,  const simd8_avx3& a, const simd8_avx3& b )  { return( _mm512_select( a.vec, b.vec, _mm512_cmpeq_epi64( val.vec, _mm512_setzero_si512() ) ) ); }
template<> INLINE simd8_avx3     SelectWithMask<  simd8_avx3 >( const simd8_avx3& mask, const simd8_avx3& a, const simd8_avx3& b )  { return( _mm512_select( b, a, mask ) ); }

#endif // PIGEON_ENABLE_AVX3

#endif // PIGEON_SIMD_H__
};
