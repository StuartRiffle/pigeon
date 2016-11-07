// cpu-avx3.h - PIGEON CHESS ENGINE (c) 2012-2016 Stuart Riffle

namespace Pigeon {
#ifndef PIGEON_CPU_AVX3_H__
#define PIGEON_CPU_AVX3_H__

// UNTESTED!

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

template<> INLINE simd8_avx3     MaskAllClear<    simd8_avx3 >()                                                                    { return( _mm512_setzero_si512() ); } 
template<> INLINE simd8_avx3     MaskAllSet<      simd8_avx3 >()                                                                    { return( _mm512_set1_epi8( ~0 ) ); } 
template<> INLINE simd8_avx3     CountBits<       simd8_avx3 >( const simd8_avx3& val )                                             { return( _mm512_popcnt_epi64_avx3( val.vec ) ); }
template<> INLINE simd8_avx3     ByteSwap<        simd8_avx3 >( const simd8_avx3& val )                                             { return( _mm512_bswap_epi64_avx3( val.vec ) ); }
template<> INLINE simd8_avx3     MulLow32<        simd8_avx3 >( const simd8_avx3& val,  u32 scale )                                 { return( _mm512_mul_epu32( val, _mm512_set1_epi64x( scale ) ) ); }
template<> INLINE simd8_avx3     MaskOut<         simd8_avx3 >( const simd8_avx3& val,  const simd8_avx3& bitsToClear )             { return( _mm512_andnot_si512( bitsToClear.vec, val.vec ) ); }
template<> INLINE simd8_avx3     CmpEqual<        simd8_avx3 >( const simd8_avx3& a,    const simd8_avx3& b )                       { return( _mm512_cmpeq_epi64( a.vec, b.vec ) ); }
template<> INLINE simd8_avx3     SelectIfZero<    simd8_avx3 >( const simd8_avx3& val,  const simd8_avx3& a )                       { return( _mm512_and_si512( a.vec, _mm512_cmpeq_epi64( val.vec, _mm512_setzero_si512() ) ) ); }
template<> INLINE simd8_avx3     SelectIfZero<    simd8_avx3 >( const simd8_avx3& val,  const simd8_avx3& a, const simd8_avx3& b )  { return( _mm512_select( b.vec, a.vec, _mm512_cmpeq_epi64( val.vec, _mm512_setzero_si512() ) ) ); }
template<> INLINE simd8_avx3     SelectIfNotZero< simd8_avx3 >( const simd8_avx3& val,  const simd8_avx3& a )                       { return( _mm512_andnot_si512( _mm512_cmpeq_epi64( val.vec, _mm512_setzero_si512() ), a.vec ) ); }
template<> INLINE simd8_avx3     SelectIfNotZero< simd8_avx3 >( const simd8_avx3& val,  const simd8_avx3& a, const simd8_avx3& b )  { return( _mm512_select( a.vec, b.vec, _mm512_cmpeq_epi64( val.vec, _mm512_setzero_si512() ) ) ); }
template<> INLINE simd8_avx3     SelectWithMask<  simd8_avx3 >( const simd8_avx3& mask, const simd8_avx3& a, const simd8_avx3& b )  { return( _mm512_select( b, a, mask ) ); }

#endif // PIGEON_ENABLE_AVX3
#endif // PIGEON_CPU_AVX3_H__
};
