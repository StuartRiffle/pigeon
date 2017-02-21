// cpu-avx3.h - PIGEON CHESS ENGINE (c) 2012-2016 Stuart Riffle

namespace Pigeon {
#ifndef PIGEON_CPU_AVX3_H__
#define PIGEON_CPU_AVX3_H__

// UNTESTED!

#if PIGEON_ENABLE_AVX3

INLINE __m512i _mm512_popcnt_epi64_avx3( const __m512i& v )
{
    static const __m512i nibbleBits = _mm512_setr_epi8( 
        0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
        0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
        0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
        0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4 );

    __m512i loNib = _mm512_shuffle_epi8( nibbleBits, _mm512_and_si512( v,                         _mm512_set1_epi8( 0x0F ) ) );
    __m512i hiNib = _mm512_shuffle_epi8( nibbleBits, _mm512_and_si512( _mm512_srli_epi16( v, 4 ), _mm512_set1_epi8( 0x0F ) ) );
    __m512i pop8  = _mm512_add_epi8( loNib, hiNib );
    __m512i pop64 = _mm512_sad_epu8( pop8, _mm512_setzero_si512() );

    return( pop64 );
}

INLINE __m512i _mm512_bswap_epi64_avx3( const __m512i& v )
{
    static const __m512i perm = _mm512_setr_epi8( 
         7,  6,  5,  4,  3,  2,  1,  0, 
        15, 14, 13, 12, 11, 10,  9,  8,
        23, 22, 21, 20, 19, 18, 17, 16,
        31, 30, 29, 28, 27, 26, 25, 24,
        39, 38, 37, 36, 35, 34, 33, 32, 
        47, 46, 45, 44, 43, 42, 41, 40,
        55, 54, 53, 52, 51, 50, 49, 48,
        63, 62, 61, 60, 59, 58, 57, 56 );

    return( _mm512_shuffle_epi8( v, perm ) );
}

INLINE __m512i _mm512_select( const __m512i& a, const __m512i& b, const __m512i& mask )
{          
    return _mm512_blendv_epi8( a, b, mask ); // mask? b : a
}

struct simd8_avx3
{
    __m512i vec;

    INLINE simd8_avx3() {}
    INLINE simd8_avx3( u64 s )                               : vec( _mm512_set1_epi64x( s ) ) {}
    INLINE simd8_avx3( const __m512i& v )                    : vec( v ) {}
    INLINE simd8_avx3( const simd8_avx3& v )                 : vec( v.vec ) {}

    INLINE explicit      operator   __m512i()                const { return( vec ); }
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
    INLINE simd8_avx3&   operator+= ( const simd8_avx3& v )        { return( vec = _mm512_add_epi64( vec, v.vec ), *this ); }
    INLINE simd8_avx3&   operator&= ( const simd8_avx3& v )        { return( vec = _mm512_and_si512( vec, v.vec ), *this ); }
    INLINE simd8_avx3&   operator|= ( const simd8_avx3& v )        { return( vec = _mm512_or_si512(  vec, v.vec ), *this ); }
    INLINE simd8_avx3&   operator^= ( const simd8_avx3& v )        { return( vec = _mm512_xor_si512( vec, v.vec ), *this ); }
};                                                                                                  

template<> INLINE simd8_avx3     MaskAllClear<    simd8_avx3 >()                                                                    { return( _mm512_setzero_si512() ); } 
template<> INLINE simd8_avx3     MaskAllSet<      simd8_avx3 >()                                                                    { return( _mm512_set1_epi8( ~0 ) ); } 
template<> INLINE simd8_avx3     CountBits< DISABLE_POPCNT, simd8_avx3 >( const simd8_avx3& val )                                   { return( _mm512_popcnt_epi64_avx3( val.vec ) ); }
template<> INLINE simd8_avx3     CountBits< ENABLE_POPCNT,  simd8_avx3 >( const simd8_avx3& val )                                   { return( _mm512_popcnt_epi64_avx3( val.vec ) ); }
template<> INLINE simd8_avx3     ByteSwap<        simd8_avx3 >( const simd8_avx3& val )                                             { return( _mm512_bswap_epi64_avx3( val.vec ) ); }
template<> INLINE simd8_avx3     MulSigned32<     simd8_avx3 >( const simd8_avx3& val,  i32 scale )                                 { return( _mm512_mul_epi32( val, _mm512_set1_epi64x( scale ) ) ); }
template<> INLINE simd8_avx3     MaskOut<         simd8_avx3 >( const simd8_avx3& val,  const simd8_avx3& bitsToClear )             { return( _mm512_andnot_si512( bitsToClear.vec, val.vec ) ); }
template<> INLINE simd8_avx3     CmpEqual<        simd8_avx3 >( const simd8_avx3& a,    const simd8_avx3& b )                       { return( _mm512_cmpeq_epi64( a.vec, b.vec ) ); }
template<> INLINE simd8_avx3     SelectIfZero<    simd8_avx3 >( const simd8_avx3& val,  const simd8_avx3& a )                       { return( _mm512_and_si512( a.vec, _mm512_cmpeq_epi64( val.vec, _mm512_setzero_si512() ) ) ); }
template<> INLINE simd8_avx3     SelectIfZero<    simd8_avx3 >( const simd8_avx3& val,  const simd8_avx3& a, const simd8_avx3& b )  { return( _mm512_select( b.vec, a.vec, _mm512_cmpeq_epi64( val.vec, _mm512_setzero_si512() ) ) ); }
template<> INLINE simd8_avx3     SelectIfNotZero< simd8_avx3 >( const simd8_avx3& val,  const simd8_avx3& a )                       { return( _mm512_andnot_si512( _mm512_cmpeq_epi64( val.vec, _mm512_setzero_si512() ), a.vec ) ); }
template<> INLINE simd8_avx3     SelectIfNotZero< simd8_avx3 >( const simd8_avx3& val,  const simd8_avx3& a, const simd8_avx3& b )  { return( _mm512_select( a.vec, b.vec, _mm512_cmpeq_epi64( val.vec, _mm512_setzero_si512() ) ) ); }
template<> INLINE simd8_avx3     SelectWithMask<  simd8_avx3 >( const simd8_avx3& mask, const simd8_avx3& a, const simd8_avx3& b )  { return( _mm512_select( b, a, mask ) ); }

template<>
struct SimdWidth< simd8_avx3 >
{
    enum { LANES = 8 };
};

template<>
void SimdInsert< simd8_avx3 >( simd8_avx3& dest, u64 val, int lane )
{
    // FIXME: do something better using insert/extract intrinsics

    u64 PIGEON_ALIGN_SIMD qword[8];

    *((simd8_avx3*) qword) = dest;
    qword[lane] = val;
    dest = *((simd8_avx3*) qword);
}

template<> 
INLINE simd8_avx3 SubClampZero< simd8_avx3 >( const simd8_avx3& a, const simd8_avx3& b )                        
{ 
    simd8_avx3 diff = a - b;
    simd8_avx3 sign = diff & (1ULL << 63);

    return( SelectIfZero( sign, diff ) );
}


template<> 
INLINE void Transpose< simd8_avx3 >( const simd8_avx3* src, int srcStep, simd8_avx3* dest, int destStep )
{
    const __m512i idx = _mm512_setr_epi64( 0, 1, 2, 3, 4, 5, 6, 7 );
    const u64* src64 = (const u64*) src;

    simd8_avx3 col0 = _mm512_i64gather_epi64( idx, src64 + 0, _MM_UPCONV_EPI64_NONE, _MM_SCALE_8, _MM_HINT_NONE ); 
    simd8_avx3 col1 = _mm512_i64gather_epi64( idx, src64 + 1, _MM_UPCONV_EPI64_NONE, _MM_SCALE_8, _MM_HINT_NONE ); 
    simd8_avx3 col2 = _mm512_i64gather_epi64( idx, src64 + 2, _MM_UPCONV_EPI64_NONE, _MM_SCALE_8, _MM_HINT_NONE ); 
    simd8_avx3 col3 = _mm512_i64gather_epi64( idx, src64 + 3, _MM_UPCONV_EPI64_NONE, _MM_SCALE_8, _MM_HINT_NONE ); 
    simd8_avx3 col4 = _mm512_i64gather_epi64( idx, src64 + 4, _MM_UPCONV_EPI64_NONE, _MM_SCALE_8, _MM_HINT_NONE ); 
    simd8_avx3 col5 = _mm512_i64gather_epi64( idx, src64 + 5, _MM_UPCONV_EPI64_NONE, _MM_SCALE_8, _MM_HINT_NONE ); 
    simd8_avx3 col6 = _mm512_i64gather_epi64( idx, src64 + 6, _MM_UPCONV_EPI64_NONE, _MM_SCALE_8, _MM_HINT_NONE ); 
    simd8_avx3 col7 = _mm512_i64gather_epi64( idx, src64 + 7, _MM_UPCONV_EPI64_NONE, _MM_SCALE_8, _MM_HINT_NONE ); 

    dest[destStep * 0] = col0;
    dest[destStep * 1] = col1;
    dest[destStep * 2] = col2;
    dest[destStep * 3] = col3;
    dest[destStep * 4] = col4;
    dest[destStep * 5] = col5;
    dest[destStep * 6] = col6;
    dest[destStep * 7] = col7;
}

template<>
INLINE simd8_avx3 LoadIndirect32< simd8_avx3 >( const i32* ptr, const simd8_avx3& ofs )
{
    __m128i dwords = _mm512_i64gather_epi32( ptr, ofs, sizeof( i32 ) );
    __m512i qwords = _mm512_cvtepi32_epi64( dwords );
    return( qwords );
}

template<>
INLINE simd8_avx3 LoadIndirectMasked32< simd8_avx3 >( const i32* ptr, const simd8_avx3& ofs, const simd8_avx3& mask )
{
    __m128i mask32 = _mm512_extracti128_si512( _mm512_unpacklo_epi32( mask, mask ), 0 );
    __m128i dwords = _mm512_mask_i64gather_epi32( _mm_setzero_si128(), ptr, ofs, mask32, sizeof( i32 ) );
    __m512i qwords = _mm512_cvtepi32_epi64( dwords );
    return( qwords );
}





#endif // PIGEON_ENABLE_AVX3
#endif // PIGEON_CPU_AVX3_H__
};
