// cpu-avx512.h - PIGEON CHESS ENGINE (c) 2012-2017 Stuart Riffle

namespace Pigeon {
#ifndef PIGEON_CPU_AVX512_H__
#define PIGEON_CPU_AVX512_H__

#if PIGEON_ENABLE_AVX512   // UNTESTED!

struct simd8_avx512
{
    __m512i vec;

    INLINE simd8_avx512() {}
    INLINE simd8_avx512( u64 s )                                    : vec( _mm512_set1_epi64( s ) ) {}
    INLINE simd8_avx512( const __m512i& v )                         : vec( v ) {}
    INLINE simd8_avx512( const simd8_avx512& v )                    : vec( v.vec ) {}

    INLINE explicit      operator   __m512i()                 const { return( vec ); }
    INLINE simd8_avx512  operator~  ()                        const { return( _mm512_xor_si512(  vec, _mm512_set1_epi8( ~0 ) ) ); }
    INLINE simd8_avx512  operator-  ( u64 s )                 const { return( _mm512_sub_epi64(  vec, _mm512_set1_epi64( s ) ) ); }
    INLINE simd8_avx512  operator+  ( u64 s )                 const { return( _mm512_add_epi64(  vec, _mm512_set1_epi64( s ) ) ); }
    INLINE simd8_avx512  operator&  ( u64 s )                 const { return( _mm512_and_si512(  vec, _mm512_set1_epi64( s ) ) ); }
    INLINE simd8_avx512  operator|  ( u64 s )                 const { return( _mm512_or_si512(   vec, _mm512_set1_epi64( s ) ) ); }
    INLINE simd8_avx512  operator^  ( u64 s )                 const { return( _mm512_xor_si512(  vec, _mm512_set1_epi64( s ) ) ); }
    INLINE simd8_avx512  operator<< ( int c )                 const { return( _mm512_slli_epi64( vec, c ) ); }
    INLINE simd8_avx512  operator>> ( int c )                 const { return( _mm512_srli_epi64( vec, c ) ); }
    INLINE simd8_avx512  operator<< ( const simd8_avx512& v ) const { return( _mm512_sllv_epi64( vec, v.vec ) ); }
    INLINE simd8_avx512  operator>> ( const simd8_avx512& v ) const { return( _mm512_srlv_epi64( vec, v.vec ) ); }
    INLINE simd8_avx512  operator-  ( const simd8_avx512& v ) const { return( _mm512_sub_epi64(  vec, v.vec ) ); }
    INLINE simd8_avx512  operator+  ( const simd8_avx512& v ) const { return( _mm512_add_epi64(  vec, v.vec ) ); }
    INLINE simd8_avx512  operator&  ( const simd8_avx512& v ) const { return( _mm512_and_si512(  vec, v.vec ) ); }
    INLINE simd8_avx512  operator|  ( const simd8_avx512& v ) const { return( _mm512_or_si512(   vec, v.vec ) ); }
    INLINE simd8_avx512  operator^  ( const simd8_avx512& v ) const { return( _mm512_xor_si512(  vec, v.vec ) ); }
    INLINE simd8_avx512& operator+= ( const simd8_avx512& v )       { return( vec = _mm512_add_epi64( vec, v.vec ), *this ); }
    INLINE simd8_avx512& operator&= ( const simd8_avx512& v )       { return( vec = _mm512_and_si512( vec, v.vec ), *this ); }
    INLINE simd8_avx512& operator|= ( const simd8_avx512& v )       { return( vec = _mm512_or_si512(  vec, v.vec ), *this ); }
    INLINE simd8_avx512& operator^= ( const simd8_avx512& v )       { return( vec = _mm512_xor_si512( vec, v.vec ), *this ); }
};       

template<>
struct SimdWidth< simd8_avx512 >
{
    enum { LANES = 8 };
};


INLINE __m512i _mm512_popcnt_epi64_avx512( const __m512i& v )
{
    __m512i mask  = _mm512_set1_epi8( 0x0F );
    __m512i shuf  = _mm512_set1_epi64( 0x4332322132212110ULL );
    __m512i lo4   = _mm512_shuffle_epi8( shuf, _mm512_and_si512( mask, v ) );
    __m512i hi4   = _mm512_shuffle_epi8( shuf, _mm512_and_si512( mask, _mm512_srli_epi16( v, 4 ) ) );
    __m512i pop8  = _mm512_add_epi8( lo4, hi4 );
    __m512i pop64 = _mm512_sad_epu8( pop8, _mm512_setzero_si512() );

    return( pop64 );
}

INLINE __m512i _mm512_select( const __m512i& a, const __m512i& b, const __m512i& mask )
{ 
    return( _mm512_ternarylogic_epi64( mask, a, b, 0xCA ) );  // mask? b : a
}

template<> 
INLINE simd8_avx512 MaskAllClear< simd8_avx512 >() 
{ 
    return( _mm512_setzero_si512() );
} 

template<>
INLINE simd8_avx512 MaskAllSet< simd8_avx512 >() 
{
    return( _mm512_set1_epi8( ~0 ) );
} 

template<>
INLINE simd8_avx512 ByteSwap< simd8_avx512 >( const simd8_avx512& val ) 
{ 
    const __m512i perm = _mm512_set_epi8( 
         7,  6,  5,  4,  3,  2,  1,  0, 
        15, 14, 13, 12, 11, 10,  9,  8,
        23, 22, 21, 20, 19, 18, 17, 16,
        31, 30, 29, 28, 27, 26, 25, 24,
        39, 38, 37, 36, 35, 34, 33, 32, 
        47, 46, 45, 44, 43, 42, 41, 40,
        55, 54, 53, 52, 51, 50, 49, 48,
        63, 62, 61, 60, 59, 58, 57, 56 );

    return( _mm512_shuffle_epi8( val.vec, perm ) );
}

template<>
INLINE simd8_avx512 MulSigned32< simd8_avx512 >( const simd8_avx512& val, i32 scale ) 
{
    return( _mm512_mul_epi32( val.vec, _mm512_set1_epi64( scale ) ) );
}

template<>
INLINE simd8_avx512 MaskOut< simd8_avx512 >( const simd8_avx512& val, const simd8_avx512& bitsToClear ) 
{
    return( _mm512_andnot_si512( bitsToClear.vec, val.vec ) );
}

template<>
INLINE simd8_avx512 CmpEqual< simd8_avx512 >( const simd8_avx512& a, const simd8_avx512& b ) 
{
    __mmask8 mask = _mm512_cmpeq_epi64_mask( a.vec, b.vec );
    return( _mm512_mask_blend_epi64( mask, _mm512_setzero_si512(), _mm512_set1_epi8( ~0 ) ) );
}

template<>
INLINE simd8_avx512 SelectIfZero< simd8_avx512 >( const simd8_avx512& val, const simd8_avx512& a ) 
{
    __mmask8 mask = _mm512_cmpeq_epi64_mask( val.vec, _mm512_setzero_si512() ); 
    return( _mm512_mask_blend_epi64( mask, _mm512_setzero_si512(), a.vec ) );
}

template<>
INLINE simd8_avx512 SelectIfZero< simd8_avx512 >( const simd8_avx512& val, const simd8_avx512& a, const simd8_avx512& b ) 
{ 
    __mmask8 mask = _mm512_cmpeq_epi64_mask( val.vec, _mm512_setzero_si512() ); 
    return( _mm512_mask_blend_epi64( mask, b.vec, a.vec ) );
}

template<> 
INLINE simd8_avx512 SelectIfNotZero< simd8_avx512 >( const simd8_avx512& val, const simd8_avx512& a ) 
{
    __mmask8 mask = _mm512_cmpneq_epi64_mask( val.vec, _mm512_setzero_si512() ); 
    return( _mm512_mask_blend_epi64( mask, _mm512_setzero_si512(), a.vec ) );
}

template<>
INLINE simd8_avx512 SelectIfNotZero< simd8_avx512 >( const simd8_avx512& val, const simd8_avx512& a, const simd8_avx512& b ) 
{
    __mmask8 mask = _mm512_cmpneq_epi64_mask( val.vec, _mm512_setzero_si512() ); 
    return( _mm512_mask_blend_epi64( mask, b.vec, a.vec ) );
}

template<>
INLINE simd8_avx512 SelectWithMask< simd8_avx512 >( const simd8_avx512& mask, const simd8_avx512& a, const simd8_avx512& b ) 
{ 
    return( _mm512_select( b.vec, a.vec, mask.vec ) );
}

template<> 
INLINE simd8_avx512 CountBits< DISABLE_POPCNT, simd8_avx512 >( const simd8_avx512& val ) 
{
    return( _mm512_popcnt_epi64_avx512( val.vec ) );
}

template<> 
INLINE simd8_avx512 CountBits< ENABLE_POPCNT, simd8_avx512 >( const simd8_avx512& val ) 
{ 
    return( _mm512_popcnt_epi64_avx512( val.vec ) );
}

template<>
void SimdInsert< simd8_avx512 >( simd8_avx512& dest, u64 val, int lane )
{
    // FIXME: do something better using insert/extract intrinsics

    u64 PIGEON_ALIGN( sizeof( simd8_avx512 ) ) qwords[8];

    *((simd8_avx512*) qwords) = dest;
    qwords[lane] = val;
    dest = *((simd8_avx512*) qwords);
}

template<> 
INLINE simd8_avx512 SubClampZero< simd8_avx512 >( const simd8_avx512& a, const simd8_avx512& b )                        
{ 
    simd8_avx512 diff = a - b;
    simd8_avx512 sign = diff & (1ULL << 63);

    return( SelectIfZero( sign, diff ) );
}


template<> 
INLINE void Transpose< simd8_avx512 >( const simd8_avx512* src, int srcStep, simd8_avx512* dest, int destStep )
{
    const __m512i idx = _mm512_setr_epi64( 0, 1, 2, 3, 4, 5, 6, 7 );
    const u64* src64 = (const u64*) src;

    simd8_avx512 col0 = _mm512_i64gather_epi64( idx, src64 + 0, _MM_SCALE_8 ); 
    simd8_avx512 col1 = _mm512_i64gather_epi64( idx, src64 + 1, _MM_SCALE_8 ); 
    simd8_avx512 col2 = _mm512_i64gather_epi64( idx, src64 + 2, _MM_SCALE_8 ); 
    simd8_avx512 col3 = _mm512_i64gather_epi64( idx, src64 + 3, _MM_SCALE_8 ); 
    simd8_avx512 col4 = _mm512_i64gather_epi64( idx, src64 + 4, _MM_SCALE_8 ); 
    simd8_avx512 col5 = _mm512_i64gather_epi64( idx, src64 + 5, _MM_SCALE_8 ); 
    simd8_avx512 col6 = _mm512_i64gather_epi64( idx, src64 + 6, _MM_SCALE_8 ); 
    simd8_avx512 col7 = _mm512_i64gather_epi64( idx, src64 + 7, _MM_SCALE_8 ); 

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
INLINE simd8_avx512 LoadIndirect32< simd8_avx512 >( const i32* ptr, const simd8_avx512& ofs )
{
    __m128i dwords = _mm512_i64gather_epi32( ptr, ofs, sizeof( i32 ) );
    __m512i qwords = _mm512_cvtepi32_epi64( dwords );
    return( qwords );
}

template<>
INLINE simd8_avx512 LoadIndirectMasked32< simd8_avx512 >( const i32* ptr, const simd8_avx512& ofs, const simd8_avx512& mask )
{
    __m128i mask32 = _mm512_extracti128_si512( _mm512_unpacklo_epi32( mask, mask ), 0 );
    __m128i dwords = _mm512_mask_i64gather_epi32( _mm_setzero_si128(), ptr, ofs, mask32, sizeof( i32 ) );
    __m512i qwords = _mm512_cvtepi32_epi64( dwords );
    return( qwords );
}


#endif // PIGEON_ENABLE_AVX512
#endif // PIGEON_CPU_AVX512_H__
};
