// cpu-sse2.h - PIGEON CHESS ENGINE (c) 2012-2016 Stuart Riffle

namespace Pigeon {
#ifndef PIGEON_CPU_SSE2_H__
#define PIGEON_CPU_SSE2_H__


#if PIGEON_ENABLE_SSE2

INLINE __m128i _mm_select( const __m128i& a, const __m128i& b, const __m128i& mask )
{          
    return _mm_xor_si128( a, _mm_and_si128( mask, _mm_xor_si128( b, a ) ) ); // mask? b : a
}

inline __m128i _mm_sllv_epi64x( const __m128i& v, const __m128i& n )
{
    __m128i lowCount  = _mm_move_epi64( n );
    __m128i highCount = _mm_unpackhi_epi64( n, _mm_setzero_si128() ); 
    __m128i result    = _mm_unpacklo_epi64( _mm_sll_epi64( v, lowCount ), _mm_sll_epi64( v, highCount ) );

    return( result );
}

INLINE __m128i _mm_cmpeq_epi64_sse2( const __m128i& a, const __m128i& b )
{
    __m128i eq32 = _mm_cmpeq_epi32( a, b );
    __m128i eq64 = _mm_and_si128( eq32, _mm_shuffle_epi32( eq32, _MM_SHUFFLE( 2, 3, 0, 1 ) ) );

    return( eq64 );
}

INLINE __m128i _mm_popcnt_epi64_sse2( __m128i v )
{
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

INLINE __m128i _mm_i64gather_epi32_sse2( const i32* ptr, const __m128i& ofs )
{
    i64 PIGEON_ALIGN_SIMD qword[2];

    *((__m128i*) qword) = ofs;

    qword[0] = (i64) ptr[qword[0]];
    qword[1] = (i64) ptr[qword[1]];

    return( *((__m128i*) qword) );
}

INLINE __m128i _mm_mask_i64gather_epi32_sse2( const i32* ptr, const __m128i& ofs, const __m128i& mask )
{
    i64 PIGEON_ALIGN_SIMD qword[2];
    i64 PIGEON_ALIGN_SIMD qmask[2];

    *((__m128i*) qword) = ofs;
    *((__m128i*) qmask) = mask;

    // This is gross, but it doesn't happen often

    qword[0] = qmask[0]? (ptr[qword[0]] & qmask[0]) : 0;
    qword[1] = qmask[1]? (ptr[qword[1]] & qmask[1]) : 0;

    return( *((__m128i*) qword) );
}
                                    
struct simd2_sse2
{
    __m128i vec;

    INLINE simd2_sse2() {}
    INLINE simd2_sse2( u64 s )                              : vec( _mm_set1_epi64x( s ) ) {}
    INLINE simd2_sse2( const __m128i& v )                   : vec( v ) {}
    INLINE simd2_sse2( const simd2_sse2& v )                : vec( v.vec ) {}

    INLINE              operator __m128i()                  const { return( vec ); }
    INLINE simd2_sse2   operator~  ()                       const { return( _mm_xor_si128(   vec, _mm_set1_epi8(  ~0 ) ) ); }
    INLINE simd2_sse2   operator-  ( u64 s )                const { return( _mm_sub_epi64(   vec, _mm_set1_epi64x( s ) ) ); }
    INLINE simd2_sse2   operator+  ( u64 s )                const { return( _mm_add_epi64(   vec, _mm_set1_epi64x( s ) ) ); }
    INLINE simd2_sse2   operator&  ( u64 s )                const { return( _mm_and_si128(   vec, _mm_set1_epi64x( s ) ) ); }
    INLINE simd2_sse2   operator|  ( u64 s )                const { return( _mm_or_si128(    vec, _mm_set1_epi64x( s ) ) ); }
    INLINE simd2_sse2   operator^  ( u64 s )                const { return( _mm_xor_si128(   vec, _mm_set1_epi64x( s ) ) ); }
    INLINE simd2_sse2   operator<< ( int c )                const { return( _mm_slli_epi64(  vec, c ) ); }
    INLINE simd2_sse2   operator>> ( int c )                const { return( _mm_srli_epi64(  vec, c ) ); }
    INLINE simd2_sse2   operator<< ( const simd2_sse2& v )  const { return( _mm_sllv_epi64x( vec, v.vec ) ); }
    INLINE simd2_sse2   operator-  ( const simd2_sse2& v )  const { return( _mm_sub_epi64(   vec, v.vec ) ); }
    INLINE simd2_sse2   operator+  ( const simd2_sse2& v )  const { return( _mm_add_epi64(   vec, v.vec ) ); }
    INLINE simd2_sse2   operator&  ( const simd2_sse2& v )  const { return( _mm_and_si128(   vec, v.vec ) ); }
    INLINE simd2_sse2   operator|  ( const simd2_sse2& v )  const { return( _mm_or_si128(    vec, v.vec ) ); }
    INLINE simd2_sse2   operator^  ( const simd2_sse2& v )  const { return( _mm_xor_si128(   vec, v.vec ) ); }
    INLINE simd2_sse2&  operator+= ( const simd2_sse2& v )        { return( vec = _mm_add_epi64( vec, v.vec ), *this ); }
    INLINE simd2_sse2&  operator&= ( const simd2_sse2& v )        { return( vec = _mm_and_si128( vec, v.vec ), *this ); }
    INLINE simd2_sse2&  operator|= ( const simd2_sse2& v )        { return( vec = _mm_or_si128(  vec, v.vec ), *this ); }
    INLINE simd2_sse2&  operator^= ( const simd2_sse2& v )        { return( vec = _mm_xor_si128( vec, v.vec ), *this ); }
};             

template<> INLINE simd2_sse2    CountBits< 0, simd2_sse2 >( const simd2_sse2& val )                                                 { return( _mm_popcnt_epi64_sse2( val.vec ) ); }
template<> INLINE simd2_sse2    CountBits< 1, simd2_sse2 >( const simd2_sse2& val )                                                 { return( _mm_popcnt_epi64_sse2( val.vec ) ); }

template<> INLINE simd2_sse2    MaskAllClear<    simd2_sse2 >()                                                                     { return( _mm_setzero_si128() ); } 
template<> INLINE simd2_sse2    MaskAllSet<      simd2_sse2 >()                                                                     { return( _mm_set1_epi8( ~0 ) ); } 
template<> INLINE simd2_sse2    ByteSwap<        simd2_sse2 >( const simd2_sse2& val )                                              { return( _mm_bswap_epi64_sse2( val.vec ) ); }
template<> INLINE simd2_sse2    MulSigned32<     simd2_sse2 >( const simd2_sse2& val,  i32 scale )                                  { return( _mm_mul_epi32( val.vec, _mm_set1_epi64x( scale ) ) ); }
template<> INLINE simd2_sse2    MaskOut<         simd2_sse2 >( const simd2_sse2& val,  const simd2_sse2& bitsToClear )              { return( _mm_andnot_si128( bitsToClear.vec, val.vec ) ); }
template<> INLINE simd2_sse2    CmpEqual<        simd2_sse2 >( const simd2_sse2& a,    const simd2_sse2& b )                        { return( _mm_cmpeq_epi64_sse2( a.vec, b.vec ) ); }
template<> INLINE simd2_sse2    SelectIfZero<    simd2_sse2 >( const simd2_sse2& val,  const simd2_sse2& a )                        { return( _mm_and_si128( a.vec, _mm_cmpeq_epi64_sse2( val.vec, _mm_setzero_si128() ) ) ); }
template<> INLINE simd2_sse2    SelectIfZero<    simd2_sse2 >( const simd2_sse2& val,  const simd2_sse2& a, const simd2_sse2& b )   { return( _mm_select( b.vec, a.vec, _mm_cmpeq_epi64_sse2( val.vec, _mm_setzero_si128() ) ) ); }
template<> INLINE simd2_sse2    SelectIfNotZero< simd2_sse2 >( const simd2_sse2& val,  const simd2_sse2& a )                        { return( _mm_andnot_si128( _mm_cmpeq_epi64_sse2( val.vec, _mm_setzero_si128() ), a.vec ) ); }
template<> INLINE simd2_sse2    SelectIfNotZero< simd2_sse2 >( const simd2_sse2& val,  const simd2_sse2& a, const simd2_sse2& b )   { return( _mm_select( a.vec, b.vec, _mm_cmpeq_epi64_sse2( val.vec, _mm_setzero_si128() ) ) ); }
template<> INLINE simd2_sse2    SelectWithMask<  simd2_sse2 >( const simd2_sse2& mask, const simd2_sse2& a, const simd2_sse2& b )   { return( _mm_select( b.vec, a.vec, mask.vec ) ); }

template<>
struct SimdWidth< simd2_sse2 >
{
    enum { LANES = 2 };
};

template<>
void SimdInsert< simd2_sse2 >( simd2_sse2& dest, u64 val, int lane )
{
    u64 PIGEON_ALIGN_SIMD qword[2];

    *((simd2_sse2*) qword) = dest;
    qword[lane] = val;
    dest = *((simd2_sse2*) qword);
}

template<> 
INLINE simd2_sse2 SubClampZero< simd2_sse2 >( const simd2_sse2& a, const simd2_sse2& b )                        
{ 
    simd2_sse2 diff = a - b;
    simd2_sse2 sign = diff & (1ULL << 63);

    return( SelectIfZero( sign, diff ) );
}

template<> 
INLINE void Transpose< simd2_sse2 >( const simd2_sse2* src, int srcStep, simd2_sse2* dest, int destStep )
{
    const simd2_sse2* RESTRICT  src_r  = src;
    simd2_sse2* RESTRICT        dest_r = dest;

    dest_r[0]         = _mm_unpacklo_epi64( src_r[0], src_r[srcStep] );
    dest_r[destStep]  = _mm_unpackhi_epi64( src_r[0], src_r[srcStep] );
}

template<>
INLINE simd2_sse2 LoadIndirect32< simd2_sse2 >( const i32* ptr, const simd2_sse2& ofs )
{
    return( _mm_i64gather_epi32_sse2( ptr, ofs ) );
}

template<>
INLINE simd2_sse2 LoadIndirectMasked32< simd2_sse2 >( const i32* ptr, const simd2_sse2& ofs, const simd2_sse2& mask )
{
    return( _mm_mask_i64gather_epi32_sse2( ptr, ofs, mask ) );
}

#endif // PIGEON_ENABLE_SSE2
#endif // PIGEON_CPU_SSE2_H__
};
