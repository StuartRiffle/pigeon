// cpu-neon.h - PIGEON CHESS ENGINE (c) 2012-2017 Stuart Riffle

namespace Pigeon {
#ifndef PIGEON_CPU_NEON_H__
#define PIGEON_CPU_NEON_H__

// WORK IN PROGRESS!

#if PIGEON_ENABLE_NEON

struct simd2_neon
{
    uint64x2_t vec;

    INLINE simd2_neon() {}
    INLINE simd2_neon( u64 s )                              : vec( vmovq_n_u64( s ) ) {}
    INLINE simd2_neon( const uint64x2_t& v )                : vec( v ) {}
    INLINE simd2_neon( const simd2_neon& v )                : vec( v.vec ) {}

    INLINE              operator uint64x2_t()               const { return( vec ); }
    INLINE simd2_neon   operator~  ()                       const { return( vmvnq_u64(   vec ) ); }
    INLINE simd2_neon   operator-  ( u64 s )                const { return( vsubq_u64(   vec, vmovq_n_u64( s ) ) ); }
    INLINE simd2_neon   operator+  ( u64 s )                const { return( vaddq_u64(   vec, vmovq_n_u64( s ) ) ); }
    INLINE simd2_neon   operator&  ( u64 s )                const { return( vandq_u64(   vec, vmovq_n_u64( s ) ) ); }
    INLINE simd2_neon   operator|  ( u64 s )                const { return( vorrq_u64(   vec, vmovq_n_u64( s ) ) ); }
    INLINE simd2_neon   operator^  ( u64 s )                const { return( veorq_u64(   vec, vmovq_n_u64( s ) ) ); }
    INLINE simd2_neon   operator<< ( int c )                const { return( vshlq_n_u64( vec, c ) ); }
    INLINE simd2_neon   operator>> ( int c )                const { return( vshrq_n_u64( vec, c ) ); }
    INLINE simd2_neon   operator<< ( const simd2_neon& v )  const { return( vshlq_u64(   vec, v.vec ) ); }
    INLINE simd2_neon   operator-  ( const simd2_neon& v )  const { return( vsubq_u64(   vec, v.vec ) ); }
    INLINE simd2_neon   operator+  ( const simd2_neon& v )  const { return( vaddq_u64(   vec, v.vec ) ); }
    INLINE simd2_neon   operator&  ( const simd2_neon& v )  const { return( vandq_u64(   vec, v.vec ) ); }
    INLINE simd2_neon   operator|  ( const simd2_neon& v )  const { return( vorrq_u64(   vec, v.vec ) ); }
    INLINE simd2_neon   operator^  ( const simd2_neon& v )  const { return( veorq_u64(   vec, v.vec ) ); }
    INLINE simd2_neon&  operator+= ( const simd2_neon& v )        { return( vec = vaddq_u64( vec, v.vec ), *this ); }
    INLINE simd2_neon&  operator&= ( const simd2_neon& v )        { return( vec = vandq_u64( vec, v.vec ), *this ); }
    INLINE simd2_neon&  operator|= ( const simd2_neon& v )        { return( vec = vorrq_u64( vec, v.vec ), *this ); }
    INLINE simd2_neon&  operator^= ( const simd2_neon& v )        { return( vec = veorq_u64( vec, v.vec ), *this ); }
}; 

template<>
struct SimdWidth< simd2_neon >
{
    enum { LANES = 2 };
};

INLINE uint64x2_t _neon_select( const uint64x2_t& a, const uint64x2_t& b, const uint64x2_t& mask )
{          
    return( vbslq_u64( b, a, mask ) ); // mask? b : a
}

INLINE uint64x2_t _neon_cmp64( const uint64x2_t& a, const uint64x2_t& b )
{
    uint32x4_t cmp32 = vceqq_u32( a, b );
    uint64x2_t cmp64 = vandq_u64( cmp32, vrev64q_u32( cmp32 ) );

    return( cmp64 );
}

INLINE uint64x2_t _neon_popcnt64( const uint64x2_t& v )
{
    return( vpaddlq_u32( vpaddlq_u16( vpaddlq_u8( vcntq_u8( (uint8x16_t) v ) ) ) ) );
}

INLINE uint64x2_t _neon_zero()
{
    return( vmovq_n_u8( 0 ) ); 
}
     
template< int POPCNT >
INLINE simd2_neon CountBits< POPCNT, simd2_neon >( const simd2_neon& val )                              
{ 
    return( _neon_popcnt64( val.vec ) );
}
 
template<>
INLINE simd2_neon MaskAllClear< simd2_neon >()                                                                     
{
    return( _neon_zero() ); 
} 

template<> 
INLINE simd2_neon MaskAllSet< simd2_neon >()                                                                    
{ 
    return( vmovq_n_u8( 0xFF ) ); 
} 

template<>
INLINE simd2_neon ByteSwap< simd2_neon >( const simd2_neon& val )                                              
{ 
    return( vrev64q_u8( val.vec ) );
}

template<>
INLINE simd2_neon MulSigned32< simd2_neon >( const simd2_neon& val, i32 scale )                                 
{
    return( _mm_mul_epi32( val.vec, _mm_set1_epi64x( scale ) ) );
}

template<>
INLINE simd2_neon MaskOut< simd2_neon >( const simd2_neon& val, const simd2_neon& bitsToClear )              
{
    return( vbicq_u64( val,vec, bitsToClear.vec ) );
}

template<>
INLINE simd2_neon CmpEqual< simd2_neon >( const simd2_neon& a, const simd2_neon& b )                        
{ 
    return( _neon_cmp64( a.vec, b.vec ) );
}

template<>
INLINE simd2_neon SelectIfZero< simd2_neon >( const simd2_neon& val, const simd2_neon& a )                        
{
    return( vandq_u64( a.vec, _neon_cmp64( val.vec, _neon_zero() ) ) );
}

template<>
INLINE simd2_neon SelectIfZero< simd2_neon >( const simd2_neon& val, const simd2_neon& a, const simd2_neon& b )   
{ 
    return( _neon_select( b.vec, a.vec, _neon_cmp64( val.vec, _neon_zero() ) ) );
}

template<>
INLINE simd2_neon SelectIfNotZero< simd2_neon >( const simd2_neon& val, const simd2_neon& a )                        
{ 
    return( vandq_u64( vmvnq_u64( _neon_cmp64( val.vec, _neon_zero() ) ), a.vec ) );
}

template<> 
INLINE simd2_neon SelectIfNotZero< simd2_neon >( const simd2_neon& val, const simd2_neon& a, const simd2_neon& b )   
{
    return( _neon_select( a.vec, b.vec, _neon_cmp64( val.vec, _neon_zero() ) ) ); 
}

template<> 
INLINE simd2_neon SelectWithMask< simd2_neon >( const simd2_neon& mask, const simd2_neon& a, const simd2_neon& b )   
{
    return( _neon_select( b.vec, a.vec, mask.vec ) );
}

template<>
INLINE void SimdInsert< simd2_neon >( simd2_neon& dest, u64 val, int lane )
{
    dest.vec = vsetq_lane_u64( val, dest.vec, lane );
}

template<> 
INLINE simd2_neon SubClampZero< simd2_neon >( const simd2_neon& a, const simd2_neon& b )                        
{ 
    simd2_neon diff = a - b;
    simd2_neon sign = diff & (1ULL << 63);

    return( SelectIfZero( sign, diff ) );
}

template<> 
INLINE void Transpose< simd2_neon >( const simd2_neon* src, int srcStep, simd2_neon* dest, int destStep )
{
    const simd2_neon* RESTRICT  src_r  = src;
    simd2_neon* RESTRICT        dest_r = dest;

    dest_r[0]         = _mm_unpacklo_epi64( src_r[0], src_r[srcStep] );
    dest_r[destStep]  = _mm_unpackhi_epi64( src_r[0], src_r[srcStep] );
}

template<>
INLINE simd2_neon LoadIndirect32< simd2_neon >( const i32* ptr, const simd2_neon& ofs )
{
    return( _mm_i64gather_epi32_sse2( ptr, ofs ) );
}

template<>
INLINE simd2_neon LoadIndirectMasked32< simd2_neon >( const i32* ptr, const simd2_neon& ofs, const simd2_neon& mask )
{
    return( _mm_mask_i64gather_epi32_sse2( ptr, ofs, mask ) );
}

#endif // PIGEON_ENABLE_NEON
#endif // PIGEON_CPU_NEON_H__
};
