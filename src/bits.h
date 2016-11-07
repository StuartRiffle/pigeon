// bits.h - PIGEON CHESS ENGINE (c) 2012-2016 Stuart Riffle

namespace Pigeon {
#ifndef PIGEON_BITS_H__
#define PIGEON_BITS_H__


template< int SHIFT, typename T >
INLINE PDECL T Shift( const T& bits )
{
    if( SHIFT > 0 )
        return( bits << (SHIFT) );
    else 
        return( bits >> ((-SHIFT)) );
}

template< int SHIFT, typename T >
INLINE PDECL T Rotate( const T& bits )
{
    return( Shift< SHIFT >( bits ) | Shift< SHIFT - 64 >( bits ) );
}

template< int SHIFT, typename T >
INLINE PDECL T Propagate( const T& bits, const T& allow )
{
    T v     = bits;
    T mask  = allow;

    v    |= Shift< SHIFT     >( v ) & mask;
    mask &= Shift< SHIFT     >( mask );
    v    |= Shift< SHIFT * 2 >( v ) & mask;
    mask &= Shift< SHIFT * 2 >( mask );
    v    |= Shift< SHIFT * 4 >( v ) & mask;

    return( v );
}

template< typename T >
INLINE PDECL T StepKnights( const T& val, const T& allow = ALL_SQUARES )
{
    T a = Shift< SHIFT_W >( val & ~FILE_A ) | Shift< SHIFT_E >( val & ~FILE_H );                                  //  . C . C .
    T b = Shift< SHIFT_W * 2 >( val & ~(FILE_A | FILE_B) ) | Shift< SHIFT_E * 2 >( val & ~(FILE_G | FILE_H) );    //  D . . . D
    T c = Shift< SHIFT_N * 2 >( a ) | Shift< SHIFT_S * 2 >( a );                                                  //  b a(N)a b
    T d = Shift< SHIFT_N >( b ) | Shift< SHIFT_S >( b );                                                          //  D . . . D
    return( (c | d) & allow );                                                                                    //  . C . C .
}

template< typename T > INLINE PDECL   T   StepN(         const T& val, const T& allow = ALL_SQUARES )         { return( Shift< SHIFT_N  >( val ) & allow ); }
template< typename T > INLINE PDECL   T   StepNW(        const T& val, const T& allow = ALL_SQUARES )         { return( Shift< SHIFT_NW >( val ) & allow & ~FILE_H ); }
template< typename T > INLINE PDECL   T   StepW(         const T& val, const T& allow = ALL_SQUARES )         { return( Shift< SHIFT_W  >( val ) & allow & ~FILE_H ); }
template< typename T > INLINE PDECL   T   StepSW(        const T& val, const T& allow = ALL_SQUARES )         { return( Shift< SHIFT_SW >( val ) & allow & ~FILE_H ); }
template< typename T > INLINE PDECL   T   StepS(         const T& val, const T& allow = ALL_SQUARES )         { return( Shift< SHIFT_S  >( val ) & allow ); }
template< typename T > INLINE PDECL   T   StepSE(        const T& val, const T& allow = ALL_SQUARES )         { return( Shift< SHIFT_SE >( val ) & allow & ~FILE_A ); }
template< typename T > INLINE PDECL   T   StepE(         const T& val, const T& allow = ALL_SQUARES )         { return( Shift< SHIFT_E  >( val ) & allow & ~FILE_A ); }
template< typename T > INLINE PDECL   T   StepNE(        const T& val, const T& allow = ALL_SQUARES )         { return( Shift< SHIFT_NE >( val ) & allow & ~FILE_A ); }
template< typename T > INLINE PDECL   T   StepOrtho(     const T& val, const T& allow = ALL_SQUARES )         { return( StepN(     val, allow ) | StepW(    val, allow ) | StepS(  val, allow ) | StepE ( val, allow ) ); }
template< typename T > INLINE PDECL   T   StepDiag(      const T& val, const T& allow = ALL_SQUARES )         { return( StepNW(    val, allow ) | StepSW(   val, allow ) | StepSE( val, allow ) | StepNE( val, allow ) ); }
template< typename T > INLINE PDECL   T   StepOut(       const T& val, const T& allow = ALL_SQUARES )         { return( StepOrtho( val, allow ) | StepDiag( val, allow ) ); }

template< typename T > INLINE PDECL   T   PropN(         const T& val, const T& allow = ALL_SQUARES )         { return( Propagate< SHIFT_N  >( val, allow ) ); }
template< typename T > INLINE PDECL   T   PropNW(        const T& val, const T& allow = ALL_SQUARES )         { return( Propagate< SHIFT_NW >( val, allow & ~FILE_H ) ); }
template< typename T > INLINE PDECL   T   PropW(         const T& val, const T& allow = ALL_SQUARES )         { return( Propagate< SHIFT_W  >( val, allow & ~FILE_H ) ); }
template< typename T > INLINE PDECL   T   PropSW(        const T& val, const T& allow = ALL_SQUARES )         { return( Propagate< SHIFT_SW >( val, allow & ~FILE_H ) ); }
template< typename T > INLINE PDECL   T   PropS(         const T& val, const T& allow = ALL_SQUARES )         { return( Propagate< SHIFT_S  >( val, allow ) ); }
template< typename T > INLINE PDECL   T   PropSE(        const T& val, const T& allow = ALL_SQUARES )         { return( Propagate< SHIFT_SE >( val, allow & ~FILE_A ) ); }
template< typename T > INLINE PDECL   T   PropE(         const T& val, const T& allow = ALL_SQUARES )         { return( Propagate< SHIFT_E  >( val, allow & ~FILE_A ) ); }
template< typename T > INLINE PDECL   T   PropNE(        const T& val, const T& allow = ALL_SQUARES )         { return( Propagate< SHIFT_NE >( val, allow & ~FILE_A ) ); }
template< typename T > INLINE PDECL   T   PropOrtho(     const T& val, const T& allow = ALL_SQUARES )         { return( PropN(  val, allow ) | PropW(  val, allow ) | PropS(  val, allow ) | PropE(  val, allow ) ); }
template< typename T > INLINE PDECL   T   PropDiag(      const T& val, const T& allow = ALL_SQUARES )         { return( PropNW( val, allow ) | PropSW( val, allow ) | PropSE( val, allow ) | PropNE( val, allow ) ); }

template< typename T > INLINE PDECL   T   PropExN(       const T& val, const T& allow = ALL_SQUARES )         { return( MaskOut( PropN(  val, allow ), val ) ); }
template< typename T > INLINE PDECL   T   PropExNW(      const T& val, const T& allow = ALL_SQUARES )         { return( MaskOut( PropNW( val, allow ), val ) ); }
template< typename T > INLINE PDECL   T   PropExW(       const T& val, const T& allow = ALL_SQUARES )         { return( MaskOut( PropW(  val, allow ), val ) ); }
template< typename T > INLINE PDECL   T   PropExSW(      const T& val, const T& allow = ALL_SQUARES )         { return( MaskOut( PropSW( val, allow ), val ) ); }
template< typename T > INLINE PDECL   T   PropExS(       const T& val, const T& allow = ALL_SQUARES )         { return( MaskOut( PropS(  val, allow ), val ) ); }
template< typename T > INLINE PDECL   T   PropExSE(      const T& val, const T& allow = ALL_SQUARES )         { return( MaskOut( PropSE( val, allow ), val ) ); }
template< typename T > INLINE PDECL   T   PropExE(       const T& val, const T& allow = ALL_SQUARES )         { return( MaskOut( PropE(  val, allow ), val ) ); }
template< typename T > INLINE PDECL   T   PropExNE(      const T& val, const T& allow = ALL_SQUARES )         { return( MaskOut( PropNE( val, allow ), val ) ); }
template< typename T > INLINE PDECL   T   PropExOrtho(   const T& val, const T& allow = ALL_SQUARES )         { return( PropExN(  val, allow ) | PropExW(  val, allow ) | PropExS(  val, allow ) | PropExE(  val, allow ) ); }
template< typename T > INLINE PDECL   T   PropExDiag(    const T& val, const T& allow = ALL_SQUARES )         { return( PropExNW( val, allow ) | PropExSW( val, allow ) | PropExSE( val, allow ) | PropExNE( val, allow ) ); }

template< typename T > INLINE PDECL   T   SlideIntoN(    const T& val, const T& through, const T& into )      { T acc = PropN(  val, through ); acc |= StepN(  acc, into ); return( acc ); }
template< typename T > INLINE PDECL   T   SlideIntoNW(   const T& val, const T& through, const T& into )      { T acc = PropNW( val, through ); acc |= StepNW( acc, into ); return( acc ); }
template< typename T > INLINE PDECL   T   SlideIntoW(    const T& val, const T& through, const T& into )      { T acc = PropW(  val, through ); acc |= StepW(  acc, into ); return( acc ); }
template< typename T > INLINE PDECL   T   SlideIntoSW(   const T& val, const T& through, const T& into )      { T acc = PropSW( val, through ); acc |= StepSW( acc, into ); return( acc ); }
template< typename T > INLINE PDECL   T   SlideIntoS(    const T& val, const T& through, const T& into )      { T acc = PropS(  val, through ); acc |= StepS(  acc, into ); return( acc ); }
template< typename T > INLINE PDECL   T   SlideIntoSE(   const T& val, const T& through, const T& into )      { T acc = PropSE( val, through ); acc |= StepSE( acc, into ); return( acc ); }
template< typename T > INLINE PDECL   T   SlideIntoE(    const T& val, const T& through, const T& into )      { T acc = PropE(  val, through ); acc |= StepE(  acc, into ); return( acc ); }
template< typename T > INLINE PDECL   T   SlideIntoNE(   const T& val, const T& through, const T& into )      { T acc = PropNE( val, through ); acc |= StepNE( acc, into ); return( acc ); }

template< typename T > INLINE PDECL   T   SlideIntoExN(  const T& val, const T& through, const T& into )      { T acc = PropN(  val, through ); T poke = StepN(  acc, into ); return( MaskOut( acc, val ) | poke ); }
template< typename T > INLINE PDECL   T   SlideIntoExNW( const T& val, const T& through, const T& into )      { T acc = PropNW( val, through ); T poke = StepNW( acc, into ); return( MaskOut( acc, val ) | poke ); }
template< typename T > INLINE PDECL   T   SlideIntoExW(  const T& val, const T& through, const T& into )      { T acc = PropW(  val, through ); T poke = StepW(  acc, into ); return( MaskOut( acc, val ) | poke ); }
template< typename T > INLINE PDECL   T   SlideIntoExSW( const T& val, const T& through, const T& into )      { T acc = PropSW( val, through ); T poke = StepSW( acc, into ); return( MaskOut( acc, val ) | poke ); }
template< typename T > INLINE PDECL   T   SlideIntoExS(  const T& val, const T& through, const T& into )      { T acc = PropS(  val, through ); T poke = StepS(  acc, into ); return( MaskOut( acc, val ) | poke ); }
template< typename T > INLINE PDECL   T   SlideIntoExSE( const T& val, const T& through, const T& into )      { T acc = PropSE( val, through ); T poke = StepSE( acc, into ); return( MaskOut( acc, val ) | poke ); }
template< typename T > INLINE PDECL   T   SlideIntoExE(  const T& val, const T& through, const T& into )      { T acc = PropE(  val, through ); T poke = StepE(  acc, into ); return( MaskOut( acc, val ) | poke ); }
template< typename T > INLINE PDECL   T   SlideIntoExNE( const T& val, const T& through, const T& into )      { T acc = PropNE( val, through ); T poke = StepNE( acc, into ); return( MaskOut( acc, val ) | poke ); }

template< typename T > INLINE PDECL   T   SlideIntoExOrtho( const T& val, const T& through, const T& into )   { return( SlideIntoExN(  val, through, into ) | SlideIntoExW(  val, through, into ) | SlideIntoExS(  val, through, into ) | SlideIntoExE(  val, through, into ) ); }
template< typename T > INLINE PDECL   T   SlideIntoExDiag(  const T& val, const T& through, const T& into )   { return( SlideIntoExNW( val, through, into ) | SlideIntoExSW( val, through, into ) | SlideIntoExSE( val, through, into ) | SlideIntoExNE( val, through, into ) ); }


template< typename SIMD, typename PACKED, typename UNPACKED >
INLINE PDECL void Unswizzle( const PACKED* srcStruct, UNPACKED* destStruct )
{
    const int LANES = SimdWidth< SIMD >::LANES;

    int blockSize    = (int) LANES * sizeof( SIMD );
    int blockCount   = (int) sizeof( PACKED ) / blockSize;

    const SIMD* RESTRICT    src     = (SIMD*) srcStruct;
    SIMD* RESTRICT          dest    = (SIMD*) destStruct;

    while( blockCount-- )
    {
        Transpose< SIMD >( src, 1, dest, sizeof( UNPACKED ) / sizeof( SIMD ) );

        src  += LANES;
        dest += 1;
    }
}

#endif // PIGEON_BITS_H__
};
