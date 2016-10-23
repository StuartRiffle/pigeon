// movelist.h - PIGEON CHESS ENGINE (c) 2012-2016 Stuart Riffle
        
#ifndef PIGEON_MOVELIST_H__
#define PIGEON_MOVELIST_H__
namespace Pigeon {


/// The parameters for one move of the game
//       
template< typename T >
struct MoveSpecT
{
    T   mSrc;
    T   mDest;
    T   mType;
    T   mFlags;

    INLINE PDECL MoveSpecT() {}
    INLINE PDECL MoveSpecT( const T& _src, const T& _dest, const T& _type = MOVE ) : mSrc(  _src ), mDest(  _dest ), mType(  _type ), mFlags(  0 ) {}
    INLINE PDECL void Set(  const T& _src, const T& _dest, const T& _type = MOVE ) { mSrc = _src;   mDest = _dest;   mType = _type;   mFlags = 0; }

    template< typename U > INLINE PDECL MoveSpecT( const MoveSpecT< U >& rhs ) : mSrc( rhs.mSrc ), mDest( rhs.mDest ), mType( rhs.mType ) {}

    INLINE PDECL int  IsCapture() const       { return( ((mType >= CAPTURE_LOSING) && (mType <= CAPTURE_WINNING)) || ((mType >= CAPTURE_PROMOTE_KNIGHT) && (mType <= CAPTURE_PROMOTE_QUEEN)) ); }
    INLINE PDECL int  IsPromotion() const     { return( (mType >= PROMOTE_KNIGHT) && (mType <= CAPTURE_PROMOTE_QUEEN) ); }
    INLINE PDECL int  IsSpecial() const       { return( mFlags != 0 ); }
    INLINE PDECL void Flip()                  { mSrc = FlipSquareIndex( mSrc ); mDest = FlipSquareIndex( mDest ); }
    INLINE PDECL char GetPromoteChar() const  { return( "\0\0\0\0nbrqnbrq\0\0"[mType] ); }

    INLINE PDECL bool operator==( const MoveSpecT& rhs ) const { return( (mSrc == rhs.mSrc) && (mDest == rhs.mDest) && (mType == rhs.mType) ); }
    INLINE PDECL bool operator!=( const MoveSpecT& rhs ) const { return( (mSrc != rhs.mSrc) || (mDest != rhs.mDest) || (mType != rhs.mType) ); }
};


/// A list of valid moves
//
struct MoveList
{
    int         mCount;
    int         mTried;
    MoveSpec    mMove[MAX_MOVE_LIST];

    INLINE PDECL      MoveList()                      { this->Clear(); }
    INLINE PDECL void FlipAll()                       { for( int i = 0; i < mCount; i++ ) mMove[i].Flip(); }
    INLINE PDECL void Clear()                         { mCount = 0; mTried = 0; }
    INLINE PDECL void Append( const MoveSpec& spec )  { mMove[mCount++] = spec; }

    PDECL void Append( const MoveList& other )
    {
        MoveSpec*       RESTRICT dest = mMove + mCount;
        const MoveSpec* RESTRICT src  = other.mMove;

        for( int i = 0; i < other.mCount; i++ )
            dest[i] = src[i];

        mCount += other.mCount;
    }

    PDECL int LookupMove( const MoveSpec& spec )
    {
        for( int idx = 0; idx < mCount; idx++ )
        {
            if( (mMove[idx].mSrc != spec.mSrc) || (mMove[idx].mDest != spec.mDest) )
                continue;

            if( mMove[idx].GetPromoteChar() != spec.GetPromoteChar() )
                continue;

            return( idx );
        }

        return( -1 );
    }

    PDECL int ChooseBestUntried()
    {
        int best = mTried;

        for( int idx = best + 1; idx < mCount; idx++ )
            if( mMove[idx].mType > mMove[best].mType )
                best = idx;

        Exchange( mMove[mTried], mMove[best] );
        return( mTried++ );
    }

    PDECL void DiscardMovesBelow( int type )
    {
        int prevCount = mCount;

        for( mCount = 0; mCount < prevCount; mCount++ )
            if( mMove[mCount].mType < type )
                break;

        for( int idx = mCount + 1; idx < prevCount; idx++ )
            if( mMove[idx].mType >= type )
                mMove[mCount++] = mMove[idx];

        mTried = 0;
    }

    PDECL void DiscardQuietMoves()
    {
        this->DiscardMovesBelow( CAPTURE_EQUAL );
    }

    PDECL int MarkSpecialMoves( int src, int dest, int flag )
    {
        int marked = 0;

        for( int idx = 0; idx < mCount; idx++ )
            if( (mMove[idx].mSrc == src) && (mMove[idx].mDest == dest) )
                mMove[idx].mFlags |= (marked++, flag);

        return( marked );
    }

    PDECL void PrioritizeDest( u64 mask )
    {
        int selected = 0;

        while( (selected < mCount) && (SquareBit( mMove[selected].mDest ) & mask) )
            selected++;

        for( int idx = selected + 1; idx < mCount; idx++ )
            if( SquareBit( mMove[idx].mDest ) & mask )
                Exchange( mMove[idx], mMove[selected++] ); 
    }

    PDECL void UnpackMoveMap( const Position& pos, const MoveMap& mmap )
    {
        this->Clear();

        u64 whitePieces = pos.mWhitePawns | pos.mWhiteKnights | pos.mWhiteBishops | pos.mWhiteRooks | pos.mWhiteQueens | pos.mWhiteKing;

        if( mmap.mPawnMovesN )      this->StorePawnMoves( pos, mmap.mPawnMovesN,     SHIFT_N            );
        if( mmap.mPawnDoublesN )    this->StorePawnMoves( pos, mmap.mPawnDoublesN,   SHIFT_N * 2        );
        if( mmap.mPawnAttacksNE )   this->StorePawnMoves( pos, mmap.mPawnAttacksNE,  SHIFT_NE           );
        if( mmap.mPawnAttacksNW )   this->StorePawnMoves( pos, mmap.mPawnAttacksNW,  SHIFT_NW           );
        if( mmap.mCastlingMoves )   this->StoreKingMoves( pos, mmap.mCastlingMoves,  pos.mWhiteKing     );
        if( mmap.mKingMoves )       this->StoreKingMoves( pos, mmap.mKingMoves,      pos.mWhiteKing     );

        if( mmap.mSlidingMovesNW )  this->StoreSlidingMoves< SHIFT_NW >( pos, mmap.mSlidingMovesNW, whitePieces, mmap.mCheckMask );
        if( mmap.mSlidingMovesNE )  this->StoreSlidingMoves< SHIFT_NE >( pos, mmap.mSlidingMovesNE, whitePieces, mmap.mCheckMask );
        if( mmap.mSlidingMovesSW )  this->StoreSlidingMoves< SHIFT_SW >( pos, mmap.mSlidingMovesSW, whitePieces, mmap.mCheckMask );
        if( mmap.mSlidingMovesSE )  this->StoreSlidingMoves< SHIFT_SE >( pos, mmap.mSlidingMovesSE, whitePieces, mmap.mCheckMask );
        if( mmap.mSlidingMovesN  )  this->StoreSlidingMoves< SHIFT_N  >( pos, mmap.mSlidingMovesN,  whitePieces, mmap.mCheckMask );
        if( mmap.mSlidingMovesW  )  this->StoreSlidingMoves< SHIFT_W  >( pos, mmap.mSlidingMovesW,  whitePieces, mmap.mCheckMask );
        if( mmap.mSlidingMovesE  )  this->StoreSlidingMoves< SHIFT_E  >( pos, mmap.mSlidingMovesE,  whitePieces, mmap.mCheckMask );
        if( mmap.mSlidingMovesS  )  this->StoreSlidingMoves< SHIFT_S  >( pos, mmap.mSlidingMovesS,  whitePieces, mmap.mCheckMask );

        if( mmap.mKnightMovesNNW )  this->StoreStepMoves( pos, mmap.mKnightMovesNNW, SHIFT_N + SHIFT_NW );
        if( mmap.mKnightMovesNNE )  this->StoreStepMoves( pos, mmap.mKnightMovesNNE, SHIFT_N + SHIFT_NE );
        if( mmap.mKnightMovesWNW )  this->StoreStepMoves( pos, mmap.mKnightMovesWNW, SHIFT_W + SHIFT_NW );
        if( mmap.mKnightMovesENE )  this->StoreStepMoves( pos, mmap.mKnightMovesENE, SHIFT_E + SHIFT_NE );
        if( mmap.mKnightMovesWSW )  this->StoreStepMoves( pos, mmap.mKnightMovesWSW, SHIFT_W + SHIFT_SW );
        if( mmap.mKnightMovesESE )  this->StoreStepMoves( pos, mmap.mKnightMovesESE, SHIFT_E + SHIFT_SE );
        if( mmap.mKnightMovesSSW )  this->StoreStepMoves( pos, mmap.mKnightMovesSSW, SHIFT_S + SHIFT_SW );
        if( mmap.mKnightMovesSSE )  this->StoreStepMoves( pos, mmap.mKnightMovesSSE, SHIFT_S + SHIFT_SE );

        this->PrioritizeDest( ~mmap.mBlackControl );

        if( pos.mBoardFlipped )
            this->FlipAll();
    }

    PDECL int FindMoves( const Position& pos )
    {
        MoveMap mmap;

        this->Clear();
        pos.CalcMoveMap( &mmap );
        this->UnpackMoveMap( pos, mmap );

        return( this->mCount );
    }

private:
    INLINE PDECL void ClassifyAndStoreMove( const Position& pos, int srcIdx, int destIdx, int promote = 0 ) 
    {
        u64 src         = SquareBit( (u64) srcIdx );
        u64 dest        = SquareBit( (u64) destIdx );
        int src_val     = (src  & pos.mWhitePawns)? 1 : ((src  & (pos.mWhiteKnights | pos.mWhiteBishops))? 3 : ((src  & pos.mWhiteRooks)? 5 : ((src  & pos.mWhiteQueens)? 9 : 20)));
        int dest_val    = (dest & pos.mBlackPawns)? 1 : ((dest & (pos.mBlackKnights | pos.mBlackBishops))? 3 : ((dest & pos.mBlackRooks)? 5 : ((dest & pos.mBlackQueens)? 9 :  0)));
        int relative    = SignOrZero( dest_val - src_val );
        int capture     = dest_val? (relative + 2) : 0;
        int type        = promote? (promote + (capture? 4 : 0)) : capture;

        mMove[mCount++].Set( srcIdx, destIdx, type );
    }

    PDECL void StorePromotions( const Position& pos, u64 dest, int ofs ) 
    {
        while( dest )
        {
            int idx = (int) ConsumeLowestBitIndex( dest );

            this->ClassifyAndStoreMove( pos, idx - ofs, idx, PROMOTE_QUEEN  );
            this->ClassifyAndStoreMove( pos, idx - ofs, idx, PROMOTE_KNIGHT );
        }
    }

    INLINE PDECL void StoreStepMoves( const Position& pos, u64 dest, int ofs ) 
    {
        while( dest )
        {
            int idx = (int) ConsumeLowestBitIndex( dest );
            this->ClassifyAndStoreMove( pos, idx - ofs, idx );
        }
    }

    template< int SHIFT >
    INLINE PDECL void StoreSlidingMoves( const Position& pos, u64 dest, u64 pieces, u64 checkMask ) 
    {
        u64 src = Shift< -SHIFT >( dest ) & pieces;
        u64 cur = Shift<  SHIFT >( src );
        int ofs = SHIFT;

        while( cur )
        {
            this->StoreStepMoves( pos, cur & checkMask, ofs );
            cur = Shift< SHIFT >( cur ) & dest;
            ofs += SHIFT;
        }
    }

    PDECL void StorePawnMoves( const Position& pos, u64 dest, int ofs ) 
    {
        this->StoreStepMoves(  pos, dest & ~RANK_8, ofs );
        this->StorePromotions( pos, dest &  RANK_8, ofs );
    }

    PDECL void StoreKingMoves( const Position& pos, u64 dest, u64 king ) 
    {
        int kingIdx = (int) LowestBitIndex( king );

        do
        {
            int idx = (int) ConsumeLowestBitIndex( dest );
            this->ClassifyAndStoreMove( pos, kingIdx, idx );
        }
        while( dest );
    }
};

};     // namespace Pigeon
#endif // PIGEON_MOVELIST_H__
