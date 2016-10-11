// negamax.h - PIGEON CHESS ENGINE (c) 2012-2016 Stuart Riffle

// WORK IN PROGRESS

#ifndef PIGEON_NEGAMAX_H__
#define PIGEON_NEGAMAX_H__
namespace Pigeon {



template< typename SIMD >
struct SearchState
{
    const int LANES = SimdWidth< SIMD >::LANES;

    enum
    {
        STEP_PREPARE,
        STEP_ITERATE_CHILDREN,
        STEP_CHECK_SCORE,
        STEP_FINALIZE,
        STEP_DONE
    };


    /// The local state for one stack frame

    struct Frame
    {
        Position        pos; 
        MoveMap         moveMap; 
        int             step;
        int             ply; 
        int             depth; 
        EvalTerm        score; 
        EvalTerm        alpha; 
        EvalTerm        beta; 
        EvalTerm        result;
        bool            onPv;
        bool            inCheck;
        int             movesTried;
        EvalTerm        bestScore;
        MoveSpec        bestMove;
        MoveList        moves;
        int             simdIdx;

        MoveSpec PIGEON_ALIGN_SIMD  childSpec[LANES];
        Position PIGEON_ALIGN_SIMD  childPos[LANES];
        MoveMap  PIGEON_ALIGN_SIMD  childMoveMap[LANES];
        EvalTerm PIGEON_ALIGN_SIMD  childScore[LANES];    
    };

    int                 mFrameIdx;
    MoveList            mBestLine;
    Frame               mFrames[MAX_SEARCH_DEPTH];


    PDECL INLINE Frame* HandleLeaf( Frame* f )
    {
        if( f->depth < 1 )
        {
            f->alpha = Max( f->alpha, f->score );

            if( f->alpha >= f->beta )
            {
                f->result = f->beta;
                f--;
            }
        }

        return( f );
    }


    PDECL INLINE Frame* HandleMate( Frame* f )
    {
        f->moves.UnpackMoveMap( f->pos, f->moveMap );
        f->inCheck = (f->moveMap.IsInCheck() != 0);

        if( f->moves.mCount == 0 )
        {
            f->result = f->inCheck? EVAL_CHECKMATE : EVAL_STALEMATE;
            f--;
        }        

        return( f );
    }


    PDECL INLINE Frame* Quieten( Frame* f )
    {
        if( f->depth < 1 )
        {
            if( !f->inCheck )
                f->moves.DiscardMovesBelow( CAPTURE_LOSING );

            if( f->moves.mCount == 0 )
            {
                f->result = f->score;
                f--;
            }
        }

        return( f );
    }


    PDECL INLINE Frame* CheckHashTable( Frame* f )
    {
        if( f->depth > 0 )
        {
            TableEntry tt;

            hashTable.Load( f->pos.mHash, tt );

            u32 verify = (u32) (f->pos.mHash >> 40);
            if( tt.mHashVerify == verify )
            {
                mMetrics.mHashHitsAtPly[f->ply]++;

                bool        samePlayer          = (f->pos.mWhiteToMove != 0) == tt.mWhiteMove;
                bool        failedHighBefore    = samePlayer? tt.mFailHigh : tt.mFailLow;
                EvalTerm    lowerBoundBefore    = samePlayer? tt.mScore    : -tt.mScore;
                int         depthBefore         = tt.mDepth;

                if( failedHighBefore && (lowerBoundBefore >= f->beta) && (depthBefore >= f->depth) )
                {
                    f->result = f->beta;
                    f--;
                }
                else
                {
                    f->moves.MarkSpecialMoves( tt.mBestSrc, tt.mBestDest, FLAG_TT_BEST_MOVE );
                }
            }
        }

        return( f );
    }


    PDECL INLINE Frame* PrepareToIterate( Frame* f )
    {
        if( f->onPv && (mBestLine.mCount > f->ply) )
        {
            MoveSpec& pvMove = mBestLine.mMove[f->ply];
            f->moves.MarkSpecialMoves( pvMove.mSrc, pvMove.mDest, FLAG_PRINCIPAL_VARIATION );
        }

        f->movesTried   = 0;
        f->bestScore    = f->alpha;
        f->simdIdx      = LANES - 1;
        f->step         = STEP_ITERATE_CHILDREN;        

        return( f );
    }


    PDECL INLINE Frame* IterateChildren( Frame* f )
    {
        if( (f->movesTried >= f->moves.mCount) || (f->bestScore >= f->beta) )
        {
            f->step = STEP_FINALIZE;
        }
        else
        {
            f->simdIdx++;
            if( f->simdIdx >= LANES )
            {
                MoveSpecT< SIMD >   simdSpec;
                PositionT< SIMD >   simdPos;
                MoveMapT< SIMD >    simdMoveMap;
                SIMD                simdScore;

                simdSpec.mSrc  = 0;
                simdSpec.mDest = 0;
                simdSpec.mType = 0;

                for( int idxLane = 0; idxLane < LANES; idxLane++ )
                {
                    if( f->moves.mTried >= f->moves.mCount )
                        break;

                    int idxMove = this->ChooseNextMove( f->moves, (int) f->pos.mWhiteToMove );
                    f->childSpec[idxLane] = f->moves.mMove[idxMove];

                    SimdInsert( simdSpec.mSrc,  f->childSpec[idxLane].mSrc,  idxLane );
                    SimdInsert( simdSpec.mDest, f->childSpec[idxLane].mDest, idxLane );
                    SimdInsert( simdSpec.mType, f->childSpec[idxLane].mType, idxLane );

                    mMetrics.mNodesTotalSimd++;
                }

                const MaterialTable* whiteMat = NULL;
                const MaterialTable* blackMat = NULL;

                if( 0 ) // FIXME
                {
                    whiteMat = &mMaterialTable[pos.mWhiteToMove];
                    blackMat = &mMaterialTable[pos.mWhiteToMove ^ 1];
                }

                simdPos.Broadcast( f->pos );
                simdPos.Step( simdSpec, whiteMat, blackMat );
                simdPos.CalcMoveMap( &simdMoveMap );
                simdScore = mEvaluator.Evaluate< POPCNT, SIMD >( simdPos, simdMoveMap, mRootWeights );

                Unswizzle< SIMD >( &simdPos,     f->childPos );
                Unswizzle< SIMD >( &simdMoveMap, f->childMoveMap );

                u64 PIGEON_ALIGN_SIMD unpackScore[LANES];
                *((SIMD*) unpackScore) = simdScore;

                for( int idxLane = 0; idxLane < LANES; idxLane++ )
                    f->childScore[idxLane] = (EvalTerm) unpackScore[idxLane];

                f->simdIdx = 0;
            }   

            Frame* n = f + 1;

            n->pos          = f->childPos[f->simdIdx];
            n->moveMap      = f->childMoveMap[f->simdIdx];
            n->score        = f->childScore[f->simdIdx];
            n->ply          = f->ply + 1; 
            n->depth        = f->depth - 1; 
            n->alpha        = -f->beta; 
            n->beta         = -f->bestScore;
            f->step         = STEP_CHECK_SCORE;
            f->movesTried++;

            f++;
        }

        return( f );        
    }


    PDECL INLINE Frame* CheckScore( Frame* f )
    {
        EvalTerm subScore = f[1].result;

        if( subScore > f->bestScore )
        {
            f->bestScore    = subScore;
            f->bestMove     = f->childSpec[f->simdIdx];

            mBestLine.Clear();
            for( int i = 0; i <= f->ply; i++ )
                mBestLine.Append( mFrames[i].bestMove );

            RegisterBestLine( pv, f->alpha, f->beta );
        }

        f->movesTried++;
        f->step = STEP_ITERATE_CHILDREN;

        return( f );
    }


    PDECL INLINE Frame* StoreIntoHashTable( Frame* f )
    {
        bool        failedHigh  = (f->bestScore >= f->beta);
        bool        failedLow   = (f->bestScore == f->alpha);
        EvalTerm    result      = failedHigh? f->beta : (failedLow? f->alpha : f->bestScore);

        if( f->depth > 0 )
        {
            TableEntry tt;

            tt.mHashVerify  = f->pos.mHash >> 40;
            tt.mDepth       = f->depth;
            tt.mScore       = result;
            tt.mBestSrc     = f->bestMove.mSrc;
            tt.mBestDest    = f->bestMove.mDest;
            tt.mFailLow     = failedLow;
            tt.mFailHigh    = failedHigh;
            tt.mWhiteMove   = f->pos.mWhiteToMove? true : false;

            hashTable.Store( f->pos.mHash, tt );
        }

        f->result = result;
        f--;

        return( f );
    }


    PDECL bool Advance()
    {
        Frame* f = mFrames + mFrameIdx;

        if( f->step == STEP_PREPARE )            
            f = this->HandleLeaf( f );

        if( f->step == STEP_PREPARE )            
            f = this->HandleMate( f );

        if( f->step == STEP_PREPARE )            
            f = this->Quieten( f );

        if( f->step == STEP_PREPARE )            
            f = this->CheckHashTable( f );

        if( f->step == STEP_PREPARE )            
            f = this->PrepareToIterate( f );

        if( f->step == STEP_ITERATE_CHILDREN )   
            f = this->IterateChildren( f );

        if( f->step == STEP_CHECK_SCORE )        
            f = this->CheckScore( f );

        if( f->step == STEP_FINALIZE )           
            f = this->StoreIntoHashTable( f );

        mFrameIdx = f - mFrames;

        return( f->step == STEP_DONE );
    }
};


};
#endif // PIGEON_NEGAMAX_H__
