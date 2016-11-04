// search.h - PIGEON CHESS ENGINE (c) 2012-2016 Stuart Riffle

#if PIGEON_ENABLE_CUDA
#include <vector>
#endif

namespace Pigeon {
#ifndef PIGEON_SEARCH_H__
#define PIGEON_SEARCH_H__


/// Parameters for a best-move search (mirrors UCI move options)

struct SearchConfig
{
    int                 mWhiteTimeLeft;   
    int                 mBlackTimeLeft;   
    int                 mWhiteTimeInc;    
    int                 mBlackTimeInc;    
    int                 mTimeControlMoves;
    int                 mMateSearchDepth; 
    int                 mDepthLimit;       
    int                 mNodesLimit;       
    int                 mTimeLimit; 
    MoveList            mLimitMoves;

    SearchConfig()      { this->Clear(); }
    void Clear()        { PlatClearMemory( this, sizeof( *this ) ); }
};


/// Diagnostic engine metrics

struct SearchMetrics
{
    u64                 mNodesTotal;
    u64                 mNodesTotalSimd;
    u64                 mGpuNodesTotal;
    //u64                 mNodesAtPly[METRICS_DEPTH];
    //u64                 mHashLookupsAtPly[METRICS_DEPTH];
    //u64                 mHashHitsAtPly[METRICS_DEPTH];
    //u64                 mMovesTriedByPly[METRICS_DEPTH][METRICS_MOVES];

    PDECL SearchMetrics()     { this->Clear(); }
    PDECL void Clear()        { PlatClearMemory( this, sizeof( *this ) ); }
};


#include "gpu.h"


template< int POPCNT, typename SIMD >
struct SearchState
{
    static const int LANES = SimdWidth< SIMD >::LANES;

    enum
    {
        STEP_PROCESS,
        STEP_ITERATE_CHILDREN,
        STEP_ALLOC_BATCH,
        STEP_CHECK_SCORE,
        STEP_FINALIZE,
    };


    /// The local state for one stack frame

    struct Frame
    {
        const Position* pos; 
        const MoveMap*  moveMap; 
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
    int                 mSearchDepth;
    int                 mDeepestPly;
    int                 mAsyncSpawnPly;
    MoveList            mBestLine;
    HashTable*          mHashTable;
    Evaluator*          mEvaluator;
    SearchMetrics*      mMetrics;
    volatile bool*      mExitSearch;
    EvalWeight          mWeights[EVAL_TERMS];
    Frame               mFrames[MAX_SEARCH_DEPTH];

#if PIGEON_CUDA_HOST
    CudaChessContext*   mCudaContext;
    SearchBatch*        mCudaBatch;
    int                 mBatchesInFlight;
#endif


    PDECL SearchState()
    {
        mFrameIdx           = 0;
        mSearchDepth        = 0;
        mDeepestPly         = 0;
        mAsyncSpawnPly      = -1;
        mHashTable          = NULL;
        mEvaluator          = NULL;
        mMetrics            = NULL;
        mExitSearch         = NULL;
        mBestLine.Clear();

#if PIGEON_CUDA_HOST
        mCudaContext        = NULL;
        mCudaBatch          = NULL;
        mBatchesInFlight    = 0;
#endif

#if !PIGEON_CUDA_DEVICE
        PlatClearMemory( mWeights, sizeof( mWeights ) );
        memset( mFrames, 0xAA, sizeof( mFrames ) );
#endif
    }


    PDECL INLINE int ChooseNextMove( Frame* f )
    {
        int best = f->moves.mTried;

        for( int idx = best + 1; idx < f->moves.mCount; idx++ )
        {
            MoveSpec& bestMove = f->moves.mMove[best];
            MoveSpec& currMove = f->moves.mMove[idx];

            if( currMove.mFlags < bestMove.mFlags )
                continue;

            if( currMove.mType < bestMove.mType )
                continue;

            // FIXME: I disabled the history table to simplify things while converting from
            // recursive to iterative search. That's done now, so this needs hooking up again.

            //if( (currMove.mType == bestMove.mType) && (currMove.mFlags == bestMove.mFlags) )
            //    if( mHistoryTable[whiteToMove][currMove.mDest][currMove.mSrc] < mHistoryTable[whiteToMove][bestMove.mDest][bestMove.mSrc] )
            //        continue;   

            best = idx;
        }

        Exchange( f->moves.mMove[f->moves.mTried], f->moves.mMove[best] );
        return( f->moves.mTried++ );        
    }


    PDECL INLINE Frame* HandleLeaf( Frame* f )
    {
        mMetrics->mNodesTotal++;
        //mMetrics->mNodesAtPly[f->ply]++;

        if( f->ply > mDeepestPly )
            mDeepestPly = f->ply;

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
        f->moves.UnpackMoveMap( *f->pos, *f->moveMap );
        f->inCheck = (f->moveMap->IsInCheck() != 0);

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
            mHashTable->Load( f->pos->mHash, tt );

            u32 verify = (u32) (f->pos->mHash >> 40);
            if( tt.mHashVerify == verify )
            {
                bool        samePlayer          = (f->pos->mWhiteToMove != 0) == tt.mWhiteMove;
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
        if( f->onPv && (mDeepestPly > f->ply) )
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
            bool spawnAsync = false;

#if PIGEON_CUDA_HOST

            spawnAsync = (mCudaContext && ((f->ply + 1) == mAsyncSpawnPly));
            if( spawnAsync )
            {
                if( mCudaBatch == NULL )
                {
                    f->step = STEP_ALLOC_BATCH;
                    return( f );
                }
            }
#endif
            const MaterialTable* whiteMat = NULL;
            const MaterialTable* blackMat = NULL;

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

                    int idxMove = this->ChooseNextMove( f );
                    f->childSpec[idxLane] = f->moves.mMove[idxMove];

                    SimdInsert( simdSpec.mSrc,  f->childSpec[idxLane].mSrc,  idxLane );
                    SimdInsert( simdSpec.mDest, f->childSpec[idxLane].mDest, idxLane );
                    SimdInsert( simdSpec.mType, f->childSpec[idxLane].mType, idxLane );
                }

                simdPos.Broadcast( *f->pos );
                simdPos.Step( simdSpec, whiteMat, blackMat );
                simdPos.CalcMoveMap( &simdMoveMap );
                simdScore = mEvaluator->Evaluate< POPCNT, SIMD >( simdPos, simdMoveMap, mWeights );

                Unswizzle< SIMD >( &simdPos,     f->childPos );
                Unswizzle< SIMD >( &simdMoveMap, f->childMoveMap );

                u64 PIGEON_ALIGN_SIMD unpackScore[LANES];
                *((SIMD*) unpackScore) = simdScore;

                for( int idxLane = 0; idxLane < LANES; idxLane++ )
                    f->childScore[idxLane] = (EvalTerm) unpackScore[idxLane];

                mMetrics->mNodesTotalSimd += LANES;

                f->simdIdx = 0;
            }

#if PIGEON_CUDA_HOST

            if( spawnAsync )
            {
                assert( mCudaBatch != NULL );

                SearchJobInput* input = mCudaBatch->mInputHost + mCudaBatch->mCount;

                input->mPosition    = f->childPos[f->simdIdx];
                input->mMoveMap     = f->childMoveMap[f->simdIdx];
                input->mScore       = f->childScore[f->simdIdx];
                input->mPly         = f->ply + 1; 
                input->mDepth       = f->depth - 1;
                input->mAlpha       = -f->beta; 
                input->mBeta        = -f->bestScore;

                mCudaBatch->mCount++;
                if( mCudaBatch->mCount == mCudaBatch->mLimit )
                    this->FlushBatch();

                f->movesTried++;
                return( f );
            }
#endif
            Frame* n = f + 1;

            n->pos      = &f->childPos[f->simdIdx];
            n->moveMap  = &f->childMoveMap[f->simdIdx];
            n->score    = f->childScore[f->simdIdx];
            n->ply      = f->ply + 1; 
            n->depth    = f->depth - 1; 
            n->alpha    = -f->beta; 
            n->beta     = -f->bestScore;
            n->onPv     = (f->childSpec[f->simdIdx].mFlags & FLAG_PRINCIPAL_VARIATION)? true : false;
            n->step     = STEP_PROCESS;

            f->step     = STEP_CHECK_SCORE;
            f++;
        }

        return( f );        
    }


    PDECL INLINE Frame* AllocBatch( Frame* f )
    {
#if PIGEON_CUDA_HOST
        assert( mCudaBatch == NULL );

        mCudaBatch = mCudaContext->AllocBatch();
        if( mCudaBatch )
        {
            f->step = STEP_ITERATE_CHILDREN;
        }
#endif

        return( f );
    }


    PDECL INLINE Frame* CheckScore( Frame* f )
    {
        EvalTerm subScore = -f[1].result;

        if( subScore > f->bestScore )
        {
            f->bestScore    = subScore;
            f->bestMove     = f->childSpec[f->simdIdx];

            if( subScore < f->beta )
            {
                //mBestLine.Clear();
                //for( int i = 0; i <= f->ply; i++ )
                //    mBestLine.Append( mFrames[i].bestMove );
            }

            // FIXME
            //this->RegisterBestLine( mBestLine, f->alpha, f->beta );
        }

        f->step = STEP_ITERATE_CHILDREN;
        f->movesTried++;

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

            tt.mHashVerify  = f->pos->mHash >> 40;
            tt.mDepth       = f->depth;
            tt.mScore       = result;
            tt.mBestSrc     = f->bestMove.mSrc;
            tt.mBestDest    = f->bestMove.mDest;
            tt.mFailLow     = failedLow;
            tt.mFailHigh    = failedHigh;
            tt.mWhiteMove   = f->pos->mWhiteToMove? true : false;

            mHashTable->Store( f->pos->mHash, tt );
        }

        f->result = result;
        f--;

        return( f );
    }


    PDECL INLINE bool IsDoneIterating()
    {
        // Have we popped the last frame off the stack?

        return( mFrameIdx < 0 );
    }


    PDECL INLINE bool IsDone()
    {
        bool done = this->IsDoneIterating();

#if PIGEON_CUDA_HOST
        if( mBatchesInFlight )
            done = false;
#endif
        return( done );
    }


#if PIGEON_CUDA_HOST

    PDECL INLINE void FlushBatch()
    {
        assert( mCudaBatch != NULL );

        mCudaContext->SubmitBatch( mCudaBatch );
        mCudaBatch = NULL;

        mBatchesInFlight++;
    }

    PDECL void ProcessCompletedBatch( SearchBatch* batch )
    {
        SearchJobInput*  input  = batch->mInputHost;
        SearchJobOutput* output = batch->mOutputHost;

        for( int i = 0; i < batch->mCount; i++ )
        {
            mMetrics->mGpuNodesTotal += output->mNodes;


            input++;
            output++;
        }
    }

    PDECL void ProcessAllCompletedBatches()
    {
        for( ;; )
        {
            SearchBatch* batch = mCudaContext->GetCompletedBatch();
            if( batch == NULL )
                break;

            mBatchesInFlight--;
            assert( mBatchesInFlight >= 0 );
            printf( "!" );

            this->ProcessCompletedBatch( batch );
            mCudaContext->ReleaseBatch( batch );
        }
    }
#endif


    PDECL void Advance()
    {
#if PIGEON_CUDA_HOST

        if( mCudaBatch && this->IsDoneIterating() )
            this->FlushBatch();

        if( mBatchesInFlight )
        {
            this->ProcessAllCompletedBatches();

            if( this->IsDoneIterating() )
            {
                if( mBatchesInFlight )
                    PlatSleep( 1 );

                return;
            }
        }
#endif

        assert( mFrameIdx >= 0 );
        assert( mFrameIdx < MAX_SEARCH_DEPTH );

        Frame* f = mFrames + mFrameIdx;

        if( f->step == STEP_PROCESS )           f = this->HandleLeaf( f );
        if( f->step == STEP_PROCESS )           f = this->HandleMate( f );
        if( f->step == STEP_PROCESS )           f = this->Quieten( f );
        if( f->step == STEP_PROCESS )           f = this->CheckHashTable( f );
        if( f->step == STEP_PROCESS )           f = this->PrepareToIterate( f );
        if( f->step == STEP_ITERATE_CHILDREN )  f = this->IterateChildren( f );
        if( f->step == STEP_ALLOC_BATCH )       f = this->AllocBatch( f );
        if( f->step == STEP_CHECK_SCORE )       f = this->CheckScore( f );
        if( f->step == STEP_FINALIZE )          f = this->StoreIntoHashTable( f );

        mFrameIdx = (int) (f - mFrames);
    }


    PDECL void ExtractBestLine( MoveList* bestLine )
    {
        bestLine->Clear();
        for( int i = 0; i < mSearchDepth; i++ )
            bestLine->Append( mFrames[i].bestMove );

    }


    PDECL EvalTerm GetFinalScore()
    {
        return( mFrames[0].result );
    }


    PDECL void PrepareSearch( const Position* root, const MoveMap* moveMap, int depth, int ply, EvalTerm score, EvalTerm alpha, EvalTerm beta )
    {
        Frame* f = mFrames;

        float gamePhase = mEvaluator->CalcGamePhase< POPCNT >( *root );
        mEvaluator->GenerateWeights( mWeights, gamePhase );

        f->pos      = root;
        f->moveMap  = moveMap;
        f->score    = score;
        f->ply      = ply;
        f->depth    = depth;
        f->alpha    = alpha; 
        f->beta     = beta;
        f->onPv     = true;
        f->step     = STEP_PROCESS;       

        mSearchDepth = depth;
        mFrameIdx = 0;
    }


    PDECL EvalTerm RunToDepth( const Position* root, const MoveMap* moveMap, int depth, int ply, EvalTerm score, EvalTerm alpha, EvalTerm beta )
    {
        this->PrepareSearch( root, moveMap, depth, ply, score, alpha, beta );

        while( !this->IsDone() )
        {
            this->Advance();

            if( mExitSearch && *mExitSearch )
                break;
        }

        return( this->GetFinalScore() );
    }
};



#endif // PIGEON_SEARCH_H__
};
