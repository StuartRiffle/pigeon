// search.h - PIGEON CHESS ENGINE (c) 2012-2016 Stuart Riffle
        
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
    u64                 mNodesAtPly[METRICS_DEPTH];
    u64                 mHashLookupsAtPly[METRICS_DEPTH];
    u64                 mHashHitsAtPly[METRICS_DEPTH];
    u64                 mMovesTriedByPly[METRICS_DEPTH][METRICS_MOVES];

    PDECL SearchMetrics()     { this->Clear(); }
    PDECL void Clear()        { PlatClearMemory( this, sizeof( *this ) ); }
};


struct SearchJobInput
{
	Position            mPosition;
	HashTable*          mHashTable;
	Evaluator*          mEvaluator;
	int                 mSearchDepth;
};

struct SearchJobOutput
{
	MoveList			mBestLine;
    u64                 mNodes;
	EvalTerm			mScore;
};


#ifdef PIGEON_ENABLE_CUDA

struct SearchBatch
{
    SearchJobInput*     mInputHost;
    SearchJobInput*     mInputDev;
    SearchJobOutput*    mOutputHost;
    SearchJobOutput*    mOutputDev;
    cudaEvent_t         mEvent;
    cudaStream_t        mStream;
    int                 mCount;
    int                 mLimit;
};

extern "C" void QueueSearchBatch( SearchBatch* batch, int blockSize );

#endif


template< int POPCNT, typename SIMD >
struct SearchState
{
    static const int LANES = SimdWidth< SIMD >::LANES;

    enum
    {
        STEP_PROCESS,
        STEP_ITERATE_CHILDREN,
        STEP_CHECK_SCORE,
        STEP_FINALIZE,
    };


    /// The local state for one stack frame

    struct Frame
    {
        Position*       pos; 
        MoveMap*        moveMap; 
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
    Position            mRoot;
    MoveMap             mRootMoveMap;
    MoveList            mBestLine;
    HashTable*          mHashTable;
    Evaluator*          mEvaluator;
    SearchMetrics*      mMetrics;
    volatile bool*      mExitSearch;
    EvalWeight          mWeights[EVAL_TERMS];
    Frame               mFrames[MAX_SEARCH_DEPTH];

#if PIGEON_CUDA_HOST
    CudaChessContext*   mCudaContext;
#endif


    PDECL SearchState()
    {
        mFrameIdx       = 0;
        mSearchDepth    = 0;
        mDeepestPly     = 0;
        mHashTable      = NULL;
        mEvaluator      = NULL;
        mMetrics        = NULL;
        mExitSearch     = NULL;
        mBestLine.Clear();

#if !PIGEON_CUDA_DEVICE
        PlatClearMemory( mWeights, sizeof( mWeights ) );
        memset( mFrames, 0xAA, sizeof( mFrames ) );
#endif
    }


    PDECL INLINE int ChooseNextMove( Frame* f )
    {
        int best = f->moves.mTried;
        //int whiteToMove = f->pos->mWhiteToMove;

        for( int idx = best + 1; idx < f->moves.mCount; idx++ )
        {
            MoveSpec& bestMove = f->moves.mMove[best];
            MoveSpec& currMove = f->moves.mMove[idx];

            if( currMove.mFlags < bestMove.mFlags )
                continue;

            if( currMove.mType < bestMove.mType )
                continue;

            //if( (currMove.mType == bestMove.mType) && (currMove.mFlags == bestMove.mFlags) )
            //    if( mHistoryTable[whiteToMove][currMove.mDest][currMove.mSrc] < mHistoryTable[whiteToMove][bestMove.mDest][bestMove.mSrc] )
            //        continue;   

            best = idx;
        }

        Exchange( f->moves.mMove[f->moves.mTried], f->moves.mMove[best] );
        return( f->moves.mTried++ );        
    }


    PDECL void QueueAsyncSearch( const Position* pos )
    {

    }


    PDECL INLINE Frame* HandleLeaf( Frame* f )
    {
        mMetrics->mNodesTotal++;
        mMetrics->mNodesAtPly[f->ply]++;

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


#if PIGEON_CUDA_HOST
    PDECL void ProcessCompletedBatch( SearchBatch* batch )
    {
        for( int i = 0; i < batch->mCount; i++ )
        {
            SearchJobOutput* result = batch->mOutputHost + i;

            mMetrics->mGpuNodesTotal += result->mNodes;
        }
    }

    PDECL void ProcessCompletedBatches()
    {
        for( ;; )
        {
            SearchBatch* batch = mCudaContext->GetCompletedBatch();
            if( batch == NULL )
                break;

            this->ProcessCompletedBatch( batch );
            mCudaContext->ReleaseBatch( batch );
        }
    }
#endif


    PDECL INLINE Frame* HandleSpawnAsync( Frame* f )
    {
#if PIGEON_CUDA_HOST
        if( mCudaContext && (f->ply == mAsyncSpawnPly) )
        {
            bool triedBefore = false;

            for( ;; )
            {
                if( triedBefore )
                {
                    // We are GPU bound

                    PlatSleep( 1 );
                }

                mCudaBatch = mCudaContext->AllocBatch();
                if( mCudaBatch )
                    break;

                triedBefore = true;

                this->ProcessCompletedBatches();
            }

            SearchJobInput* input = mCudaBatch->mInputHost + mCudaBatch->mCount;

            input->mPosition    = *f->pos;
            input->mSearchDepth = f->depth;

            mCudaBatch->mCount++;
            if( mCudaBatch->mCount == mCudaBatch->mLimit )
            {
                mCudaDevice->SubmitBatch( mCudaBatch );
                mCudaBatch = NULL;
            }

            // This subtree will be processed async, so back out

            f->result = f->score;
            f--;
        }
#endif

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


    PDECL INLINE Frame* CheckScore( Frame* f )
    {
        EvalTerm subScore = -f[1].result;

        //FEN::PrintMoveSpec( f->childSpec[f->simdIdx] );
        //printf( " %d\n", subScore );

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


    PDECL void Advance()
    {
        assert( mFrameIdx >= 0 );
        assert( mFrameIdx < MAX_SEARCH_DEPTH );

        Frame* f = mFrames + mFrameIdx;

        if( f->step == STEP_PROCESS )           f = this->HandleLeaf( f );
        if( f->step == STEP_PROCESS )           f = this->HandleMate( f );
        if( f->step == STEP_PROCESS )           f = this->HandleSpawnAsync( f );
        if( f->step == STEP_PROCESS )           f = this->Quieten( f );
        if( f->step == STEP_PROCESS )           f = this->CheckHashTable( f );
        if( f->step == STEP_PROCESS )           f = this->PrepareToIterate( f );
        if( f->step == STEP_ITERATE_CHILDREN )  f = this->IterateChildren( f );
        if( f->step == STEP_CHECK_SCORE )       f = this->CheckScore( f );
        if( f->step == STEP_FINALIZE )          f = this->StoreIntoHashTable( f );

        mFrameIdx = (int) (f - mFrames);
    }


    PDECL INLINE bool IsDone()
    {
        return( mFrameIdx < 0 );
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


    PDECL void PrepareSearch( const Position& root, int depth )
    {
        Frame* f = mFrames;

        float gamePhase = mEvaluator->CalcGamePhase< POPCNT >( root );
        mEvaluator->GenerateWeights( mWeights, gamePhase );

        mRoot = root;
        mRoot.CalcMoveMap( &mRootMoveMap );

        mSearchDepth = depth;

        f->pos      = &mRoot;
        f->moveMap  = &mRootMoveMap;
        f->score    = (EvalTerm) mEvaluator->Evaluate< POPCNT >( root, mRootMoveMap, mWeights );
        f->ply      = 0;
        f->depth    = depth;
        f->alpha    = -EVAL_MAX; 
        f->beta     = EVAL_MAX;
        f->onPv     = true;
        f->step     = STEP_PROCESS;       

        mFrameIdx = 0;
    }


    PDECL EvalTerm RunToDepth( const Position& root, int depth )
    {
        this->PrepareSearch( root, depth );

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
