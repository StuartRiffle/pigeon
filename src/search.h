// search.h - PIGEON CHESS ENGINE (c) 2012-2016 Stuart Riffle

#if PIGEON_ENABLE_CUDA
#include <vector>
#include <stack>
#include <queue>
#include <memory>
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
    u64                 mSteps;                 
    u64                 mNodesAtPly[METRICS_DEPTH];
    u64                 mHashLookupsAtPly[METRICS_DEPTH];
    u64                 mHashHitsAtPly[METRICS_DEPTH];
    u64                 mPlyMovesTriedByOrder[METRICS_DEPTH][METRICS_MOVES];

    PDECL SearchMetrics()     { this->Clear(); }
    PDECL void Clear()        { PlatClearMemory( this, sizeof( *this ) ); }
};

struct SearchBatch;

struct IAsyncSearcher
{
    virtual SearchBatch*    AllocBatch() = 0;
    virtual void            SubmitBatch( SearchBatch* batch ) = 0;
    virtual SearchBatch*    GetCompletedBatch() = 0;
    virtual void            ReleaseBatch( SearchBatch* batch ) = 0;
    virtual void            CancelAllBatchesSync() = 0;
};


struct HistoryTable
{
    HistoryTerm     mTerm[2][64][64];   /// Indexed as [whiteToMove][dest][src]

    PDECL void Clear()    
    { 
        PlatClearMemory( this, sizeof( *this ) ); 
    }

    void Decay()
    {
        size_t count = sizeof( mTerm ) / sizeof( mTerm[0] );

        for( int i = 0; i < count; i++ )
            ((HistoryTerm*) mTerm)[i] >>= 1;
    }
};

#include "gpu-cuda.h"


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
        STEP_FINALIZE
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
    HashTable*          mHashTable;
    Evaluator*          mEvaluator;
    SearchMetrics*      mMetrics;
    volatile bool*      mExitSearch;
    EvalWeight          mWeights[EVAL_TERMS];
    Frame               mFrames[MAX_SEARCH_DEPTH];
    HistoryTable*       mHistoryTable;  

#if PIGEON_CUDA_HOST
    IAsyncSearcher*     mCudaSearcher;
    SearchBatch*        mCudaBatch;
    int                 mBatchesInFlight;
    int                 mBatchLimit;
    int                 mStepsUntilPoll;
#endif


    PDECL SearchState()
    {
        mFrameIdx           = 0;
        mSearchDepth        = 0;
        mDeepestPly         = 0;
        mAsyncSpawnPly      = -10;
        mHashTable          = NULL;
        mEvaluator          = NULL;
        mMetrics            = NULL;
        mExitSearch         = NULL;

#if PIGEON_CUDA_HOST
        mCudaSearcher       = NULL;
        mCudaBatch          = NULL;
        mBatchesInFlight    = 0;
        mBatchLimit         = 0;
        mStepsUntilPoll     = 0;
#endif

        PlatClearMemory( mWeights, sizeof( mWeights ) );
        //PlatClearMemory( mHistoryTable, sizeof( mHistoryTable ) );

#if !PIGEON_CUDA_DEVICE
        memset( mFrames, 0xAA, sizeof( mFrames ) );
#endif
    }



    //--------------------------------------------------------------------------
    /// Compare two moves and decide which should be tried first
    ///
    /// Effective order is:
    ///     - Principal variation [identified by a flag]
    ///     - Transition table move [identified by a flag]
    ///     - Promotions (queen, rook, bishop, knight) [based on the type index]
    ///     - Captures (winning, even, losing) [based on the type index]
    ///     - Regular moves ordered by the history table
    ///
    /// \param curr         The move being considered
    /// \param best         The best move found so far
    /// \param whiteToMove  Used to index the history table
    /// \return             True if move curr looks better 

    PDECL INLINE bool IsBetterMove( const MoveSpec& curr, const MoveSpec& best, int whiteToMove ) const
    {
        if( curr.mFlags != best.mFlags )
            return( curr.mFlags > best.mFlags );

        if( curr.mType != best.mType )
            return( curr.mType > best.mType );

        HistoryTerm& currHist = mHistoryTable->mTerm[whiteToMove][curr.mDest][curr.mSrc];
        HistoryTerm& bestHist = mHistoryTable->mTerm[whiteToMove][best.mDest][best.mSrc];
        
        return( currHist > bestHist );
    }


    //--------------------------------------------------------------------------
    /// Select the best move to try from the ones remaining
    ///
    /// This works like a selection sort. Each move chosen is swapped
    /// into position starting from the front of the list. 
    ///
    /// \param f    Stack frame
    /// \return     Move index into f->moves.mMove[]
    
    PDECL INLINE int ChooseNextMove( Frame* f )
    {
        assert( f->moves.mTried < f->moves.mCount );

        int best        = f->moves.mTried;
        int whiteToMove = (int) f->pos->mWhiteToMove;

        for( int idx = best + 1; idx < f->moves.mCount; idx++ )
            if( this->IsBetterMove( f->moves.mMove[idx], f->moves.mMove[best], whiteToMove ) )
                best = idx;

        Exchange( f->moves.mMove[f->moves.mTried], f->moves.mMove[best] );

        return( f->moves.mTried++ );        
    }


    //--------------------------------------------------------------------------
    /// Terminate the search at the leaf nodes
    ///
    /// \param f    Current stack frame
    /// \return     Stack frame after processing 

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


    //--------------------------------------------------------------------------
    /// If no moves are available, distinguish between checkmate and stalemate
    ///
    /// \param f    Current stack frame
    /// \return     Stack frame after processing 

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


    //--------------------------------------------------------------------------
    /// Quiescence search
    ///
    /// \param f    Current stack frame
    /// \return     Stack frame after processing 

    PDECL INLINE Frame* Quieten( Frame* f )
    {
        if( f->depth < 1 )
        {
            bool atStackLimit = ((mFrameIdx + 1) >= MAX_SEARCH_DEPTH);

            if( !f->inCheck )
                f->moves.DiscardMovesBelow( CAPTURE_LOSING );

            if( (f->moves.mCount == 0) || atStackLimit )
            {
                f->result = f->score;
                f--;
            }
        }

        return( f );
    }


    //--------------------------------------------------------------------------
    /// Look up the current position in the hash table
    ///
    /// \param f    Current stack frame
    /// \return     Stack frame after processing 

    PDECL INLINE Frame* CheckHashTable( Frame* f )
    {
        if( (f->depth > 0) && !f->onPv )
        {
            mMetrics->mHashLookupsAtPly[f->ply]++;

            TableEntry tt;
            mHashTable->Load( f->pos->mHash, tt );

            u32 verify = (u32) (f->pos->mHash >> 40);
            if( tt.mHashVerify == verify )
            {
                mMetrics->mHashHitsAtPly[f->ply]++;

                bool returnScore = false;

                if( tt.mDepth >= f->depth )
                {
                    if( tt.mFailHigh )
                    {
                        f->alpha = Max( f->alpha, tt.mScore );
                    }
                    else if( tt.mFailLow )
                    {
                        f->beta = Min( f->beta, tt.mScore );
                    }
                    else
                    {
                        returnScore = true;
                    }
                
                    if( f->alpha >= f->beta )
                        returnScore = true;
                }
                
                if( returnScore )
                {
                    f->result = tt.mScore;
                    f--;
                }
                else
                {
                    f->moves.FlagSpecialMove( tt.mBestSrc, tt.mBestDest, FLAG_TT_BEST_MOVE );
                }
            }
        }

        return( f );
    }


    //--------------------------------------------------------------------------
    /// Initialize the per-child loop
    ///
    /// \param f    Current stack frame
    /// \return     Stack frame after processing 

    PDECL INLINE Frame* PrepareToIterate( Frame* f )
    {
        if( f->onPv && (mDeepestPly >= f->ply) )
        {
            f->moves.FlagSpecialMove( f->bestMove.mSrc, f->bestMove.mDest, FLAG_PRINCIPAL_VARIATION );
        }

        f->movesTried   = 0;
        f->bestScore    = f->alpha;
        f->simdIdx      = LANES - 1;
        f->step         = STEP_ITERATE_CHILDREN;        

        return( f );
    }


    //--------------------------------------------------------------------------
    /// Step forward a group of positions in SIMD
    ///
    /// LANES is the SIMD width. Make that many moves in parallel, calculate
    /// valid moves for the resulting positions, and evaluate them.
    ///
    /// For scalar types (including CUDA device code!) all the SIMD stuff
    /// will mostly optimize away. The last part (from Unswizzle() on)
    /// has a couple of copies that could be avoided.

    PDECL INLINE void StepPositionsSIMD( Frame* f )
    {
        const MaterialTable* whiteMat = NULL;
        const MaterialTable* blackMat = NULL;

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

        // IMPORTANT: only the low 32 bits of the raw score are valid,
        // and they contain a signed value

        for( int idxLane = 0; idxLane < LANES; idxLane++ )
            f->childScore[idxLane] = (EvalTerm) unpackScore[idxLane];

        mMetrics->mNodesTotalSimd += LANES;

        // TODO: peek at the hash table entries for these new positions, and
        // re-order them if it looks like it would help!
    }


    //--------------------------------------------------------------------------
    /// Perform one iteration of this position's subtrees
    ///
    /// If spawnAsync, we're going to cut off the child subtree for async execution.
    /// 
    /// \param f    Current stack frame
    /// \return     Stack frame after processing 

    PDECL INLINE Frame* IterateChildren( Frame* f )
    {
        if( (f->movesTried >= f->moves.mCount) || (f->bestScore >= f->beta) )
        {
            f->step = STEP_FINALIZE;
        }
        else
        {
#if PIGEON_CUDA_HOST
            bool spawnAsync = (mCudaSearcher && ((f->ply + 1) == mAsyncSpawnPly));

            if( mBatchesInFlight >= mBatchLimit )
                spawnAsync = false;

            if( (f->movesTried < 1) || f->onPv )
                spawnAsync = false;

            if( spawnAsync )
            {
                if( mCudaBatch == NULL )
                {
                    f->step = STEP_ALLOC_BATCH;
                    return( f );
                }
            }
#endif

            f->simdIdx++;
            if( f->simdIdx >= LANES )
            {
                this->StepPositionsSIMD( f );
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

                for( int i = 0; i <= f->ply; i++ )
                    input->mPath[i] = mFrames[i].moves.mMove[mFrames[i].movesTried];

                mCudaBatch->mCount++;

                if( mCudaBatch->mCount == mCudaBatch->mLimit )
                    this->FlushBatch();

                f->movesTried++;
                return( f );
            }
#endif

            if( f->movesTried < METRICS_MOVES )
                mMetrics->mPlyMovesTriedByOrder[f->ply][f->movesTried]++;

            Frame* n = f + 1;

            n->pos          = &f->childPos[f->simdIdx];
            n->moveMap      = &f->childMoveMap[f->simdIdx];
            n->score        = f->childScore[f->simdIdx];
            n->ply          = f->ply + 1; 
            n->depth        = f->depth - 1; 
            n->alpha        = -f->beta; 
            n->beta         = -f->bestScore;
            n->onPv         = (f->childSpec[f->simdIdx].mFlags & FLAG_PRINCIPAL_VARIATION)? true : false;
            n->step         = STEP_PROCESS;

            f->step = STEP_CHECK_SCORE;
            f++;
        }

        return( f );        
    }


    //--------------------------------------------------------------------------
    /// Allocate a new job batch
    ///
    /// \param f    Current stack frame
    /// \return     Stack frame after processing 

    PDECL INLINE Frame* AllocBatch( Frame* f )
    {
#if PIGEON_CUDA_HOST
        assert( mCudaBatch == NULL );

        mCudaBatch = mCudaSearcher->AllocBatch();
        if( mCudaBatch )
        {
            f->step = STEP_ITERATE_CHILDREN;
        }
#endif

        return( f );
    }


    //--------------------------------------------------------------------------
    /// After processing a subtree, see if we can (effectively) raise alpha
    ///
    /// \param f    Current stack frame
    /// \return     Stack frame after processing 

    PDECL INLINE Frame* CheckScore( Frame* f )
    {
        EvalTerm subScore = -f[1].result;

        bool failedHigh = (subScore >= f->beta);
        bool failedLow  = (subScore == f->bestScore);

        if( subScore > f->bestScore )
        {
            f->bestScore    = subScore;
            f->bestMove     = f->childSpec[f->simdIdx];

            if( (f->bestMove.mType == MOVE) && (f->depth > 1) )
            {
                HistoryTerm& term = mHistoryTable->mTerm[f->pos->mWhiteToMove][f->bestMove.mDest][f->bestMove.mSrc];
            
                term += (1 << f->depth);
            }
        }

        f->step = STEP_ITERATE_CHILDREN;
        f->movesTried++;

        return( f );
    }


    //--------------------------------------------------------------------------
    /// Write the current position into the hash table
    ///
    /// \param f    Current stack frame
    /// \return     Stack frame after processing 

    PDECL INLINE Frame* StoreIntoHashTable( Frame* f )
    {
        u8          failedHigh  = (f->bestScore >= f->beta)?  1 : 0;
        u8          failedLow   = (f->bestScore == f->alpha)? 1 : 0;
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

            mHashTable->Store( f->pos->mHash, tt );
        }

        f->result = result;
        f--;

        return( f );
    }


    //--------------------------------------------------------------------------
    /// Check if the main tree search is done
    ///
    /// There may still be async searches pending.
    ///
    /// \param f    Current stack frame
    /// \return     Stack frame after processing 

    PDECL INLINE bool IsDoneIterating()
    {
        // Have we popped the last frame off the stack?

        return( mFrameIdx < 0 );
    }


    //--------------------------------------------------------------------------
    /// Check if the entire search is done, async jobs and all
    ///
    /// \param f    Current stack frame
    /// \return     Stack frame after processing 

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
    //--------------------------------------------------------------------------
    /// Submit a batch of async search jobs
    ///
    /// \param f    Current stack frame
    /// \return     Stack frame after processing 

    PDECL INLINE void FlushBatch()
    {
        assert( mCudaBatch != NULL );

        mCudaSearcher->SubmitBatch( mCudaBatch );
        mCudaBatch = NULL;

        mBatchesInFlight++;
    }


    //--------------------------------------------------------------------------
    /// If any of the async searches has found a new best line,
    /// we modify the call stack accordingly. 
    ///
    /// \param f    Current stack frame
    /// \return     Stack frame after processing 

    PDECL void ProcessCompletedBatch( SearchBatch* batch )
    {
        SearchJobInput*  input  = batch->mInputHost;
        SearchJobOutput* output = batch->mOutputHost;

        int nodes = 0;
        int mostSteps = 0;
        for( int i = 0; i < batch->mCount; i++ )
        {
            nodes += (int) output->mNodes;
            mostSteps = Max( mostSteps, (int) output->mSteps );

            mMetrics->mGpuNodesTotal += output->mNodes;

            //int ply = input->mPly;
            mDeepestPly = Max( mDeepestPly, input->mPly + output->mDeepestPly );

            EvalTerm scoreAsync = -output->mScore;

            int sharedMoves = 0;
            while( (mFrames[sharedMoves].bestMove == input->mPath[sharedMoves]) && (sharedMoves <= input->mPly) )
                sharedMoves++;


            int ply = input->mPly;
            while( (ply > mFrameIdx) && (ply >= sharedMoves) )
            {
                ply--;
                scoreAsync *= -1;
            }

            if( ply >= 0 )
            {
                if( scoreAsync > mFrames[ply].bestScore )
                {
                    //printf( "Ply %d, replacing bestScore %d with async %d\n", ply, mFrames[ply].bestScore, scoreAsync );

                    mFrames[ply].bestScore = Max( scoreAsync, mFrames[ply].beta );
                    mFrameIdx = ply;

                    for( int i = ply; i < input->mPly; i++ )
                        mFrames[i].bestMove = input->mPath[i];
                
                    for( int i = 0; i <= input->mDepth; i++ )
                        mFrames[i + input->mPly].bestMove = output->mPath[i];

                }
            }

            input++;
            output++;
        }

        //int npms = (int) (nodes / batch->mGpuTime);
        //printf( "%4d jobs, %6d nodes, GPU time %6.1fms, CPU latency %6.1fms, most steps %4d, nps %4dk\n", batch->mCount, nodes, batch->mGpuTime, batch->mCpuLatency, mostSteps, npms );
    }


    //--------------------------------------------------------------------------
    /// Process any batches of async search jobs that are ready
    ///
    /// \param f    Current stack frame
    /// \return     Stack frame after processing 

    PDECL void CheckForCompletedBatches()
    {
        for( ;; )
        {
            SearchBatch* batch = mCudaSearcher->GetCompletedBatch();
            if( batch == NULL )
                break;

            mBatchesInFlight--;
            assert( mBatchesInFlight >= 0 );
            //printf( "!" );

            this->ProcessCompletedBatch( batch );
            mCudaSearcher->ReleaseBatch( batch );
        }
    }
#endif


    //--------------------------------------------------------------------------
    /// Advance the current stack frame
    ///
    /// This is the core of the iterative search. On a single CPU, this is
    /// equivalent to a (tortuously) unwound recursive search. On parallel
    /// hardware, only some threads in a warp will be in any given state.
    /// The others will be predicated off. But all the threads will make
    /// forward progress of some sort in each pass.
    ///
    /// \param f    Current stack frame
    /// \return     Stack frame after processing 
    
    PDECL INLINE Frame* AdvanceState( Frame* f )
    {
        if( f->step == STEP_PROCESS )           
            f = this->HandleLeaf( f );

        if( f->step == STEP_PROCESS )           
            f = this->HandleMate( f );

        if( f < mFrames )
            return( f );

        if( f->step == STEP_PROCESS )           
            f = this->Quieten( f );

        if( f->step == STEP_PROCESS )           
            f = this->CheckHashTable( f );

        if( f->step == STEP_PROCESS )           
            f = this->PrepareToIterate( f );

        if( f->step == STEP_ITERATE_CHILDREN )  
            f = this->IterateChildren( f );

        if( f->step == STEP_ALLOC_BATCH )       
            f = this->AllocBatch( f );

        if( f->step == STEP_CHECK_SCORE )       
            f = this->CheckScore( f );

        if( f->step == STEP_FINALIZE )          
            f = this->StoreIntoHashTable( f );

        return( f );
    }


    //--------------------------------------------------------------------------
    /// Take a step forward in the search
    ///
    /// \param f    Current stack frame
    /// \return     Stack frame after processing 
    
    PDECL void Step()
    {
#if PIGEON_CUDA_HOST

        if( mCudaBatch && this->IsDoneIterating() )
            this->FlushBatch();

        if( mBatchesInFlight )
        {
            if( --mStepsUntilPoll < 1 )
            {
                this->CheckForCompletedBatches();
                mStepsUntilPoll = GPU_BATCH_POLL_STEPS;

                if( this->IsDoneIterating() && mBatchesInFlight )
                    PlatSleep( 1 );
            }
        }

        if( this->IsDoneIterating() )
            return;
#endif
        assert( mFrameIdx >= 0 );
        assert( mFrameIdx < MAX_SEARCH_DEPTH );

        Frame* f = mFrames + mFrameIdx;

        f = this->AdvanceState( f );
        mMetrics->mSteps++;

        mFrameIdx = (int) (f - mFrames);
    }


    //--------------------------------------------------------------------------
    /// Extract the best line from the corpses of the stack frames
    ///
    /// \param f    Current stack frame
    /// \return     Stack frame after processing 
    
    PDECL void ExtractBestLine( MoveList* bestLine )
    {
        assert( this->IsDone() );

        bestLine->Clear();
        for( int i = 0; i < mSearchDepth; i++ )
            bestLine->Append( mFrames[i].bestMove );

    }

    PDECL void InsertBestLine( MoveList* bestLine )
    {
        //assert( this->IsDone() );

        for( int i = 0; i < bestLine->mCount; i++ )
            mFrames[i].bestMove = bestLine->mMove[i];
    }


    PDECL EvalTerm GetFinalScore()
    {
        return( mFrames[0].result );
    }


    PDECL void PrepareSearch( const Position* root, const MoveMap* moveMap, int depth, int ply, EvalTerm score, EvalTerm alpha, EvalTerm beta )
    {
        Frame* f = mFrames;
        mFrameIdx = 0;

        float gamePhase = mEvaluator->CalcGamePhase< POPCNT >( *root );
        mEvaluator->GenerateWeights( mWeights, gamePhase );

        f->pos          = root;
        f->moveMap      = moveMap;
        f->score        = score;
        f->ply          = ply;
        f->depth        = depth;
        f->alpha        = alpha; 
        f->beta         = beta;
        f->onPv         = true;
        f->step         = STEP_PROCESS;       

        mSearchDepth = depth;
    }


    //--------------------------------------------------------------------------
    /// Perform a search to a given depth
    ///
    /// The CPU code uses this function to set up a search, then step it until
    /// done. The GPU code in kernel has its own loop to step the search.
    ///
    PDECL EvalTerm RunToDepth( const Position* root, const MoveMap* moveMap, int depth, int ply, EvalTerm score, EvalTerm alpha, EvalTerm beta )
    {
        this->PrepareSearch( root, moveMap, depth, ply, score, alpha, beta );

        while( !this->IsDone() )
        {
            this->Step();

            if( mExitSearch && *mExitSearch )
                break;
        }

        return( this->GetFinalScore() );
    }
};



#endif // PIGEON_SEARCH_H__
};
