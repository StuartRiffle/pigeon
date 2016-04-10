// engine.h - PIGEON CHESS ENGINE (c) 2012-2016 Stuart Riffle
        
namespace Pigeon {
#ifndef PIGEON_ENGINE_H__
#define PIGEON_ENGINE_H__


class Engine
{
    HashTable           mHashTable;                 // Transposition table
    Position            mRoot;                      // The root of the search tree (the "current position")
    SearchConfig        mConfig;                    // Search parameters
    SearchMetrics       mMetrics;                   // Runtime metrics
    Evaluator           mEvaluator;                 // Evaluation weights
    MoveList            mBestLine;                  // Best line found in the search
    MoveList*           mStorePv;                   // Target for PV in active search
    EvalTerm            mValuePv;                   // The evaluation of *mStorePv
    int                 mTableSize;                 // Transposition table size (in megs)
    int                 mTargetTime;                // Time to stop current search
    int                 mDepthLimit;                // Depth limit for current search (not counting quiesence)
    Timer               mSearchElapsed;             // Time elapsed since the "go" command
    volatile bool       mExitSearch;                // Flag to terminate search threads immediately
    int                 mThreadsRunning;            // Number of worker threads currently running
    Semaphore           mThreadsDone;               // Semaphore to help gather up completed threads                                                
    bool                mPrintBestMove;             // Output best move while searching
    bool                mPrintedMove;               // Make sure only one "bestmove" is output per "go" 
    bool                mDebugMode;                 // This currently does nothing
    int                 mRecalcWeightFreq;          // Frequency at which to recalculate weights based on game phase (power of two) 
    bool                mUsePopcnt;                 // Enable use of the hardware POPCNT instruction
    int                 mCpuLevel;                  // A CPU_* enum value to select the code path

public:
    Engine()
    {
        mConfig.Clear();
        mMetrics.Clear();
        mRoot.Reset();

        mStorePv            = NULL;
        mValuePv            = EVAL_MAX;
        mTableSize          = TT_MEGS_DEFAULT;
        mTargetTime         = NO_TIME_LIMIT;
        mDepthLimit         = 0;
        mExitSearch         = false;
        mThreadsRunning     = 0;
        mPrintBestMove      = false;
        mPrintedMove        = false;
        mDebugMode          = false;
        mRecalcWeightFreq   = 1;
        mUsePopcnt          = PlatDetectPopcnt();
        mCpuLevel           = PlatDetectCpuLevel();

        mHashTable.SetSize( mTableSize );
    }

    ~Engine()
    {
        this->Stop();
    }

    void Reset()
    {
        this->Stop();
        mRoot.Reset();
    }

    void SetPosition( const Position& pos )
    {
        this->Reset();
        mRoot = pos;

        mRoot.mHash = mRoot.CalcHash();
    }

    Position GetPosition() const
    {
        return( mRoot );    
    }

    void SetHashTableSize( int megs )
    {
        mTableSize = megs;
    }

    void Init()
    {
        if( mTableSize != mHashTable.GetSize() )
            mHashTable.SetSize( mTableSize );
    }

    void SetDebug( bool debug )
    {
        mDebugMode = debug;
    }

    void OverrideCpuLevel( int level )
    {
        mCpuLevel = level;
    }

    void OverridePopcnt( bool enabled )
    {
        mUsePopcnt = enabled;
    }    

    void LoadWeightParam( const char* name, int openingVal, int midgameVal, int endgameVal ) 
    {
        int idx = mEvaluator.GetWeightIdx( name );

        if( idx >= 0 )
            mEvaluator.SetWeight( idx, openingVal, midgameVal, endgameVal );
    }


    void Move( const char* movetext )
    {
        MoveSpec move;
        if( FEN::StringToMoveSpec( movetext, move ) )
        {
            MoveList valid;
            valid.FindMoves( mRoot );

            int idx = valid.LookupMove( move );
            if( idx >= 0 )
            {
                mRoot.Step( move );
            }
            else
            {
                printf( "info string ERROR: \"%s\" is not a valid move\n", movetext );
                this->PrintPosition();
            }
        }
        else
        {
            printf( "info string ERROR: \"%s\" cannot be parsed as a move\n", movetext );
            this->PrintPosition();
        }
    }


    void PonderHit()
    {
        printf( "info string WARNING: ponderhit not implemented\n" );
    }


    void Go( SearchConfig* conf )
    {
        mSearchElapsed.Reset();

        MoveList valid;
        valid.FindMoves( mRoot );

        if( valid.mCount == 0 )
        {
            printf( "info string ERROR: no moves available at position " );
            FEN::PrintPosition( mRoot );
            printf( "\n" );
            return;
        }

        this->Stop();

        mBestLine.Clear();

        mConfig         = *conf;
        mStorePv        = &mBestLine;
        mValuePv        = 0;
        mDepthLimit     = mConfig.mDepthLimit;
        mTargetTime     = this->CalcTargetTime();
        mExitSearch     = false;
        mPrintBestMove  = true;
        mPrintedMove    = false;

        PlatSpawnThread( &Engine::SearchThreadProc, this );
        mThreadsRunning++;

        if( mTargetTime != NO_TIME_LIMIT )
        {
            PlatSpawnThread( &Engine::TimerThreadProc, this );
            mThreadsRunning++;
        }
    }


    void Stop( bool printResult = false )
    {
        this->JoinAllThreads();

        if( printResult )
            this->PrintResult();
    }

    void PrintPosition()
    {
        printf( "info string position " );
        FEN::PrintPosition( mRoot );
        printf( "\n" );
    }

private:

    static void* SearchThreadProc( void* param )
    {
        Engine* engine = reinterpret_cast< Engine* >( param );
        engine->SearchThread();
        engine->mThreadsDone.Post();
        return( NULL );
    }

    static void* TimerThreadProc( void* param )
    {
        Engine* engine = reinterpret_cast< Engine* >( param );
        engine->TimerThread();
        engine->mThreadsDone.Post();
        return( NULL );
    }

    void SearchThread()
    {
        int depth = 1;
        i64 prevLevelElapsed = 0;

        while( !mExitSearch )
        {
            if( (mDepthLimit > 0) && (depth > mDepthLimit) )
                break;

            Timer levelTimer;
            this->RunToDepthForCpu( depth );

            i64 currLevelElapsed = levelTimer.GetElapsedMs();
            if( !mExitSearch && (mTargetTime != NO_TIME_LIMIT) )
            {
                if( currLevelElapsed > 500 )
                {
                    if( mSearchElapsed.GetElapsedMs() + (currLevelElapsed * 4) > mTargetTime )
                    {
                        printf( "info string bailing at level %d\n", depth );
                        break;
                    }
                }
            }

            depth++;

            prevLevelElapsed = currLevelElapsed;
        }

        if( mPrintBestMove )
        {
            this->PrintResult();
            mPrintBestMove = false;
        }
    }

    void TimerThread()
    {
        for( ;; )
        {
            i64 elapsed   = mSearchElapsed.GetElapsedMs();
            i64 remaining = mTargetTime - elapsed;

            if( remaining < MIN_TIME_SLICE )
                break;

            i64 sleepTime = (remaining < MAX_TIME_SLICE)? remaining : MAX_TIME_SLICE;

            PlatSleep( (int) sleepTime );
        }        

        mExitSearch = true;
    }

    void JoinAllThreads()
    {
        mExitSearch = true;

        while( mThreadsRunning > 0 )
        {
            mThreadsDone.Wait();
            mThreadsRunning--;
        }

        mExitSearch = false;
    }

    void PrintValidMoves()
    {
        MoveList valid;
        valid.FindMoves( mRoot );

        printf( "info string validmoves " );
        FEN::PrintMoveList( valid );
        printf( "\n" );
    }

    void PrintResult()
    {
        if( !mPrintedMove )
        {
            i64 elapsed = mSearchElapsed.GetElapsedMs();

            printf( "bestmove " );
            FEN::PrintMoveSpec( mStorePv->mMove[0] );
            printf( "\n" );
            fflush( stdout );

            mPrintedMove = true;
        }
    }

    int CalcTargetTime()
    {
        int moveTimeLimit   = mConfig.mTimeLimit;
        int totalTimeLeft   = mRoot.mWhiteToMove? mConfig.mWhiteTimeLeft : mConfig.mBlackTimeLeft;
        int timeBonus       = mRoot.mWhiteToMove? mConfig.mWhiteTimeInc  : mConfig.mBlackTimeInc;

        if( !moveTimeLimit && !totalTimeLeft && !timeBonus )
            return( NO_TIME_LIMIT );

        int targetTime = 0;

        if( totalTimeLeft )
        {
            if( timeBonus )
            {
                targetTime = timeBonus;

                if( mRoot.mFullmoveNum < 4 )     
                    targetTime += (int) mRoot.mFullmoveNum * 1000;
                else
                    targetTime += totalTimeLeft / 10;
            }
            else
            {
                struct TimePolicy
                {
                    int     remaining;
                    int     time;
                    int     depth;
                };
            
                TimePolicy policyTable[] =
                {
                    {   60000,  5000,   0   },  // > 60s left? think 5s
                    {   30000,  3000,   0   },  // > 30s left? think 3s
                    {   15000,  2000,   0   },  // > 15s left? think 2s
                    {   10000,  1000,   0   },  // > 10s left? think 1s
                    {    5000,   500,   0   },  // >  5s left? think 0.5s
                    {    3000,   500,   9   },  // >  3s left? think 0.5s and limit depth to 9
                    {    2000,   500,   7   },  // >  2s left? think 0.5s and limit depth to 7
                    {    1000,   500,   5   },  // >  1s left? think 0.5s and limit depth to 5
                    {       0,     0,   3   },  //  PANIC NOW: limit depth to 3
                };

                TimePolicy* policy = policyTable;
                while( policy->remaining > totalTimeLeft )
                    policy++;

                targetTime = policy->time;
                if( policy->depth )
                    mDepthLimit = Min( mDepthLimit, policy->depth );
            }
        }

        if( moveTimeLimit )
            targetTime = targetTime? Min( targetTime, moveTimeLimit ) : moveTimeLimit;

        if( totalTimeLeft )
            targetTime = Min( targetTime, totalTimeLeft - LAG_SAFETY_BUFFER );

        return( targetTime );
    }

    template< int POPCNT, typename SIMD >
    EvalTerm NegaMax( const Position& pos, const MoveMap& moveMap, EvalTerm score, int ply, int depth, EvalTerm alpha, EvalTerm beta, MoveList* pv_new, bool onPvPrev )
    {
        const int LANES = SimdWidth< SIMD >::LANES;

        mMetrics.mNodesTotal++;
        mMetrics.mNodesAtPly[ply]++;

        if( mExitSearch )
            return( EVAL_SEARCH_ABORTED );

        if( POPCNT )
            mHashTable.Prefetch( pos.mHash );

        MoveList moves;
        moves.UnpackMoveMap( pos, moveMap );

        if( moves.mCount == 0 )
            return( EVAL_NO_MOVES );

        if( depth <= 0 )
        {
            alpha = Max( alpha, score );
            if( alpha >= beta )
                return( beta );

            moves.DiscardQuietMoves();
            if( moves.mCount == 0 )
                return( score );
        }

        if( depth >= 0 )
        {
            TableEntry tt;

            mHashTable.Load( pos.mHash, tt );
            mMetrics.mHashLookupsAtPly[ply]++;

            u32 verify = (u32) (pos.mHash >> 40);
            if( tt.mHashVerify == verify )
            {
                mMetrics.mHashHitsAtPly[ply]++;

                bool    samePlayer          = (pos.mWhiteToMove != 0) == tt.mWhiteMove;
                bool    failedHighBefore    = samePlayer? tt.mFailHigh : tt.mFailLow;
                int     lowerBoundBefore    = samePlayer? tt.mScore    : -tt.mScore;
                int     depthBefore         = tt.mDepth;

                if( failedHighBefore && (lowerBoundBefore >= beta) && (depthBefore >= depth) )
                    return( beta );

                if( !(tt.mFailHigh || tt.mFailLow) )
                    moves.MarkSpecialMoves( tt.mBestSrc, tt.mBestDest, TT_BEST_MOVE );
            }
        }

        if( onPvPrev && (mStorePv->mCount > ply) )
        {
            MoveSpec& pvMove = mStorePv->mMove[ply];
            moves.MarkSpecialMoves( pvMove.mSrc, pvMove.mDest, PRINCIPAL_VARIATION );
        }

        EvalTerm    weights[EVAL_TERMS];
        int         movesTried  = 0;
        int         simdIdx     = LANES - 1;
        EvalTerm    bestScore   = alpha;
        MoveSpec    bestMove;

        float gamePhase = mEvaluator.CalcGamePhase< POPCNT >( pos );
        mEvaluator.GenerateWeights( weights, gamePhase );

        MoveSpec PIGEON_ALIGN_SIMD childSpec[LANES];
        Position PIGEON_ALIGN_SIMD childPos[LANES];
        MoveMap  PIGEON_ALIGN_SIMD childMoveMap[LANES];
        u64      PIGEON_ALIGN_SIMD childScore[LANES];

        while( (movesTried < moves.mCount) && (bestScore < beta) )
        {
            simdIdx++;
            if( simdIdx >= LANES )
            {
                MoveSpecT< SIMD >   simdSpec;
                PositionT< SIMD >   simdPos;
                MoveMapT< SIMD >    simdMoveMap;
                SIMD                simdScore;

                for( int idxLane = 0; idxLane < LANES; idxLane++ )
                {
                    if( moves.mTried >= moves.mCount )
                        break;

                    int idxMove = moves.ChooseBestUntried();
                    childSpec[idxLane] = moves.mMove[idxMove];

                    SimdInsert( simdSpec.mSrc,  childSpec[idxLane].mSrc,  idxLane );
                    SimdInsert( simdSpec.mDest, childSpec[idxLane].mDest, idxLane );
                    SimdInsert( simdSpec.mType, childSpec[idxLane].mType, idxLane );

                    mMetrics.mNodesTotalSimd++;
                }

                simdPos.Broadcast( pos );
                simdPos.Step( simdSpec );
                simdPos.CalcMoveMap( &simdMoveMap );
                simdScore = mEvaluator.Evaluate< POPCNT, SIMD >( simdPos, weights );

                *((SIMD*) childScore) = simdScore;

                Unswizzle< SIMD >( &simdPos,     childPos );
                Unswizzle< SIMD >( &simdMoveMap, childMoveMap );

                simdIdx = 0;
            }

            MoveList pv_child;
            EvalTerm score = -this->NegaMax< POPCNT, SIMD >( 
                childPos[simdIdx], childMoveMap[simdIdx], (EvalTerm) childScore[simdIdx], 
                ply + 1, depth - 1, -beta, -bestScore, &pv_child, 
                (childSpec[simdIdx].mType == PRINCIPAL_VARIATION) );

            if( score > bestScore )
            {
                bestScore   = score;
                bestMove    = childSpec[simdIdx];

                pv_new->mCount = 1;
                pv_new->mMove[0] = bestMove;
                pv_new->Append( pv_child );
            }

            movesTried++;
        }

        if( mExitSearch )
            return( EVAL_SEARCH_ABORTED );

        bool        failedHigh  = (bestScore > beta);
        bool        failedLow   = (bestScore == alpha);
        EvalTerm    result      = failedHigh? beta : (failedLow? alpha : bestScore);

        if( depth > 0 )
        {
            TableEntry tt;

            tt.mHashVerify  = pos.mHash >> 40;
            tt.mDepth       = depth;
            tt.mScore       = result;
            tt.mBestSrc     = (failedLow || failedHigh)? 0 : bestMove.mSrc;
            tt.mBestDest    = (failedLow || failedHigh)? 0 : bestMove.mDest;
            tt.mFailLow     = failedLow;
            tt.mFailHigh    = failedHigh;
            tt.mWhiteMove   = pos.mWhiteToMove? true : false;

            mHashTable.Store( pos.mHash, tt );
        }

        return( result );
    }

    void RunToDepthForCpu( int depth )
    {
        switch( mCpuLevel )
        {
#if PIGEON_ENABLE_SSE2
        case  CPU_SSE2: this->RunToDepth< simd2_sse2 >( depth ); break;
#endif
#if PIGEON_ENABLE_SSE4
        case  CPU_SSE4: this->RunToDepth< simd2_sse4 >( depth ); break;
#endif
#if PIGEON_ENABLE_AVX2
        case  CPU_AVX2: this->RunToDepth< simd4_avx2 >( depth ); break;
#endif
        default:        this->RunToDepth< u64 >(        depth ); break;
        }
    }

    template< typename SIMD >
    void RunToDepth( int depth )
    {
        if( mUsePopcnt )
            this->RunToDepth< ENABLE_POPCNT, SIMD >( depth );
        else
            this->RunToDepth< DISABLE_POPCNT, SIMD >( depth );
    }

    template< int POPCNT, typename SIMD >
    void RunToDepth( int depth )
    {
        mMetrics.Clear();

        MoveList pv;
        Timer    searchTime;

        MoveMap     moveMap;
        EvalTerm    rootWeights[EVAL_TERMS];
        EvalTerm    rootScore;

        mRoot.CalcMoveMap( &moveMap );

        float gamePhase = mEvaluator.CalcGamePhase< POPCNT >( mRoot );
        mEvaluator.GenerateWeights( rootWeights, gamePhase );
        rootScore = (EvalTerm) mEvaluator.Evaluate< POPCNT >( mRoot, rootWeights );

        EvalTerm score = this->NegaMax< POPCNT, SIMD >( mRoot, moveMap, rootScore, 0, depth, -EVAL_MAX, EVAL_MAX, &pv, true );
        if( mExitSearch )
            return;

        i64 elapsed     = Max( searchTime.GetElapsedMs(), (i64) 1 );
        i64 nps         = mMetrics.mNodesTotal * 1000L / elapsed;
        int hashfull    = (int) (mHashTable.EstimateUtilization() * 1000);
        int seldepth    = 0;

        for( seldepth = MAX_METRICS_DEPTH - 1; seldepth > depth; seldepth-- )
             if( mMetrics.mNodesAtPly[seldepth] )
                 break;

        printf( "info " );
        printf( "depth %d ",        depth );
        printf( "seldepth %d ",     seldepth );
        printf( "score cp %d ",     score );
        printf( "hashfull %d ",     hashfull );
        printf( "nodes %"PRId64" ", mMetrics.mNodesTotal);
        printf( "time %d ",         mSearchElapsed.GetElapsedMs() );
        printf( "nps %"PRId64" ",   nps );
        printf( "pv " );            FEN::PrintMoveList( pv );
        printf( "\n" );

        fflush( stdout );

        if( mDebugMode )
            printf( "info string simdnodes %"PRId64"\n", mMetrics.mNodesTotalSimd );

        *mStorePv   = pv;
        mValuePv    = score;
    }
};




#endif // PIGEON_ENGINE_H__
};
