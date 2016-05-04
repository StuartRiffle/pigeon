// engine.h - PIGEON CHESS ENGINE (c) 2012-2016 Stuart Riffle
        
namespace Pigeon {
#ifndef PIGEON_ENGINE_H__
#define PIGEON_ENGINE_H__


class Engine
{
    HashTable               mHashTable;                 // Transposition table
    Position                mRoot;                      // The root of the search tree (the "current position")
    SearchConfig            mConfig;                    // Search parameters
    SearchMetrics           mMetrics;                   // Runtime metrics
    Evaluator               mEvaluator;                 // Evaluation weights
    OpeningBook             mOpeningBook;               // Placeholder opening book implementation
    MoveList                mBestLine;                  // Best line found in the search
    MoveList                mPvDepth[METRICS_DEPTH];    // Best line found at 
    MoveList*               mStorePv;                   // Target for PV in active search
    EvalTerm                mValuePv;                   // The evaluation of *mStorePv
    int                     mTableSize;                 // Transposition table size (in megs)
    int                     mTargetTime;                // Time to stop current search
    int                     mDepthLimit;                // Depth limit for current search (not counting quiesence)
    Timer                   mSearchElapsed;             // Time elapsed since the "go" command
    volatile bool           mExitSearch;                // Flag to terminate search threads immediately
    int                     mThreadsRunning;            // Number of worker threads currently running
    Semaphore               mThreadsDone;               // Semaphore to help gather up completed threads                                                
    bool                    mPrintBestMove;             // Output best move while searching
    bool                    mPrintedMove;               // Make sure only one "bestmove" is output per "go" 
    bool                    mDebugMode;                 // This currently does nothing
    bool                    mUsePopcnt;                 // Enable use of the hardware POPCNT instruction
    bool                    mAllowEarlyMove;            // Bail on iterative deepening if the next level will take too long
    int                     mCpuLevel;                  // A CPU_* enum value to select the code path
	int					    mNumHelperThreads;			// Number of lazy SMP threads to spawn
    EvalWeight              mRootWeights[EVAL_TERMS];   // Evaluation weights calculated at root position
    bool                    mUseRootWeights;            // When false, recalculate weights at every level
    bool                    mUseOpeningBook;            // Enable use of the opening book
    int                     mHistoryTable[2][64][64];   // Indexed as [whiteToMove][dest][src]
    std::map< u64, int >    mPositionReps;              // Indexed by hash, detects repetitions to avoid (unwanted) draw

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
        mUsePopcnt          = PlatDetectPopcnt();
        mCpuLevel           = PlatDetectCpuLevel();
        mAllowEarlyMove     = true;
        mNumHelperThreads   = 0;
        mUseRootWeights     = true;
        mUseOpeningBook     = OWNBOOK_DEFAULT;

        mHashTable.SetSize( mTableSize );
        mOpeningBook.Init();               
        PlatClearMemory( mHistoryTable, sizeof( mHistoryTable ) );
    }

    ~Engine()
    {
        this->Stop();
    }

    void Reset()
    {
        this->Stop();
        mRoot.Reset();
        mEvaluator.EnableOpening( true );
        mHashTable.Clear();
        mPositionReps.clear();

        PlatClearMemory( mHistoryTable, sizeof( mHistoryTable ) );
    }

    void SetPosition( const Position& pos )
    {
        this->Reset();
        mRoot = pos;
        mEvaluator.EnableOpening( false );

        mRoot.mHash = mRoot.CalcHash();
        mPositionReps[mRoot.mHash]++;
    }

    Position GetPosition() const
    {
        return( mRoot );    
    }

    void SetHashTableSize( int megs )
    {
        mTableSize = megs;
    }

    void EnableOpeningBook( bool enabled )
    {
        mUseOpeningBook = enabled;
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

	void SetThreadCount( int count )
	{
        mNumHelperThreads = (count > 1)? (count - 1) : 0;
	}

    void OverrideCpuLevel( int level )
    {
        mCpuLevel = level;
    }

    void OverridePopcnt( bool enabled )
    {
        mUsePopcnt = enabled;
    }    

    void LoadWeightParam( const char* name, float openingVal, float midgameVal, float endgameVal ) 
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
                mPositionReps[mRoot.mHash]++;

                if( mPositionReps[mRoot.mHash] > 1 )
                    printf( "info string repeated position\n" );
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
        this->Stop();
        mConfig = *conf;

        mSearchElapsed.Reset();

        MoveList valid;
        valid.FindMoves( mRoot );

        if( mUseOpeningBook )
        {
            const char* movetext = mOpeningBook.GetBookMove( mRoot );
            if( movetext != NULL )
            {
                if( mDebugMode )
                    printf( "info string opening book says %s\n", movetext );

                MoveSpec spec;
                FEN::StringToMoveSpec( movetext, spec );

                int idx = valid.LookupMove( spec );
                if( idx >= 0 )
                {
                    printf( "bestmove %s\n", movetext );
                    fflush( stdout );
                    return;
                }
            }
        }

        if( valid.mCount == 0 )
        {
            printf( "info string ERROR: no moves available at position " );
            FEN::PrintPosition( mRoot );
            printf( "\n" );
            return;
        }

        mBestLine.Clear();
        mMetrics.Clear();

        mStorePv        = &mBestLine;
        mValuePv        = 0;
        mDepthLimit     = mConfig.mDepthLimit;
        mTargetTime     = this->CalcTargetTime();
        mExitSearch     = false;
        mPrintBestMove  = true;
        mPrintedMove    = false;

        this->RunToDepthForCpu( 1, true );

        for( int i = 0; i < mNumHelperThreads; i++ )
        {
            PlatSpawnThread( &Engine::HelperThreadProc, this );
            mThreadsRunning++;
        }

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

    void PrintValidMoves()
    {
        MoveList valid;
        valid.FindMoves( mRoot );

        printf( "info string validmoves " );
        while( valid.mTried < valid.mCount )
        {
            int idx = valid.ChooseBestUntried();
            const MoveSpec& spec = valid.mMove[idx];

            FEN::PrintMoveSpec( spec );
            printf( "/%d ", spec.mType );
        }

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

	static void* HelperThreadProc( void* param )
	{
		Engine* engine = reinterpret_cast< Engine* >( param );
		engine->LazyHelperThread();
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
        int depth = 2;
        i64 prevLevelElapsed = 0;

        while( !mExitSearch )
        {
            if( (mDepthLimit > 0) && (depth > mDepthLimit) )
                break;

            Timer levelTimer;
            this->RunToDepthForCpu( depth, true );

            if( mAllowEarlyMove )
            {
                i64 currLevelElapsed = levelTimer.GetElapsedMs();
                if( !mExitSearch && (mTargetTime != NO_TIME_LIMIT) )
                {
                    if( currLevelElapsed > 500 )
                    {
                        if( mSearchElapsed.GetElapsedMs() + (currLevelElapsed * 4) > mTargetTime )
                        {
                            if( mDebugMode )
                                printf( "info string bailing at level %d\n", depth );

						    mExitSearch = true;
                            break;
                        }
                    }
                }

                prevLevelElapsed = currLevelElapsed;
            }

            
            //TODO: only under short time controls

            if( (depth < METRICS_DEPTH) && (depth > 7) )
            {
                bool sameMove = true;

                for( int i = depth - 2; i <= depth; i++ )
                    if( mPvDepth[i].mMove[0] != mPvDepth[depth].mMove[0] )
                        sameMove = false;

                if( sameMove )
                    break;
            }
            


            depth++;

        }

        if( mPrintBestMove )
        {
            this->PrintResult();
            mPrintBestMove = false;
        }
    }

	void LazyHelperThread()
	{
		int depth = 5;

		while( !mExitSearch )
        {
			this->RunToDepthForCpu( depth );
            depth++;
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
                    {   60000,  5000,   0   },  
                    {   30000,  3000,   0   },  
                    {   20000,  2000,   0   },  
                    {   10000,  1000,   0   },  
                    {    5000,   800,   0   },  
                    {    3000,   500,   0   },  
                    {    2000,   500,   7   },  
                    {    1000,   500,   5   },  
                    {       0,   200,   3   },  
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

    int ChooseNextMove( MoveList& moves, int whiteToMove )
    {
        int best = moves.mTried;

        for( int idx = best + 1; idx < moves.mCount; idx++ )
        {
            MoveSpec& bestMove = moves.mMove[best];
            MoveSpec& currMove = moves.mMove[idx];

            if( currMove.mType < bestMove.mType )
                continue;

            if( currMove.mType == bestMove.mType )
                if( mHistoryTable[whiteToMove][currMove.mDest][currMove.mSrc] < mHistoryTable[whiteToMove][bestMove.mDest][bestMove.mSrc] )
                    continue;   

            best = idx;
        }

        Exchange( moves.mMove[moves.mTried], moves.mMove[best] );
        return( moves.mTried++ );
    }

    template< int POPCNT, typename SIMD >
    EvalTerm NegaMax( const Position& pos, const MoveMap& moveMap, EvalTerm score, int ply, int depth, EvalTerm alpha, EvalTerm beta, MoveList* pv_new, bool onPvPrev )
    {
        const int LANES = SimdWidth< SIMD >::LANES;

        mMetrics.mNodesTotal++;
        mMetrics.mNodesAtPly[ply]++;

        if( depth < 1 )
        {
            alpha = Max( alpha, score );

            if( alpha >= beta )
                return( beta );
        }

        if( mExitSearch )
            return( EVAL_SEARCH_ABORTED );

        if( POPCNT )
            mHashTable.Prefetch( pos.mHash );

        bool inCheck = (moveMap.IsInCheck() != 0);

        MoveList moves;
        moves.UnpackMoveMap( pos, moveMap );

        if( moves.mCount == 0 )
            return( inCheck? EVAL_CHECKMATE : EVAL_STALEMATE );

        if( depth < 1 )
        {
            if( !inCheck )
                moves.DiscardMovesBelow( CAPTURE_LOSING );

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

                bool        samePlayer          = (pos.mWhiteToMove != 0) == tt.mWhiteMove;
                bool        failedHighBefore    = samePlayer? tt.mFailHigh : tt.mFailLow;
                EvalTerm    lowerBoundBefore    = samePlayer? tt.mScore    : -tt.mScore;
                int         depthBefore         = tt.mDepth;

                if( failedHighBefore && (lowerBoundBefore >= beta) && (depthBefore >= depth) )
                    return( beta );

                moves.MarkSpecialMoves( tt.mBestSrc, tt.mBestDest, TT_BEST_MOVE );
            }
        }

        if( onPvPrev && (mStorePv->mCount > ply) )
        {
            MoveSpec& pvMove = mStorePv->mMove[ply];
            moves.MarkSpecialMoves( pvMove.mSrc, pvMove.mDest, PRINCIPAL_VARIATION );
        }

        EvalWeight  currWeights[EVAL_TERMS];
        EvalWeight* weights     = mRootWeights;
        int         movesTried  = 0;
        int         simdIdx     = LANES - 1;
        bool        nullSearch  = false;
        EvalTerm    bestScore   = alpha;
        MoveSpec    bestMove;

        if( !mUseRootWeights )
        {
            float gamePhase = mEvaluator.CalcGamePhase< POPCNT >( pos );

            mEvaluator.GenerateWeights( currWeights, gamePhase );
            weights = currWeights;
        }

        MoveSpec PIGEON_ALIGN_SIMD childSpec[LANES];
        Position PIGEON_ALIGN_SIMD childPos[LANES];
        MoveMap  PIGEON_ALIGN_SIMD childMoveMap[LANES];
        EvalTerm PIGEON_ALIGN_SIMD childScore[LANES];

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

                    int idxMove = this->ChooseNextMove( moves, (int) pos.mWhiteToMove );
                    childSpec[idxLane] = moves.mMove[idxMove];

                    SimdInsert( simdSpec.mSrc,  childSpec[idxLane].mSrc,  idxLane );
                    SimdInsert( simdSpec.mDest, childSpec[idxLane].mDest, idxLane );
                    SimdInsert( simdSpec.mType, childSpec[idxLane].mType, idxLane );

                    mMetrics.mNodesTotalSimd++;
                }

                simdPos.Broadcast( pos );
                simdPos.Step( simdSpec );
                simdPos.CalcMoveMap( &simdMoveMap );
                simdScore = mEvaluator.Evaluate< POPCNT, SIMD >( simdPos, simdMoveMap, weights );

                Unswizzle< SIMD >( &simdPos,     childPos );
                Unswizzle< SIMD >( &simdMoveMap, childMoveMap );

				u64 PIGEON_ALIGN_SIMD unpackScore[LANES];
				*((SIMD*) unpackScore) = simdScore;

				for( int idxLane = 0; idxLane < LANES; idxLane++ )
					childScore[idxLane] = (EvalTerm) unpackScore[idxLane];

				simdIdx = 0;
            }

            bool allowMove = true;

            if( ply == 0 )
            {
                // TODO: make sure that this does not eliminate all valid moves! 

                bool repeatedPosition = (mPositionReps[childPos[simdIdx].mHash] > 1);
                bool notReadyToDraw   = (score > -ALLOW_REP_SCORE);

                if( repeatedPosition && notReadyToDraw && !inCheck )
                {
                    allowMove = false;

                    if( mDebugMode )
                    {
                        printf( "info string preventing " );
                        FEN::PrintMoveSpec( childSpec[simdIdx] );
                        printf( " to avoid draw by repetition\n" );
                    }
                }
            }

            if( allowMove )
            {
                MoveList pv_child;
                EvalTerm subScore;
            
                bool fullSearch = true;

                if( nullSearch )
                {
                    subScore = -this->NegaMax< POPCNT, SIMD >( 
                        childPos[simdIdx], childMoveMap[simdIdx], childScore[simdIdx], ply + 1, depth - 1, -(bestScore + 1), -bestScore, 
                        &pv_child, (childSpec[simdIdx].mType == PRINCIPAL_VARIATION) );
            
                    fullSearch = (subScore > bestScore) && (subScore < beta);
                }

                if( fullSearch )
                {
                    subScore = -this->NegaMax< POPCNT, SIMD >( 
                        childPos[simdIdx], childMoveMap[simdIdx], childScore[simdIdx], ply + 1, depth - 1, -beta, -bestScore, 
                        &pv_child, (childSpec[simdIdx].mType == PRINCIPAL_VARIATION) );
                }

                if( subScore > bestScore )
                {
                    bestScore   = subScore;
                    bestMove    = childSpec[simdIdx];
                    nullSearch  = true;

                    pv_new->mCount = 1;
                    pv_new->mMove[0] = bestMove;
                    pv_new->Append( pv_child );

                    if( depth > 2 )
                        mHistoryTable[pos.mWhiteToMove][bestMove.mDest][bestMove.mSrc] += (depth * depth);

                    if( depth < 1 )
                        break;
                }
            }

            if( (ply < METRICS_DEPTH) && (movesTried < METRICS_MOVES) )
                mMetrics.mMovesTriedByPly[ply][movesTried]++;

            movesTried++;

            //if( depth < -1 )
            //    break;

        }

        if( mExitSearch )
            return( EVAL_SEARCH_ABORTED );

        bool        failedHigh  = (bestScore >= beta);
        bool        failedLow   = (bestScore == alpha);
        EvalTerm    result      = failedHigh? beta : (failedLow? alpha : bestScore);

        //if( failedHigh )
        //    mHistoryTable[pos.mWhiteToMove][bestMove.mDest][bestMove.mSrc] += (ply * ply);

        if( depth > 0 )
        {
            TableEntry tt;

            tt.mHashVerify  = pos.mHash >> 40;
            tt.mDepth       = depth;
            tt.mScore       = result;
            tt.mBestSrc     = bestMove.mSrc;
            tt.mBestDest    = bestMove.mDest;
            tt.mFailLow     = failedLow;
            tt.mFailHigh    = failedHigh;
            tt.mWhiteMove   = pos.mWhiteToMove? true : false;

            mHashTable.Store( pos.mHash, tt );
        }

        return( result );
    }

    void RunToDepthForCpu( int depth, bool printPv = false )
    {
        switch( mCpuLevel )
        {
#if PIGEON_ENABLE_SSE2
        case  CPU_SSE2: this->RunToDepth< simd2_sse2 >( depth, printPv ); break;
#endif
#if PIGEON_ENABLE_SSE4
        case  CPU_SSE4: this->RunToDepth< simd2_sse4 >( depth, printPv ); break;
#endif
#if PIGEON_ENABLE_AVX2
        case  CPU_AVX2: this->RunToDepth< simd4_avx2 >( depth, printPv ); break;
#endif
        default:        this->RunToDepth< u64 >(        depth, printPv ); break;
        }
    }

    template< typename SIMD >
    void RunToDepth( int depth, bool printPv )
    {
        if( mUsePopcnt )
            this->RunToDepth< ENABLE_POPCNT, SIMD >(  depth, printPv );
        else
            this->RunToDepth< DISABLE_POPCNT, SIMD >( depth, printPv );
    }

    template< int POPCNT, typename SIMD >
    void RunToDepth( int depth, bool printPv )
    {
        MoveList	pv;
        Timer		searchTime;
        MoveMap     moveMap;
        MoveList    moves;
        EvalTerm    rootScore;

        mRoot.CalcMoveMap( &moveMap );
        moves.UnpackMoveMap( mRoot, moveMap );

        float gamePhase = mEvaluator.CalcGamePhase< POPCNT >( mRoot );
        mEvaluator.GenerateWeights( mRootWeights, gamePhase );
        rootScore = (EvalTerm) mEvaluator.Evaluate< POPCNT >( mRoot, moveMap, mRootWeights );


		searchTime.Reset();
        EvalTerm score = this->NegaMax< POPCNT, SIMD >( mRoot, moveMap, rootScore, 0, depth, -EVAL_MAX, EVAL_MAX, &pv, true );

        if( mExitSearch )
            return;

        if( printPv )
        {
            i64 elapsed     = Max( searchTime.GetElapsedMs(), (i64) 1 );
            i64 nps         = mMetrics.mNodesTotal * 1000L / elapsed;
            int hashfull    = (int) (mHashTable.EstimateUtilization() * 1000);
            int seldepth    = 0;

            for( seldepth = METRICS_DEPTH - 1; seldepth > depth; seldepth-- )
                 if( mMetrics.mNodesAtPly[seldepth] )
                     break;

            //if( score == EVAL_SEARCH_ABORTED )
            //    score = rootScore;

            printf( "info " );
            printf( "depth %d ",            depth );
            printf( "seldepth %d ",         seldepth );
            printf( "score cp %d ",         score );
            printf( "hashfull %d ",         hashfull );
            printf( "nodes %" PRId64 " ",   mMetrics.mNodesTotal);
            printf( "time %d ",             (int) mSearchElapsed.GetElapsedMs() );
            printf( "nps %" PRId64 " ",     nps );
            printf( "pv " );                FEN::PrintMoveList( pv );
            printf( "\n" );

            fflush( stdout );

            if( mDebugMode )
            { 
                printf( "info string gamephase %.2f simdnodes %" PRId64 "\n", gamePhase, mMetrics.mNodesTotalSimd );


            }

            *mStorePv   = pv;
            mValuePv    = score;

            if( depth < METRICS_DEPTH )
                mPvDepth[depth] = pv;
        }
    }
};




#endif // PIGEON_ENGINE_H__
};
