// engine.h - PIGEON CHESS ENGINE (c) 2012-2016 Stuart Riffle
        
namespace Pigeon {
#ifndef PIGEON_ENGINE_H__
#define PIGEON_ENGINE_H__

PDECL class EngineBase                                        
{
protected:
    HashTable               mHashTable;                 ///< Transposition table
    Position                mRoot;                      ///< The root of the search tree (the "current position")
    SearchConfig            mConfig;                    ///< Search parameters
    SearchMetrics           mMetrics;                   ///< Runtime metrics
    Evaluator               mEvaluator;                 ///< Evaluation weight calculator
    EvalWeight              mRootWeights[EVAL_TERMS];   ///< Evaluation weights for position mRoot
    MaterialTable           mMaterialTable[2];          ///< Material value for each piece/square combination, indexed by pos.mWhiteToMove
    OpeningBook             mOpeningBook;               ///< Placeholder opening book implementation
    MoveList                mBestLine;                  ///< Best line found in the search
    MoveList                mPvDepth[METRICS_DEPTH];    ///< Best line found at 
    MoveList*               mStorePv;                   ///< Target for PV in active search
    EvalTerm                mValuePv;                   ///< The evaluation of *mStorePv
};

PDECL class Engine : EngineBase
{
    int                     mTableSize;                 ///< Transposition table size (in megs)
    int                     mTargetTime;                ///< Time to stop current search
    int                     mDepthLimit;                ///< Depth limit for current search (not counting quiesence)
    Timer                   mSearchElapsed;             ///< Time elapsed since the "go" command
    volatile bool           mExitSearch;                ///< Flag to terminate search threads immediately
    int                     mThreadsRunning;            ///< Number of worker threads currently running
    Semaphore               mThreadsDone;               ///< Semaphore to help gather up completed threads                                                
    bool                    mPrintBestMove;             ///< Output best move while searching
    bool                    mPrintedMove;               ///< Make sure only one "bestmove" is output per "go" 
    bool                    mDebugMode;                 ///< This currently does nothing
    int                     mCpuLevel;                  ///< A CPU_* enum value that reflects the hardware capabilities
    bool                    mPopcntSupported;           ///< True if the CPU can do POPCNT
    u8                      mHistoryTable[2][64][64];   ///< Indexed as [whiteToMove][dest][src]
    std::map< u64, int >    mPositionReps;              ///< Indexed by hash, detects repetitions to avoid (unwanted) draw
    int                     mOptions[OPTION_COUNT];     ///< Runtime options exposed via UCI

public:
    Engine()
    {
        mConfig.Clear();
        mMetrics.Clear();
        mRoot.Reset();

        mStorePv            = NULL;
        mValuePv            = EVAL_MAX;
        mTargetTime         = NO_TIME_LIMIT;
        mDepthLimit         = 0;
        mExitSearch         = false;
        mThreadsRunning     = 0;
        mPrintBestMove      = false;
        mPrintedMove        = false;
        mDebugMode          = false;
        mPopcntSupported    = PlatDetectPopcnt();
        mCpuLevel           = PlatDetectCpuLevel();

        mOpeningBook.Init();               
        PlatClearMemory( mHistoryTable, sizeof( mHistoryTable ) );
        PlatClearMemory( mOptions, sizeof( mOptions ) );

        mOptions[OPTION_HASH_SIZE]      = TT_MEGS_DEFAULT;
        mOptions[OPTION_CLEAR_HASH]     = 0;
        mOptions[OPTION_OWN_BOOK]       = OWNBOOK_DEFAULT? 1 : 0;
        mOptions[OPTION_NUM_THREADS]    = PlatDetectCpuCores();
        mOptions[OPTION_ENABLE_SIMD]    = 1;
        mOptions[OPTION_ENABLE_POPCNT]  = 1;
        mOptions[OPTION_ENABLE_CUDA]    = 1;
        mOptions[OPTION_EARLY_MOVE]     = 1;
        mOptions[OPTION_GPU_HASH_SIZE]  = TT_MEGS_DEFAULT;

        mHashTable.SetSize( mOptions[OPTION_HASH_SIZE] );
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

    void OverrideCpuLevel( int level )
    {
        mCpuLevel = level;
    }

    void SetOption( int idx, int value )
    {
        if( (idx >= 0) && (idx < OPTION_COUNT) )
            mOptions[idx] = value;
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

                if( mDebugMode )
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

        if( mOptions[OPTION_OWN_BOOK] )
        {
            const char* movetext = mOpeningBook.GetBookMove( mRoot );
            if( movetext != NULL )
            {
                MoveSpec spec;
                FEN::StringToMoveSpec( movetext, spec );

                int idx = valid.LookupMove( spec );
                if( idx >= 0 )
                {
                    if( mDebugMode )
                        printf( "info string using book move %s\n", movetext );

                    printf( "bestmove %s\n", movetext );
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

        PlatClearMemory( mHistoryTable, sizeof( mHistoryTable ) );

        mStorePv        = &mBestLine;
        mValuePv        = 0;
        mDepthLimit     = mConfig.mDepthLimit;
        mTargetTime     = this->CalcTargetTime();
        mExitSearch     = false;
        mPrintBestMove  = true;
        mPrintedMove    = false;

        this->RunToDepthForCpu( 1, true );

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

        if( mDebugMode && (mTargetTime != NO_TIME_LIMIT) )
            printf( "info string DEBUG: available time %" PRId64 "\n", mTargetTime - mSearchElapsed.GetElapsedMs() );

        while( !mExitSearch )
        {
            if( (mDepthLimit > 0) && (depth > mDepthLimit) )
                break;

            Timer levelTimer;
            this->RunToDepthForCpu( depth, true );

            if( mOptions[OPTION_EARLY_MOVE] )
            {
                i64 currLevelElapsed = levelTimer.GetElapsedMs();
                if( !mExitSearch && (mTargetTime != NO_TIME_LIMIT) )
                {
                    if( (depth > 4) && (currLevelElapsed > 100) )
                    {
                        float levelRatio = currLevelElapsed * 1.0f / prevLevelElapsed;
                        levelRatio *= 1.3f;

                        i64 nextLevelExpected = (i64) (currLevelElapsed * levelRatio);

                        if( mDebugMode )
                            printf( "info string DEBUG: elapsed %" PRId64 ", prev %" PRId64 ", expect %" PRId64 ", remaining %" PRId64 "\n", currLevelElapsed, prevLevelElapsed, nextLevelExpected, mTargetTime - mSearchElapsed.GetElapsedMs() );

                        if( mSearchElapsed.GetElapsedMs() + nextLevelExpected > mTargetTime )
                        {
                            if( mDebugMode )
                                printf( "info string DEBUG: bailing at level %d\n", depth );

						    mExitSearch = true;
                            break;
                        }
                    }
                }

                prevLevelElapsed = currLevelElapsed;
            }

            if( mDebugMode && mExitSearch )
                printf( "info string DEBUG: out of time at %" PRId64 "\n", mSearchElapsed.GetElapsedMs() );
            
            //TODO: only under short time controls

            //if( (depth < METRICS_DEPTH) && (depth > 7) )
            //{
            //    bool sameMove = true;
            //
            //    for( int i = depth - 2; i <= depth; i++ )
            //        if( mPvDepth[i].mMove[0] != mPvDepth[depth].mMove[0] )
            //            sameMove = false;
            //
            //    if( sameMove )
            //        break;
            //}
            


            depth++;

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
                    {   60000,  3000,   0   },  
                    {   30000,  2000,   0   },  
                    {   20000,  1500,   0   },  
                    {   10000,  1000,   8   },  
                    {    5000,   800,   7   },  
                    {    3000,   500,   7   },  
                    {    2000,   400,   7   },  
                    {    1000,   300,   5   },  
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

            if( currMove.mFlags < bestMove.mFlags )
                continue;

            if( currMove.mType < bestMove.mType )
                continue;

            //if( (currMove.mType == bestMove.mType) && (currMove.mFlags == bestMove.mFlags) )
//            if( currMove.mFlags == bestMove.mFlags )
            //    if( mHistoryTable[whiteToMove][currMove.mDest][currMove.mSrc] < mHistoryTable[whiteToMove][bestMove.mDest][bestMove.mSrc] )
            //        continue;   

            best = idx;
        }

        Exchange( moves.mMove[moves.mTried], moves.mMove[best] );
        return( moves.mTried++ );
    }


    void RunToDepthForCpu( int depth, bool printPv = false )
    {
        int level = mCpuLevel;

        if( !mOptions[OPTION_ENABLE_SIMD] )
            level = CPU_X64;

        switch( level )
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
        bool usePopcnt = mPopcntSupported;

        if( !mOptions[OPTION_ENABLE_POPCNT] )
            usePopcnt = false;

        if( usePopcnt )
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

        Position rootFlipped;
        rootFlipped.FlipFrom( mRoot );

        float gamePhase = mEvaluator.CalcGamePhase< POPCNT >( mRoot );
        mEvaluator.GenerateWeights( mRootWeights, gamePhase );

        //mMaterialTable[mRoot.mWhiteToMove    ].CalcTableOld( gamePhase );
        //mMaterialTable[mRoot.mWhiteToMove ^ 1].CalcTableOld( gamePhase );

        //mMaterialTable[mRoot.mWhiteToMove    ].CalcTable( mRoot.CalcMaterialCategory() );
        //mMaterialTable[mRoot.mWhiteToMove ^ 1].CalcTable( rootFlipped.CalcMaterialCategory() );

        //mRoot.CalcMaterial( &mMaterialTable[mRoot.mWhiteToMove], &mMaterialTable[mRoot.mWhiteToMove ^ 1] );

        rootScore = (EvalTerm) mEvaluator.Evaluate< POPCNT >( mRoot, moveMap, mRootWeights );

		searchTime.Reset();

        SearchState< POPCNT, SIMD > ss;
        ss.mHashTable = &mHashTable;
        ss.mEvaluator = &mEvaluator;
        ss.mExitSearch = &mExitSearch;
        ss.mMetrics = &mMetrics;

        EvalTerm score = ss.RunToDepth( mRoot, depth );
        ss.ExtractBestLine( &pv  );

        if( mExitSearch )
            return;

        if( printPv )
        {
            i64 elapsed     = Max( searchTime.GetElapsedMs(), (i64) 1 );
            i64 nps         = mMetrics.mNodesTotal * 1000L / elapsed;
            int hashfull    = (int) (mHashTable.EstimateUtilization() * 1000);
            int seldepth    = ss.mDeepestPly;

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
                printf( "info string DEBUG: simdnodes %" PRId64 "  >>>> %d: %d\n", mMetrics.mNodesTotalSimd, depth, score );

            *mStorePv   = pv;
            mValuePv    = score;

            if( depth < METRICS_DEPTH )
                mPvDepth[depth] = pv;
        }
    }
};




#endif // PIGEON_ENGINE_H__
};
