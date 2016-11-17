// engine.h - PIGEON CHESS ENGINE (c) 2012-2016 Stuart Riffle



namespace Pigeon {
#ifndef PIGEON_ENGINE_H__
#define PIGEON_ENGINE_H__






//==============================================================================
///

PDECL class Engine
{
    HashTable               mHashTable;                 /// Transposition table
    Position                mRoot;                      /// The root of the search tree (the "current position")
    SearchConfig            mConfig;                    /// Search parameters
    SearchMetrics           mMetrics;                   /// Runtime metrics
    Evaluator               mEvaluator;                 /// Evaluation weight calculator
    OpeningBook             mOpeningBook;               /// Placeholder opening book implementation
    MoveList                mBestLine;                  /// Best line found in the search
    MoveList                mPvDepth[METRICS_DEPTH];    /// Best line found at 
    MoveList*               mStorePv;                   /// Target for PV in active search
    EvalTerm                mValuePv;                   /// The evaluation of *mStorePv
    int                     mTargetTime;                /// Time to stop current search
    int                     mDepthLimit;                /// Depth limit for current search (not counting quiesence)
    Timer                   mSearchElapsed;             /// Time elapsed since the "go" command
    volatile bool           mExitSearch;                /// Flag to terminate search threads immediately
    int                     mThreadsRunning;            /// Number of worker threads currently running
    Semaphore               mThreadsDone;               /// Semaphore to help gather up completed threads                                                
    bool                    mPrintBestMove;             /// Output best move while searching
    bool                    mPrintedMove;               /// Make sure only one "bestmove" is output per "go" 
    bool                    mDebugMode;                 /// This currently does nothing
    int                     mCpuLevel;                  /// A CPU_* enum value that reflects the hardware capabilities
    bool                    mPopcntSupported;           /// True if the CPU can do POPCNT
    std::map< u64, int >    mPositionReps;              /// Indexed by hash, detects repetitions to avoid (unwanted) draw
    i32                     mOptions[OPTION_COUNT];     /// Runtime options exposed via UCI
    HistoryTable            mHistoryTable;              /// History table for move ordering
    GaviotaTablebase        mTablebase;                 /// Endgame tablebase interface
    std::string             mGaviotaPath;               /// Gaviota tablebase path
//  MaterialTable           mMaterialTable[2];          /// Material value for each piece/square combination, indexed by pos.mWhiteToMove

#if PIGEON_CUDA_HOST
    CudaChessManager        mCudaSearcher;
#endif

public:

    //--------------------------------------------------------------------------
    ///

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
        PlatClearMemory( mOptions, sizeof( mOptions ) );

        // #OPTIONS

        mOptions[OPTION_HASH_SIZE]          = TT_MEGS_DEFAULT;
        mOptions[OPTION_CLEAR_HASH]         = 0;
        mOptions[OPTION_OWN_BOOK]           = OWNBOOK_DEFAULT? 1 : 0;
        mOptions[OPTION_NUM_THREADS]        = PlatDetectCpuCores();
        mOptions[OPTION_ENABLE_SIMD]        = 1;
        mOptions[OPTION_ENABLE_POPCNT]      = 1;
        mOptions[OPTION_ENABLE_CUDA]        = 0;
        mOptions[OPTION_EARLY_MOVE]         = 1;
        mOptions[OPTION_USE_PVS]            = 1;
        mOptions[OPTION_ALLOW_LMR]          = 1;
        mOptions[OPTION_ASPIRATION_WINDOW]  = 1;
        mOptions[OPTION_GAVIOTA_CACHE_SIZE] = GAVIOTA_CACHE_DEFAULT;
        mOptions[OPTION_GPU_HASH_SIZE]      = TT_MEGS_DEFAULT;
        mOptions[OPTION_GPU_BATCH_SIZE]     = BATCH_SIZE_DEFAULT;
        mOptions[OPTION_GPU_BATCH_COUNT]    = BATCH_COUNT_DEFAULT;
        mOptions[OPTION_GPU_BLOCK_WARPS]    = GPU_BLOCK_WARPS;
        mOptions[OPTION_GPU_PLIES]          = GPU_PLIES_DEFAULT;
        mOptions[OPTION_GPU_SPIN_TO_ALLOC]  = 0;

        mHashTable.SetSize( mOptions[OPTION_HASH_SIZE] );
    }


    //--------------------------------------------------------------------------
    ///

    ~Engine()
    {
        this->Stop();
    }


    //--------------------------------------------------------------------------
    ///

    void Reset()
    {
        this->Stop();
        mRoot.Reset();
        mEvaluator.EnableOpening( true );
        mHashTable.Clear();
        mPositionReps.clear();
    }

    void SetPosition( const Position& pos )
    {
        this->Reset();
        mRoot = pos;
        mEvaluator.EnableOpening( false );

        mRoot.mHash = mRoot.CalcHash();
        mPositionReps[mRoot.mHash]++;
    }


    //--------------------------------------------------------------------------
    ///

    Position GetPosition() const
    {
        return( mRoot );    
    }


    //--------------------------------------------------------------------------
    ///

    void OverrideCpuLevel( int level )
    {
        mCpuLevel = level;
    }


    //--------------------------------------------------------------------------
    ///

    void SetOption( int idx, int value )
    {
        if( (idx >= 0) && (idx < OPTION_COUNT) )
            mOptions[idx] = value;
    }


    void SetGaviotaPath( const char* path )
    {
        mGaviotaPath = path;
    }


    //--------------------------------------------------------------------------
    ///

    const i32* GetOptions() const
    {
        return( &mOptions[0] );
    }


    //--------------------------------------------------------------------------
    ///

    void LazyInitCuda()
    {
#if PIGEON_CUDA_HOST
        if( mOptions[OPTION_ENABLE_CUDA] && (CudaSystem::GetDeviceCount() > 0) )
            if( !mCudaSearcher.IsInitialized() )
                mCudaSearcher.Initialize( mOptions );
#endif
    }

    void LazyInitGaviota()
    {
        if( (mOptions[OPTION_GAVIOTA_CACHE_SIZE] > 0) && !mGaviotaPath.empty() )
        {
            if( !mTablebase.IsInitialized() )
                mTablebase.Init( mGaviotaPath.c_str(), mOptions[OPTION_GAVIOTA_CACHE_SIZE] );
        }
        else
        {
            if( mTablebase.IsInitialized() )
                mTablebase.Shutdown();
        }
    }


    //--------------------------------------------------------------------------
    ///

    void Init()
    {
        if( mOptions[OPTION_HASH_SIZE] != mHashTable.GetSize() )
            mHashTable.SetSize( mOptions[OPTION_HASH_SIZE] );

        this->LazyInitCuda();
        this->LazyInitGaviota();
    }


    //--------------------------------------------------------------------------
    ///

    void SetDebug( bool debug )
    {
        mDebugMode = debug;
    }


    //--------------------------------------------------------------------------
    ///

    void LoadWeightParam( const char* name, float openingVal, float midgameVal, float endgameVal ) 
    {
        int idx = mEvaluator.GetWeightIdx( name );

        if( idx >= 0 )
            mEvaluator.SetWeight( idx, openingVal, midgameVal, endgameVal );
    }


    //--------------------------------------------------------------------------
    ///

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


    //--------------------------------------------------------------------------
    ///

    void PonderHit()
    {
        printf( "info string WARNING: ponderhit not implemented\n" );
    }


    //--------------------------------------------------------------------------
    ///

    void Go( SearchConfig* conf )
    {
        this->Stop();
        mConfig = *conf;

        this->LazyInitCuda();
        this->LazyInitGaviota();

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
        mHistoryTable.Clear();

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


    //--------------------------------------------------------------------------
    ///

    void Stop( bool printResult = false )
    {
        this->JoinAllThreads();

        if( printResult )
            this->PrintResult();
    }


    //--------------------------------------------------------------------------
    ///

    void PrintPosition()
    {
		printf( "info string position " );
        FEN::PrintPosition( mRoot );
        printf( "\n" );
    }


    //--------------------------------------------------------------------------
    ///

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


    //--------------------------------------------------------------------------
    ///

    static void* SearchThreadProc( void* param )
    {
        PlatSetThreadName( "Search Thread" );

        Engine* engine = reinterpret_cast< Engine* >( param );
        engine->SearchThread();
        engine->mThreadsDone.Post();
        return( NULL );
    }


    //--------------------------------------------------------------------------
    ///

	static void* TimerThreadProc( void* param )
    {
        PlatSetThreadName( "Timer Thread" );

        Engine* engine = reinterpret_cast< Engine* >( param );
        engine->TimerThread();
        engine->mThreadsDone.Post();
        return( NULL );
    }


    //--------------------------------------------------------------------------
    ///

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


    //--------------------------------------------------------------------------
    ///

    void TimerThread()
    {
        for( ;; )
        {
            i64 elapsed   = mSearchElapsed.GetElapsedMs();
            i64 remaining = mTargetTime - elapsed;

            if( remaining < MIN_TIME_SLICE )
                break;

            if( mExitSearch )
                break;

            i64 sleepTime = (remaining < MAX_TIME_SLICE)? remaining : MAX_TIME_SLICE;

            PlatSleep( (int) sleepTime );
        }        

        mExitSearch = true;
    }


    //--------------------------------------------------------------------------
    ///

    void JoinAllThreads()
    {
        i64 cpuStallMs = 0;
        i64 gpuStallMs = 0;

        Timer cpuStallTimer;
        mExitSearch = true;

        while( mThreadsRunning > 0 )
        {
            mThreadsDone.Wait();
            mThreadsRunning--;
        }

        cpuStallMs = cpuStallTimer.GetElapsedMs();
        mExitSearch = false;

#if PIGEON_CUDA_HOST
        if( mCudaSearcher.IsInitialized() )
        {
            Timer gpuStallTimer;
            mCudaSearcher.CancelAllBatchesSync();

            gpuStallMs = gpuStallTimer.GetElapsedMs();
        }
#endif

        if( mDebugMode && (cpuStallMs || gpuStallMs) )
        {
            printf( "info string DEBUG: CPU blocked %dms", (int) cpuStallMs );

#if PIGEON_CUDA_HOST
            if( gpuStallMs )
                printf( ", GPU blocked %dms", (int) gpuStallMs );
#endif

            printf( " in JoinAllThreads()\n" );
        }
    }


    //--------------------------------------------------------------------------
    ///

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


    //--------------------------------------------------------------------------
    ///

    int CalcTargetTime()
    {
        int moveNumber      = (int) mRoot.mFullmoveNum;
        int movesToGo       = mConfig.mTimeControlMoves;
        int moveTimeLimit   = mConfig.mTimeLimit;   
        int totalTimeLeft   = mRoot.mWhiteToMove? mConfig.mWhiteTimeLeft : mConfig.mBlackTimeLeft;
        int timeBonus       = mRoot.mWhiteToMove? mConfig.mWhiteTimeInc  : mConfig.mBlackTimeInc;

        totalTimeLeft -= LAG_SAFETY_BUFFER;
        if( totalTimeLeft < 0 )
            totalTimeLeft = 0;

        if( !moveTimeLimit && !totalTimeLeft && !timeBonus )
            return( NO_TIME_LIMIT );

        int targetTime = (int) (totalTimeLeft * 0.05f); // TODO: make tunable

        if( movesToGo )
            targetTime = totalTimeLeft / movesToGo;

        targetTime += timeBonus;

        if( moveTimeLimit )
            targetTime = Min( targetTime, moveTimeLimit );

        return( targetTime );
    }



    //--------------------------------------------------------------------------
    ///

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


    //--------------------------------------------------------------------------
    ///

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


    //--------------------------------------------------------------------------
    ///

    template< int POPCNT, typename SIMD >
    void RunToDepth( int depth, bool printPv )
    {
        MoveList	pv;
        Timer		searchTime;
        MoveMap     moveMap;
        MoveList    moves;
        EvalTerm    rootScore;
        EvalWeight  rootWeights[EVAL_TERMS];

        mRoot.CalcMoveMap( &moveMap );
        moves.UnpackMoveMap( mRoot, moveMap );

        Position rootFlipped;
        rootFlipped.FlipFrom( mRoot );



        float gamePhase = mEvaluator.CalcGamePhase< POPCNT >( mRoot );
        mEvaluator.GenerateWeights( rootWeights, gamePhase );

        rootScore = (EvalTerm) mEvaluator.Evaluate< POPCNT >( mRoot, moveMap, rootWeights );

		searchTime.Reset();

        SearchState< POPCNT, SIMD > ss;

        ss.mHashTable       = &mHashTable;
        ss.mEvaluator       = &mEvaluator;
        ss.mExitSearch      = &mExitSearch;
        ss.mMetrics         = &mMetrics;
        ss.mHistoryTable    = &mHistoryTable;
        ss.mOptions         = mOptions;

        mHistoryTable.Decay();
        mMetrics.Clear();

#if PIGEON_ENABLE_CUDA
        if( mOptions[OPTION_ENABLE_CUDA] && mCudaSearcher.IsInitialized() )
        {
            if( depth >= (MIN_CPU_PLIES + mOptions[OPTION_GPU_PLIES]) )
            {
                ss.mCudaSearcher   = &mCudaSearcher;
                ss.mAsyncSpawnPly  = depth - mOptions[OPTION_GPU_PLIES];
                ss.mBatchLimit     = mOptions[OPTION_GPU_BATCH_COUNT];
            }
        }
#endif
        if( mTablebase.IsInitialized() )
            ss.mTablebase = &mTablebase;


        int         alphaWindow = 50;
        int         betaWindow  = 50;
        EvalTerm    score       = 0;
        bool        aspiration  = (mOptions[OPTION_ASPIRATION_WINDOW] != 0);

        if( depth < 4 )
            aspiration = false;

        for( ;; )
        {
            int alpha  = -EVAL_MAX;
            int beta   = EVAL_MAX;

            if( aspiration )
            {
                alpha = mValuePv - alphaWindow;
                beta  = mValuePv + betaWindow;

                if( (alpha < -EVAL_MAX) || (beta > EVAL_MAX) )
                    aspiration = false;
            }

            if( !aspiration )
            {
                alpha  = -EVAL_MAX;
                beta   = EVAL_MAX;
            }

            if( mDebugMode )
            {
                if( aspiration )
                    printf( "info string DEBUG: aspiration search [%d, %d]\n", alpha, beta );
                else
                    printf( "info string DEBUG: full window search\n" );
            }

            if( mStorePv && (depth > 1) )
                ss.InsertBestLine( mStorePv );

            score = ss.RunToDepth( &mRoot, &moveMap, depth, 0, rootScore, (EvalTerm) alpha, (EvalTerm) beta );
            if( mExitSearch )
                return;

            if( !aspiration )
                break;

            if( score <= alpha )
                alphaWindow *= 4;
            else if( score >= beta )
                betaWindow *= 4;
            else
                break;
        }

        ss.ExtractBestLine( &pv  );

        if( printPv )
        {
            i64 nodes       = mMetrics.mNodesTotal + mMetrics.mGpuNodesTotal;
            i64 elapsed     = Max( searchTime.GetElapsedMs(), (i64) 1 );
            i64 nps         = nodes * 1000L / elapsed;
            int hashfull    = (int) (mHashTable.EstimateUtilization() * 1000);
            int seldepth    = ss.mDeepestPly;

            //if( score == EVAL_SEARCH_ABORTED )
            //    score = rootScore;

            printf( "info " );
            printf( "depth %d ",            depth );
            printf( "seldepth %d ",         seldepth );
            printf( "score cp %d ",         score );
            printf( "hashfull %d ",         hashfull );
            printf( "nodes %" PRId64 " ",   nodes );
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
