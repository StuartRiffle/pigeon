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
    bool                mUsePopcnt;                 // Enable use of the hardware POPCNT instruction
    EvalTerm            mRootWeights[EVAL_TERMS];   // Terms calculated at the root position (used when !mRollingPhase)
    int                 mRecalcWeightFreq;          // Frequency at which to recalculate weights based on game phase (power of two) 

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
        mRecalcWeightFreq   = 1;

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

    const char* SetPosition( const char* str )
    {
        this->Reset();

        str = FEN::StringToPosition( str, mRoot );
        return( str );
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

        float gamePhase = mEvaluator.CalcGamePhase< DISABLE_POPCNT >( mRoot );
        mEvaluator.GenerateWeights( mRootWeights, gamePhase );

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

            if( mUsePopcnt )
                this->RunToDepth< ENABLE_POPCNT >( depth );
            else
                this->RunToDepth< DISABLE_POPCNT >( depth );

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

    template< int POPCNT >
    EvalTerm NegaMax( Position& pos, int ply, int depth, EvalTerm alpha, EvalTerm beta, MoveList* pv_new, bool onPvPrev )
    {
        mMetrics.mNodesTotal++;
        mMetrics.mNodesAtPly[ply]++;

        if( mExitSearch )
            return( EVAL_SEARCH_ABORTED );

        if( POPCNT )
            mHashTable.Prefetch( pos.mHash );

        EvalTerm  currWeights[EVAL_TERMS];
        EvalTerm* useWeights = mRootWeights;

        if( (ply & (mRecalcWeightFreq - 1)) == 0 )
        {
            float gamePhase = mEvaluator.CalcGamePhase< POPCNT >( pos );
            mEvaluator.GenerateWeights( currWeights, gamePhase );
            useWeights = currWeights;
        }

        EvalTerm score = (EvalTerm) mEvaluator.Evaluate< POPCNT >( pos, useWeights );

        MoveList moves;
        moves.FindMoves( pos );

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

                if( samePlayer )
                    if( failedHighBefore && (lowerBoundBefore >= beta) && (depthBefore >= depth) )
                        return( beta );

                if( samePlayer & !(tt.mFailHigh || tt.mFailLow) )
                    moves.MarkSpecialMoves( tt.mBestSrc, tt.mBestDest, TT_BEST_MOVE );
            }
        }

        if( onPvPrev && (mStorePv->mCount > ply) )
        {
            MoveSpec& pvMove = mStorePv->mMove[ply];
            moves.MarkSpecialMoves( pvMove.mSrc, pvMove.mDest, PRINCIPAL_VARIATION );
        }

        EvalTerm    best    = alpha;
        int         bestIdx = -1;

        while( (moves.mTried < moves.mCount) && (best < beta) )
        {
            int idx = moves.ChooseBestUntried();
            MoveSpec& move = moves.mMove[idx];

            Position child = pos;
            child.Step( move );

            MoveList pv_child;
            EvalTerm score = -this->NegaMax< POPCNT >( child, ply + 1, depth - 1, -beta, -best, &pv_child, (move.mType == PRINCIPAL_VARIATION) );

            if( score > best )
            {
                pv_new->mCount = 1;
                pv_new->mMove[0] = move;
                pv_new->Append( pv_child );

                best    = score;
                bestIdx = idx;

                mMetrics.mPvByOrder[moves.mTried - 1][move.mType]++;
            }
        }

        if( mExitSearch )
            return( EVAL_SEARCH_ABORTED );

        bool        failedHigh  = (best > beta);
        bool        failedLow   = (bestIdx < 0);
        EvalTerm    result      = failedHigh? beta : (failedLow? alpha : best);

        mMetrics.mCutsByOrder[moves.mTried - 1] += failedHigh? 1 : 0;

        if( depth > 0 )
        {
            TableEntry tt;

            tt.mHashVerify  = pos.mHash >> 40;
            tt.mDepth       = depth;
            tt.mScore       = result;
            tt.mBestSrc     = (failedLow || failedHigh)? 0 : moves.mMove[bestIdx].mSrc;
            tt.mBestDest    = (failedLow || failedHigh)? 0 : moves.mMove[bestIdx].mDest;
            tt.mFailLow     = failedLow;
            tt.mFailHigh    = failedHigh;
            tt.mWhiteMove   = pos.mWhiteToMove? true : false;

            mHashTable.Store( pos.mHash, tt );
        }

        return( result );
    }


    template< int POPCNT >
    void RunToDepth( int depth )
    {
        mMetrics.Clear();

        MoveList pv;
        EvalTerm score = this->NegaMax< POPCNT >( mRoot, 0, depth, -EVAL_MAX, EVAL_MAX, &pv, true );
        if( mExitSearch )
            return;

        i64 elapsed     = Max( mSearchElapsed.GetElapsedMs(), (i64) 1 );
        i64 nps         = mMetrics.mNodesTotal * 1000L / elapsed;
        int hashfull    = (int) (mHashTable.CalcUtilization() * 1000);
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
        printf( "time %d ",         elapsed );
        printf( "nps %"PRId64" ",   nps );
        printf( "pv " );            FEN::PrintMoveList( pv );
        printf( "\n" );

        fflush( stdout );

        *mStorePv   = pv;
        mValuePv    = score;
    }
};




#endif // PIGEON_ENGINE_H__
};
