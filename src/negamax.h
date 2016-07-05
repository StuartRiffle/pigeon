// negamax.h - PIGEON CHESS ENGINE (c) 2012-2016 Stuart Riffle
        
template< int POPCNT, typename SIMD >
PDECL EvalTerm NegaMaxIter( 
    const Position& pos, 
    const MoveMap&  moveMap, 
    EvalTerm        score, 
    int             ply, 
    int             depth, 
    EvalTerm        alpha, 
    EvalTerm        beta, 
    MoveList*       pv_new, 
    bool            onPvPrev )
{
#if !defined( __CUDA_ARCH__ )
//#if !defined( PIGEON_CUDA )              2
    const int LANES = SimdWidth< SIMD >::LANES;
#endif

#if PIGEON_METRICS
    mMetrics.mNodesTotal++;
    mMetrics.mNodesAtPly[ply]++;
#endif

    if( depth < 1 )
    {
        alpha = Max( alpha, score );

        if( alpha >= beta )
            return( beta );
    }

#if !defined( PIGEON_CUDA )
    if( mExitSearch )
        return( EVAL_SEARCH_ABORTED );

    if( POPCNT )
        mHashTable.Prefetch( pos.mHash );
#endif

    MoveList moves;
    moves.UnpackMoveMap( pos, moveMap );

    bool inCheck = (moveMap.IsInCheck() != 0);

    if( moves.mCount == 0 )
        return( inCheck? EVAL_CHECKMATE : EVAL_STALEMATE );

    if( depth < 1 )
    {
        if( !inCheck )
            moves.DiscardMovesBelow( CAPTURE_LOSING );

        if( moves.mCount == 0 )
            return( score );
    }

#if PIGEON_USE_HASH
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
#endif

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
                subScore = -NegaMax< POPCNT, SIMD >( 
                    childPos[simdIdx], childMoveMap[simdIdx], childScore[simdIdx], ply + 1, depth - 1, -(bestScore + 1), -bestScore, 
                    &pv_child, (childSpec[simdIdx].mType == PRINCIPAL_VARIATION) );
            
                fullSearch = (subScore > bestScore) && (subScore < beta);
            }

            if( fullSearch )
            {
                subScore = -NegaMax< POPCNT, SIMD >( 
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

                #if PIGEON_HISTORY
                if( depth > 2 )
                    mHistoryTable[pos.mWhiteToMove][bestMove.mDest][bestMove.mSrc] += (depth * depth);
                #endif

                if( depth < 1 )
                    break;
            }
        }

    #if PIGEON_METRICS
        if( (ply < METRICS_DEPTH) && (movesTried < METRICS_MOVES) )
            mMetrics.mMovesTriedByPly[ply][movesTried]++;
    #endif

        movesTried++;

        //if( depth < -1 )
        //    break;

    }

#if !defined( __CUDACC__ )
    if( mExitSearch )
        return( EVAL_SEARCH_ABORTED );
#endif

    bool        failedHigh  = (bestScore >= beta);
    bool        failedLow   = (bestScore == alpha);
    EvalTerm    result      = failedHigh? beta : (failedLow? alpha : bestScore);

#if PIGEON_HISTORY
    //if( failedHigh )
    //    mHistoryTable[pos.mWhiteToMove][bestMove.mDest][bestMove.mSrc] += (ply * ply);
#endif

#if PIGEON_USE_HASH
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
#endif

    return( result );
}

