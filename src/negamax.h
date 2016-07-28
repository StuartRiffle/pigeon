// negamax.h - PIGEON CHESS ENGINE (c) 2012-2016 Stuart Riffle

// WORK IN PROGRESS

#if 0//ndef PIGEON_NEGAMAX_H__
#define PIGEON_NEGAMAX_H__
namespace Pigeon {


struct SearchState
{
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
        EvalTerm        score; 
        bool            onPv;
        bool            inCheck;
        int             movesTried;
        EvalTerm        bestScore;
        MoveSpec        bestMove;
        MoveList        moves;
    };

    Frame               mFrames[MAX_SEARCH_DEPTH];
    Frame*              mCurrFrame;

    void Advance()
    {
        SearchFrame* RESTRICT f = mCurrFrame;

        if( f->step == UNPACK_MOVES )
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
        }

        if( f->step == UNPACK_MOVES )
        {
            f->moves.UnpackMoveMap( f->pos, f->moveMap );
            f->inCheck = (f->moveMap.IsInCheck() != 0);

            if( f->moves.mCount == 0 )
            {
                f->result = f->inCheck? EVAL_CHECKMATE : EVAL_STALEMATE;
                f--;
            }
        }

        if( f->step == UNPACK_MOVES )
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
        }

        if( f->step == UNPACK_MOVES )
        {
            if( f->depth >= 0 )
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
                    int         depthBefore         = tt.mf->depth;

                    if( failedHighBefore && (lowerBoundBefore >= f->beta) && (depthBefore >= f->depth) )
                    {
                        f->result = f->beta;
                        f--;
                    }
                    else
                    {
                        f->moves.MarkSpecialMoves( tt.mBestSrc, tt.mBestDest, TT_BEST_MOVE );
                    }
                }
            }
        }

        if( f->step == UNPACK_MOVES )
        {
            if( f->onPv && (mStorePv->mCount > f->ply) )
            {
                MoveSpec& pvMove = mStorePv->mMove[f->ply];
                f->moves.MarkSpecialMoves( pvMove.mSrc, pvMove.mDest, PRINCIPAL_VARIATION );
            }

            f->movesTried   = 0;
            f->bestScore    = f->alpha;
            f->step         = ITERATE_CHILDREN;
        }

        if( f->step == ITERATE_CHILDREN )
        {
            if( (f->movesTried >= f->moves.mCount) || (f->bestScore >= f->beta) )
            {
                f->step = FINALIZE;
            }
            else
            {
                int idxMove = this->ChooseNextMove( f->moves, (int) f->pos.mWhiteToMove );

                f[1].pos = f->pos;
                f[1].pos.Step( f->moves.mMove[idxMove] );
                f[1].pos.CalcMoveMap( &f[1].moveMap );

                f[1].score      = mEvaluator.Evaluate( f[1].pos, f[1].moveMap );
                f[1].ply        = f->ply + 1; 
                f[1].depth      = f->depth - 1; 
                f[1].alpha      = -f->beta; 
                f[1].beta       = -f->bestScore;
                f[1].step       = UNPACK_MOVES;
                f->step         = CHECK_SCORE;
                f++;
            }
        }

        if( f->step == CHECK_SCORE )
        {
            EvalTerm subScore = f[1].result;

            if( subScore > f->bestScore )
            {
                f->bestScore   = subScore;
                f->bestMove    = f->childSpec[f->simdIdx];

                MoveList pv;

                for( int i = 0; i <= f->ply; i++ )
                    pv.Append( mFrames[i].bestMove );

                RegisterBestLine( pv, f->alpha, f->beta );
            }

            f->movesTried++;
            f->step = ITERATE_CHILDREN;
        }

        if( f->step == FINALIZE )
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
        }

        mCurrFrame = f;
    }
};


};
#endif // PIGEON_NEGAMAX_H__
