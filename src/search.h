// search.h - PIGEON CHESS ENGINE (c) 2012-2016 Stuart Riffle
        
namespace Pigeon {
#ifndef PIGEON_SEARCH_H__
#define PIGEON_SEARCH_H__


/// Parameters for a best-move search
//
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
//
struct SearchMetrics
{
    u64                 mNodesTotal;
    u64                 mNodesTotalSimd;
    u64                 mNodesAtPly[METRICS_DEPTH];
    u64                 mHashLookupsAtPly[METRICS_DEPTH];
    u64                 mHashHitsAtPly[METRICS_DEPTH];
    u64                 mMovesTriedByPly[METRICS_DEPTH][METRICS_MOVES];

    SearchMetrics()     { this->Clear(); }
    void Clear()        { PlatClearMemory( this, sizeof( *this ) ); }
};


#endif // PIGEON_SEARCH_H__
};
