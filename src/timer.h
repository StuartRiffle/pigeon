// timer.h - PIGEON CHESS ENGINE (c) 2012-2016 Stuart Riffle
        
namespace Pigeon {
#ifndef PIGEON_TIMER_H__
#define PIGEON_TIMER_H__


struct Timer
{
    clock_t     mStartTime;

    Timer() { this->Reset(); }
    Timer( const Timer& rhs ) : mStartTime( rhs.mStartTime ) {}

    void    Reset()         { mStartTime = clock(); }
    i64     GetElapsedMs()  { return( ((i64) (clock() - mStartTime) * 1000) / CLOCKS_PER_SEC ); }
};

#endif // PIGEON_TIMER_H__
};
