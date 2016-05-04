// perft.h - PIGEON CHESS ENGINE (c) 2012-2016 Stuart Riffle
        
namespace Pigeon {
#ifndef PIGEON_PERFT_H__
#define PIGEON_PERFT_H__

struct Perft
{
    static void GatherPerftParallelPositions( const Position& pos, int depth, std::vector< Position >* dest )
    {
        MoveList valid;
        valid.FindMoves( pos );

        for( int i = 0; i < valid.mCount; i++ )
        {
            Position next = pos;
            next.Step( valid.mMove[i] );

            if( depth == (PERFT_PARALLEL_MAX + 1) )
                dest->push_back( next );
            else
                Perft::GatherPerftParallelPositions( next, depth - 1, dest );
        }
    }


    static u64 CalcPerftParallel( const Position& pos, int depth )
    {
        std::vector< Position > positions( 16384 );
        Perft::GatherPerftParallelPositions( pos, depth, &positions );

        u64 total = 0;

        //printf( "info string perft parallel positions %d\n", (int) positions.size() );

        #pragma omp parallel for reduction(+: total) schedule(dynamic)
        for( int i = 0; i < (int) positions.size(); i++ )
        {
            u64 subtotal = Perft::CalcPerftInternal( positions[i], PERFT_PARALLEL_MAX );
            total = total + subtotal;
        }

        return( total );
    }


    static u64 CalcPerftInternal( const Position& pos, int depth )
    {
        if( (depth > PERFT_PARALLEL_MAX) && (depth <= PERFT_PARALLEL_MAX + 3) )
        {
            return( Perft::CalcPerftParallel( pos, depth ) );
        }

        MoveList valid;
        valid.FindMoves( pos );

        u64 total = 0;

        for( int i = 0; i < valid.mCount; i++ )
        {
            Position next = pos;
            next.Step( valid.mMove[i] );

            if( depth == 2 )
            {
                MoveList dummy;
                total += dummy.FindMoves( next );
            }

            else
            {
                total += Perft::CalcPerftInternal( next, depth - 1 );
            }
        }

        return( total );
    }


    static u64 CalcPerft( const Position& pos, int depth )
    {
        if( depth < 2 )
        {
            MoveList dummy;
            return( dummy.FindMoves( pos ) );
        }

        return( Perft::CalcPerftInternal( pos, depth ) );
    }


    static void DividePerft( const Position& pos, int depth )
    {
        MoveList valid;
        valid.FindMoves( pos );

        u64 total = 0;

        for( int i = 0; i < valid.mCount; i++ )
        {
            Position next = pos;
            next.Step( valid.mMove[i] );

            u64 count = (depth > 1)? Perft::CalcPerft( next, depth - 1 ) : 1;
            total += count;

            printf( "info string divide %d ", depth );
            FEN::PrintMoveSpec( valid.mMove[i] );
            printf( "  %" PRId64 "\n", count );
        }

        printf( "info string divide %d total %" PRId64 "\n", depth, total );
    }
};

#endif // PIGEON_PERFT_H__
};
