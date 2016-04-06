// pigeon.cpp - PIGEON CHESS ENGINE (c) 2012-2016 Stuart Riffle

#include "platform.h"
#include "defs.h"
#include "bits.h"
#include "simd.h"
#include "position.h"
#include "eval.h"
#include "movelist.h"
#include "table.h"
#include "search.h"

#include <stdio.h>
#include <ctype.h>
#include <time.h>
#include <vector>

#include "timer.h"
#include "token.h"
#include "fen.h"
#include "perft.h"
#include "engine.h"
#include "uci.h"

        
int main( int argc, char** argv )
{
    Pigeon::Engine pigeon;

    printf( "\n" );                      
    printf( "     /O_"  "   PIGEON CHESS ENGINE\n" );
    printf( "     || "  "   v%d.%02d UCI\n", Pigeon::PIGEON_VER_MAJ, Pigeon::PIGEON_VER_MIN  );
    printf( "    / \\\\""   \n" );
    printf( "  =/__//"  "   (x64%s)\n", Pigeon::PlatDetectPopcnt()? "/POPCNT" : "" );
    printf( "     ^^ "  "   \n" );
    printf( "\n" );

    Pigeon::UCI::ProcessCommand( &pigeon, "uci\n" );
    Pigeon::UCI::ProcessCommand( &pigeon, "isready\n" );
    Pigeon::UCI::ProcessCommand( &pigeon, "go depth 9\n" );

    while( !feof( stdin ) )
    {
        char buf[2048];

        const char* cmd = fgets( buf, sizeof( buf ), stdin );
        if( cmd == NULL )
            continue;

        bool done = Pigeon::UCI::ProcessCommand( &pigeon, cmd );
        if( done )
            break;
    }

    return( 0 );
}

