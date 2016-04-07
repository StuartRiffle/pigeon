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
#include <string>

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

    // UCI commands can be passed as arguments, separated by semicolons (handy for debugging)

    std::string commands;
    for( int i = 1; i < argc; i++ )
        commands += std::string( argv[i] ) + " ";

    commands += ";";

    for( ;; )
    {
        size_t delimPos = commands.find( ';' );
        if( delimPos == std::string::npos )
            break;

        std::string cmd = commands.substr( 0, delimPos ) + "\n";

        const char* cmdStart = cmd.c_str();
        while( *cmdStart && isspace( *cmdStart ) )
            cmdStart++;

        if( *cmdStart ) 
        {
            printf( "%s", cmdStart );
            Pigeon::UCI::ProcessCommand( &pigeon, cmdStart );
        }

        commands = commands.substr( delimPos + 1 );
    }

    // Process standard input

    while( !feof( stdin ) )
    {
        char buf[8192];

        const char* cmd = fgets( buf, sizeof( buf ), stdin );
        if( cmd == NULL )
            continue;

        bool done = Pigeon::UCI::ProcessCommand( &pigeon, cmd );
        if( done )
            break;

        fflush( stdout );
    }

    return( 0 );
}

