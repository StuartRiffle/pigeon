// pigeon.cpp - PIGEON CHESS ENGINE (c) 2012-2016 Stuart Riffle

#include "platform.h"
#include "defs.h"
#include "bits.h"
#include "cpu-sse2.h"
#include "cpu-sse4.h"
#include "cpu-avx2.h"
#include "cpu-avx512.h"
#include "position.h"
#include "eval.h"
#include "movelist.h"
#include "table.h"
#include "tb-gaviota.h"
#include "search.h"

#include <algorithm>
#include <random>
#include <string>
#include <vector>
#include <stack>
#include <queue>
#include <list>
#include <map>
#include <functional>
#include <memory>

#include <stdio.h>
#include <ctype.h>
#include <time.h>

#include "fen.h"
#include "token.h"
#include "book.h"
#include "perft.h"
#include "thread.h"
#include "engine.h"
#include "amoeba.h"
#include "tune.h"
#include "uci.h"

//#include "tb-gaviota.h" 

        
int main( int argc, char** argv )
{
    // Disable I/O buffering

    setvbuf( stdin,  NULL, _IONBF, 0 );
    setvbuf( stdout, NULL, _IONBF, 0 );

    // Detect hardware support
    
    int cudaCount = 0;
    
#if PIGEON_ENABLE_CUDA
    cudaCount = Pigeon::CudaSystem::GetDeviceCount();
#endif

    const char* cpuDesc[] = { "x64", "SSE2", "SSE4", "AVX2", "AVX512" };

    std::string hardwareDesc = cpuDesc[Pigeon::PlatDetectCpuLevel()];
    hardwareDesc += Pigeon::PlatDetectPopcnt()? "/POPCNT" : "";
    hardwareDesc += cudaCount? "/CUDA" : "";
    
    if( cudaCount > 1 )
    {
        char cudaCountStr[16];
        sprintf( cudaCountStr, "x%d", cudaCount );
        hardwareDesc += cudaCountStr;
    }
     
    char versionStr[32];
    sprintf( versionStr, "%d.%d.%d%s", Pigeon::PIGEON_VER_MAJOR, Pigeon::PIGEON_VER_MINOR, Pigeon::PIGEON_VER_PATCH, Pigeon::PIGEON_VER_DEV? "-DEV" : "" ); 

    // Draw the pretty birdie

    printf( "\n" );                      
    printf( "     /O_"  "    Pigeon %s (UCI)\n", versionStr );
    printf( "     || "  "    %s\n", hardwareDesc.c_str() );
    printf( "    / \\\\""    \n" );
    printf( "  =/__//"  "    pigeonengine.com\n" );
    printf( "     ^^ "  "    \n" );
    printf( "\n" );          

    Pigeon::Engine pigeon;

    // UCI commands can be passed as arguments, separated by semicolons (handy for debugging):
    //      uci; setoption name OwnBook value false; isready;

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

    // Process standard input until there's no more

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

