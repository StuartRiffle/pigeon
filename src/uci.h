// uci.h - PIGEON CHESS ENGINE (c) 2012-2016 Stuart Riffle

namespace Pigeon {
#ifndef PIGEON_UCI_H__
#define PIGEON_UCI_H__

struct UCI
{
    static bool ProcessCommand( Engine* engine, const char* str )
    {
        std::string setoption;

        const char* eqpos = strstr( str, "=" );
        if( eqpos )
        {
            // For convenience setting UCI options, interpret "FOO=BAR" as "setoption name FOO value BAR"

            setoption = "setoption name " + std::string( str, eqpos ) + " value " + std::string( eqpos + 1 );
            str = setoption.c_str();
        }

        Tokenizer tokens( str );

        if( tokens.Consume( "#" ) )
        {
            // Treat line as a comment
        }
        else if( tokens.Consume( "uci" ) )
        {                                                                                        
            printf( "id name Pigeon %d.%d.%d%s\n", PIGEON_VER_MAJOR, PIGEON_VER_MINOR, PIGEON_VER_PATCH, PIGEON_VER_DEV? "-DEV" : "" );
            printf( "id author Stuart Riffle\n" );

            const i32* options = engine->GetOptions();

            printf( "option name Hash"              " type spin min 4 max 8192 default %d\n",   options[OPTION_HASH_SIZE] );
            printf( "option name ClearHash"         " type button\n" );
            printf( "option name Threads"           " type spin default 1 min 1 max %d\n",      options[OPTION_NUM_THREADS] );
            printf( "option name OwnBook"           " type check default %s\n",                 options[OPTION_OWN_BOOK]?               "true" : "false" );
            printf( "option name EarlyMove"         " type check default %s\n",                 options[OPTION_EARLY_MOVE]?             "true" : "false" );
            printf( "option name PVS"               " type check default %s\n",                 options[OPTION_USE_PVS]?                "true" : "false" );
            printf( "option name SIMD"              " type check default %s\n",                 options[OPTION_ENABLE_SIMD]?            "true" : "false" );
            printf( "option name POPCNT"            " type check default %s\n",                 options[OPTION_ENABLE_POPCNT]?          "true" : "false" );

#if PIGEON_ENABLE_CUDA
            printf( "option name CUDA"              " type check default %s\n",                 options[OPTION_ENABLE_CUDA]?            "true" : "false" );
            printf( "option name GpuHash"           " type spin min 4 max 8192 default %d\n",   options[OPTION_GPU_HASH_SIZE] );
            printf( "option name GpuBatchSize"      " type spin min 32 max 8192 default %d\n",  options[OPTION_GPU_BATCH_SIZE] );
            printf( "option name GpuBatchCount"     " type spin min 4 max 1024 default %d\n",   options[OPTION_GPU_BATCH_COUNT] );
            printf( "option name GpuBlockWarps"     " type spin min 1 max 32 default %d\n",     options[OPTION_GPU_BLOCK_WARPS] );
            printf( "option name GpuPlies"          " type spin min 0 max 8 default %d\n",      options[OPTION_GPU_PLIES] );
#endif

            printf( "option name AspirationSearch"  " type check default %s\n",                 options[OPTION_USE_ASPIRATION]?         "true" : "false" );
            printf( "option name AspirationWindow"  " type spin default %d min 0 max 1000\n",   options[OPTION_ASPIRATION_WINDOW_SIZE] );
            printf( "option name AspirationScale"   " type spin default %d min 2 max 64\n",     options[OPTION_ASPIRATION_WINDOW_SCALE] );
            printf( "option name GaviotaTbCache"    " type spin min 4 max 512 default %d\n",    options[OPTION_GAVIOTA_CACHE_SIZE] );
            printf( "option name GaviotaTbPath"     " type string\n" );

            printf( "uciok\n" );
        }
        else if( tokens.Consume( "setoption" ) )
        {
            if( tokens.Consume( "name" ) )
            {
                if( tokens.Consume( "hash" ) && tokens.Consume( "value" ) )
                    engine->SetOption( OPTION_HASH_SIZE, tokens.ConsumeInt() );

                if( tokens.Consume( "clearhash" ) )
                    engine->SetOption( OPTION_CLEAR_HASH, 1 );

                if( tokens.Consume( "ownbook" ) && tokens.Consume( "value" ) )
                    engine->SetOption( OPTION_OWN_BOOK, tokens.Consume( "true" )? 1 : 0 );

                if( tokens.Consume( "threads" ) && tokens.Consume( "value" ) )
                    engine->SetOption( OPTION_NUM_THREADS, tokens.Consume( "true" )? 1 : 0 );

                if( tokens.Consume( "earlymove" ) && tokens.Consume( "value" ) )
                    engine->SetOption( OPTION_EARLY_MOVE, tokens.Consume( "true" )? 1 : 0 );

                if( tokens.Consume( "pvs" ) && tokens.Consume( "value" ) )
                    engine->SetOption( OPTION_USE_PVS, tokens.Consume( "true" )? 1 : 0 );

                if( tokens.Consume( "simd" ) && tokens.Consume( "value" ) )
                    engine->SetOption( OPTION_ENABLE_SIMD, tokens.Consume( "true" )? 1 : 0 );

                if( tokens.Consume( "popcnt" ) && tokens.Consume( "value" ) )
                    engine->SetOption( OPTION_ENABLE_POPCNT, tokens.Consume( "true" )? 1 : 0 );

#if PIGEON_ENABLE_CUDA

                if( tokens.Consume( "cuda" ) && tokens.Consume( "value" ) )
                    engine->SetOption( OPTION_ENABLE_CUDA, tokens.Consume( "true" )? 1 : 0 );

                if( tokens.Consume( "gpuhash" ) && tokens.Consume( "value" ) )
                    engine->SetOption( OPTION_GPU_HASH_SIZE, tokens.ConsumeInt() );

                if( tokens.Consume( "gpubatchsize" ) && tokens.Consume( "value" ) )
                    engine->SetOption( OPTION_GPU_BATCH_SIZE, tokens.ConsumeInt() );

                if( tokens.Consume( "gpubatchcount" ) && tokens.Consume( "value" ) )
                    engine->SetOption( OPTION_GPU_BATCH_COUNT, tokens.ConsumeInt() );

                if( tokens.Consume( "gpublockwarps" ) && tokens.Consume( "value" ) )
                    engine->SetOption( OPTION_GPU_BLOCK_WARPS, tokens.ConsumeInt() );

                if( tokens.Consume( "gpuplies" ) && tokens.Consume( "value" ) )
                    engine->SetOption( OPTION_GPU_PLIES, tokens.ConsumeInt() );
#endif

                if( tokens.Consume( "aspirationsearch" ) && tokens.Consume( "value" ) )
                    engine->SetOption( OPTION_USE_ASPIRATION, tokens.Consume( "true" )? 1 : 0 );

                if( tokens.Consume( "aspirationwindow" ) && tokens.Consume( "value" ) )
                    engine->SetOption( OPTION_ASPIRATION_WINDOW_SIZE, tokens.ConsumeInt() );

                if( tokens.Consume( "aspirationscale" ) && tokens.Consume( "value" ) )
                    engine->SetOption( OPTION_ASPIRATION_WINDOW_SCALE, tokens.ConsumeInt() );

                if( tokens.Consume( "gaviotatbcache" ) && tokens.Consume( "value" ) )
                    engine->SetOption( OPTION_GAVIOTA_CACHE_SIZE, tokens.ConsumeInt() );

                if( tokens.Consume( "gaviotatbpath" ) && tokens.Consume( "value" ) )
                    engine->SetGaviotaPath( tokens.ConsumeAll() );

            }
        }
        else if( tokens.Consume( "debug" ) )
        {
            if( tokens.Consume( "on" ) )       
                engine->SetDebug( true );
            else if( tokens.Consume( "off" ) ) 
                engine->SetDebug( false );
        }
        else if( tokens.Consume( "isready" ) )
        {
            engine->Init();
            printf( "readyok\n" );
        }
        else if( tokens.Consume( "ucinewgame" ) )
        {
            engine->Reset();
        }
        else if( tokens.Consume( "position" ) )
        {
            if( tokens.Consume( "startpos" ) )
                engine->Reset();

            if( tokens.Consume( "fen" ) )
            {
                Position pos;

                if( tokens.ConsumePosition( pos ) )
                    engine->SetPosition( pos );
                else
                    printf( "info string ERROR: unable to parse FEN\n" );
            }

            const char* movehistory = "";

            if( tokens.Consume( "moves" ) )
            {
                movehistory = str;

                for( const char* movetext = tokens.ConsumeNext(); movetext; movetext = tokens.ConsumeNext() )
                    engine->Move( movetext );
            }

            //engine->PrintPosition();
            //engine->PrintValidMoves();
        }
        else if( tokens.Consume( "go" ) )
        {
            Pigeon::SearchConfig conf;

            for( ;; )
            {
                if(      tokens.Consume( "wtime" ) )          conf.mWhiteTimeLeft       = tokens.ConsumeInt();
                else if( tokens.Consume( "btime" ) )          conf.mBlackTimeLeft       = tokens.ConsumeInt();
                else if( tokens.Consume( "winc" ) )           conf.mWhiteTimeInc        = tokens.ConsumeInt();
                else if( tokens.Consume( "binc" ) )           conf.mBlackTimeInc        = tokens.ConsumeInt();
                else if( tokens.Consume( "movestogo" ) )      conf.mTimeControlMoves    = tokens.ConsumeInt();
                else if( tokens.Consume( "mate" ) )           conf.mMateSearchDepth     = tokens.ConsumeInt();
                else if( tokens.Consume( "depth" ) )          conf.mDepthLimit          = tokens.ConsumeInt();
                else if( tokens.Consume( "nodes" ) )          conf.mNodesLimit          = tokens.ConsumeInt();
                else if( tokens.Consume( "movetime" ) )       conf.mTimeLimit           = tokens.ConsumeInt();
                else if( tokens.Consume( "infinite" ) )       conf.mTimeLimit           = 0;
                else if( tokens.Consume( "searchmoves" ) )
                {
                    for( const char* movetext = tokens.ConsumeNext(); movetext; movetext = tokens.ConsumeNext() )
                    {
                        MoveSpec spec;
                        FEN::StringToMoveSpec( movetext, spec );

                        conf.mLimitMoves.Append( spec );
                    }
                }
                else if( tokens.Consume( "ponder" ) )
                {
                    printf( "info string WARNING: ponder not implemented\n" );
                }
                else
                    break;
            }

            if( conf.mMateSearchDepth )
                printf( "info string WARNING: mate search not implemented\n" );

            if( conf.mNodesLimit )
                printf( "info string WARNING: limiting by node count not implemented\n" );

            engine->Go( &conf );
        }
        else if( tokens.Consume( "stop" ) )
        {
            engine->Stop();
        }
        else if( tokens.Consume( "ponderhit" ) )
        {
            printf( "info string WARNING: ponderhit not implemented\n" );
            engine->PonderHit();
        }
        else if( tokens.Consume( "quit" ) )
        {
            engine->Stop();

            return( true );
        }
        else
        {
            // NOT UCI COMMANDS

            if( tokens.Consume( "autotune" ) )
            {
                const char* filename = "d:\\chess\\pgn\\all-train.txt";//tokens.ConsumeAll();
                if( filename != NULL )
                {
                    AutoTuner autoTuner;

                    //FILE* fenFile = fopen( filename, "r" );
                    //if( fenFile != NULL )
                    //{
                    //    u64 lineIdx = 0;
                    //
                    //    for( ;; )
                    //    {
                    //        char line[8192];
                    //
                    //        if( !fgets( line, sizeof( line ), fenFile ) )
                    //            break;
                    //    
                    //        lineIdx++;
                    //        if( !autoTuner.LoadGameLine( line ) )
                    //        {
                    //            //printf( "\ninvalid game line %d: %s\n", lineIdx, line );
                    //            //break;
                    //        }
                    //
                    //        if( (lineIdx % 10000) == 0 )
                    //            printf( "/" );
                    //        //if( lineIdx >= 250000 )
                    //        //    break;
                    //    }
                    //
                    //    printf( "\n%d games loaded\n", lineIdx );
                    //    fclose( fenFile );
                    //}

                    //autoTuner.Deserialize( "d:\\chess\\pgn\\all-train-end.bin" );
                    autoTuner.Deserialize( "d:\\chess\\pgn\\all-train.bin" );
                    //autoTuner.Serialize( "d:\\chess\\pgn\\all-train.bin" );

                    for( int cat = 0; cat < 54; cat++ )
                    {
                        printf( "Starting category %d\n", cat );

                        autoTuner.PrepForCategory( cat );
                        autoTuner.Reset();

                        ParameterSet orig = autoTuner.GetBest();
                        ParameterSet prev = orig;

                        int timesLowProgress = 0;

                        for( ;; )
                        {
                            autoTuner.Step();

                            const ParameterSet& best = autoTuner.GetBest();
                            u64 iters = autoTuner.GetIterCount();


                            if( iters > 50000 )
                                break;

                            if( ((iters % 100) == 0) && (best.mError < prev.mError) )
                            {
                                autoTuner.Dump( true );

                                double improvement = prev.mError - best.mError;
                                timesLowProgress = (improvement < 0.000001)? (timesLowProgress + 1) : 0;

                                if( (iters >= 10000) || (timesLowProgress > 2) )
                                    break;


                                //printf( "\n" );
                                //for( int pieces = 0; pieces < 4; pieces++ )
                                //    printf( "%16.10f", best.mElem[pieces] );
                                //printf( "\n\n" );

                            
                                for( int pieces = 0; pieces < 6; pieces++ )
                                {
                                    double pieceBase = best.mElem[pieces];
                            
                                    for( int i = 0; i < 64; i++ )
                                    {
                                        if( (i > 0) && (i % 8 == 0) )
                                            printf( "\n" );
                            
                                        int idx = 6 + (pieces * 64) + i;
                                        printf( "%5.0f", pieceBase + best.mElem[idx] );
                                    }
                                    printf( "\n\n" );
                                }

                                prev = best;
                            }
                        }
                    }

                }
            }
            else if( tokens.Consume( "cpu" ) )
            {
                const char* cpuDesc[] = { "x64", "SSE2", "SSE4", "AVX2", "AVX3" };

                int detected = PlatDetectCpuLevel();

                if( tokens.Consume( "auto" ) )
                {
                    engine->OverrideCpuLevel( detected );
                    printf( "info string instruction set detected as %s\n", cpuDesc[detected] );
                }
                else
                {
                    for( int level = 0; level < CPU_LEVELS; level++ )
                    {
                        if( tokens.Consume( cpuDesc[level] ) )
                        {
                            engine->OverrideCpuLevel( level );
                            printf( "info string instruction set override to %s\n", cpuDesc[level] );
                        
                            if( level > detected )
                                printf( "info string WARNING: this is unlikely to end well\n" );
                        }
                    }
                }
            }
            else if( tokens.Consume( "divide" ) )
            {
                int depth = tokens.ConsumeInt();
                if( depth )
                    Perft::DividePerft( engine->GetPosition(), depth );
            }
            else if( tokens.Consume( "perft" ) )
            {
                if( tokens.Consume( "verify" ) )
                {
                    int depth = tokens.ConsumeInt();
                    bool failed = false;
                    u64 result = 0;

                    for( ;; )
                    {
                        u64 expected = tokens.ConsumeInt64();
                        if( expected == 0 )
                            break;

                        result = Perft::CalcPerft( engine->GetPosition(), depth );
                        failed = (result != expected);

                        if( failed )
                        {
                            printf( "info string ERROR: perft depth %d nodes %-12I64d expected %-12I64d %s\n", depth, result, expected, failed? "[ERROR]" : "" );
                            //break;
                        }

                        depth++;

                        if( failed )
                        {
                            Perft::DividePerft( engine->GetPosition(), depth );

                            printf( "info string ERROR: perft validation FAILED\n" );
                            return( false );
                        }
                    }

                    if( !failed )
                        printf( "info string perft depth %d nodes %-12I64d\n", depth, result );
                }
                else
                {
                    int depth  = tokens.ConsumeInt();
                    u64 result = Perft::CalcPerft( engine->GetPosition(), depth );

                    printf( "info string depth %d nodes %" PRId64 "\n", depth, result );
                }
            }
            else if( tokens.Consume( "run" ) )
            {
                const char* filename = tokens.ConsumeAll();
                if( filename != NULL )
                {
                    FILE* script = fopen( filename, "r" );
                    if( script != NULL )
                    {
                        for( ;; )
                        {
                            char line[2048];

                            if( !fgets( line, sizeof( line ), script ) )
                                break;
                             
                            ProcessCommand( engine, line );
                        }

                        fclose( script );
                    }
                }
            }
            else if( tokens.Consume( "time" ) )
            {
                Timer   commandTimer;

                bool    result  = ProcessCommand( engine, tokens.ConsumeAll() );
                i64     elapsed = commandTimer.GetElapsedMs();
                i64     secs    = elapsed / 1000;
                i64     msecs   = elapsed - (secs * 1000);

                printf( "info string %" PRId64 ".%03I64d sec for command \"%s\"\n", secs, msecs, str );        
                return( result );
            }
            else if( tokens.Consume( "tune" ) )
            {
                const char* name = tokens.ConsumeNext();

                if( name )
                {
                    float openingVal = tokens.ConsumeFloat();
                    float midgameVal = tokens.ConsumeFloat();
                    float endgameVal = tokens.ConsumeFloat();

                    engine->LoadWeightParam( name, openingVal, midgameVal, endgameVal );
                }
            }
        }

        return( false );
    }

};


#endif // PIGEON_UCI_H__
};
