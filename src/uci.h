// uci.h - PIGEON CHESS ENGINE (c) 2012-2016 Stuart Riffle

namespace Pigeon {
#ifndef PIGEON_UCI_H__
#define PIGEON_UCI_H__

struct UCI
{
    static bool ProcessCommand( Engine* engine, const char* str )
    {
        Tokenizer tokens( str );

        if( tokens.Consume( "#" ) )
        {
            // Treat line as a comment
        }
        else if( tokens.Consume( "uci" ) )
        {                                                                                        
            printf( "id name Pigeon %d.%02d\n", PIGEON_VER_MAJ, PIGEON_VER_MIN );
            printf( "id author Stuart Riffle \n" );
            printf( "option name Hash type spin default %d\n", TT_MEGS_DEFAULT );
            printf( "uciok\n" );
        }
        else if( tokens.Consume( "debug" ) )
        {
            if( tokens.Consume( "on" ) )       
                engine->SetDebug( true );
            else if( tokens.Consume( "off" ) ) 
                engine->SetDebug( false );
        }
        else if( tokens.Consume( "setoption" ) )
        {
            if( tokens.Consume( "name" ) )
            {
                if( tokens.Consume( "hash" ) && tokens.Consume( "value" ) )
                    engine->SetHashTableSize( tokens.ConsumeInt() );

            }
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
                str = engine->SetPosition( str );
                if( str == NULL )
                    return( true );
            }

            const char* movehistory = "";

            if( tokens.Consume( "moves" ) )
            {
                movehistory = str;

                for( const char* movetext = tokens.ConsumeNext(); movetext; movetext = tokens.ConsumeNext() )
                    engine->Move( movetext );
            }

           engine->PrintPosition();
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

            engine->Go( &conf );
        }
        else if( tokens.Consume( "stop" ) )
        {
            engine->Stop();
        }
        else if( tokens.Consume( "ponderhit" ) )
        {
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

            if( tokens.Consume( "book" ) )
            {
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

                    printf( "info string depth %d nodes %"PRId64"\n", depth, result );
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
                             
                            if( !ProcessCommand( engine, line ) )
                                break;
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

                printf( "info string %"PRId64".%03I64d sec for command \"%s\"\n", secs, msecs, str );        
                return( result );
            }
            else if( tokens.Consume( "tune" ) )
            {
                const char* name = tokens.ConsumeNext();

                if( name )
                {
                    int openingVal = tokens.ConsumeInt();
                    int midgameVal = tokens.ConsumeInt();
                    int endgameVal = tokens.ConsumeInt();

                    engine->LoadWeightParam( name, openingVal, midgameVal, endgameVal );
                }
            }
        }

        return( false );
    }

};


#endif // PIGEON_UCI_H__
};
