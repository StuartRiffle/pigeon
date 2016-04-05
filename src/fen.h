// fen.h - PIGEON CHESS ENGINE (c) 2012-2016 Stuart Riffle

namespace Pigeon {
#ifndef PIGEON_FEN_H__
#define PIGEON_FEN_H__


class FEN
{
public:
    static void PositionToString( const Position& pos, char* str )
    {
        if( pos.mBoardFlipped )
        {
            Position flipped;
            flipped.FlipFrom( pos );

            PositionToString( flipped, str );
            return;
        }

        char* cur = str;

        for( u64 bit = 1ULL << 63; bit != 0; bit >>= 1 )
        {
            if( pos.mWhitePawns   & bit )   *cur++ = 'P'; else
            if( pos.mWhiteKnights & bit )   *cur++ = 'N'; else
            if( pos.mWhiteBishops & bit )   *cur++ = 'B'; else
            if( pos.mWhiteRooks   & bit )   *cur++ = 'R'; else
            if( pos.mWhiteQueens  & bit )   *cur++ = 'Q'; else
            if( pos.mWhiteKing    & bit )   *cur++ = 'K'; else
            if( pos.mBlackPawns   & bit )   *cur++ = 'p'; else
            if( pos.mBlackKnights & bit )   *cur++ = 'n'; else
            if( pos.mBlackBishops & bit )   *cur++ = 'b'; else
            if( pos.mBlackRooks   & bit )   *cur++ = 'r'; else
            if( pos.mBlackQueens  & bit )   *cur++ = 'q'; else
            if( pos.mBlackKing    & bit )   *cur++ = 'k'; else
            {
                if( (cur > str) && (cur[-1] >= '1') && (cur[-1] < '8') )
                    cur[-1]++;
                else
                    *cur++ = '1';
            }

            if( bit & FILE_H )
                *cur++ = (bit & RANK_1)? ' ' : '/';
        }

        str = cur;

        *str++ = pos.mWhiteToMove? 'w' : 'b';
        *str++ = ' ';

        if(   pos.mCastlingAndEP & SQUARE_H1 )     *str++ = 'K';
        if(   pos.mCastlingAndEP & SQUARE_A1 )     *str++ = 'Q';
        if(   pos.mCastlingAndEP & SQUARE_H8 )     *str++ = 'k';
        if(   pos.mCastlingAndEP & SQUARE_A8 )     *str++ = 'q';
        if( !(pos.mCastlingAndEP & CASTLE_ROOKS) ) *str++ = '-';

        *str++ = ' ';

        if( pos.mCastlingAndEP & EP_SQUARES )
        {
            int idx = (int) LowestBitIndex( (u64) (pos.mCastlingAndEP & EP_SQUARES) );
            *str++ = "hgfedcba"[idx & 7];
            *str++ = "12345678"[idx / 8]; 
            *str++ = ' ';
        }
        else
        {
            *str++ = '-';
            *str++ = ' ';
        }

        sprintf( str, "%d %d", (int) pos.mHalfmoveClock, (int) pos.mFullmoveNum );
    }


    static const char* StringToPosition( const char* str, Position& pos )
    {
        memset( &pos, 0, sizeof( pos ) );

        for( u64 bit = 1ULL << 63; bit != 0; bit >>= 1 )
        {
            switch( *str++ )
            {
            case 'P':   pos.mWhitePawns   |= bit; break;
            case 'N':   pos.mWhiteKnights |= bit; break;
            case 'B':   pos.mWhiteBishops |= bit; break;
            case 'R':   pos.mWhiteRooks   |= bit; break;
            case 'Q':   pos.mWhiteQueens  |= bit; break;
            case 'K':   pos.mWhiteKing    |= bit; break;
            case 'p':   pos.mBlackPawns   |= bit; break;
            case 'n':   pos.mBlackKnights |= bit; break;
            case 'b':   pos.mBlackBishops |= bit; break;
            case 'r':   pos.mBlackRooks   |= bit; break;
            case 'q':   pos.mBlackQueens  |= bit; break;
            case 'k':   pos.mBlackKing    |= bit; break;
            case '8':   bit >>= 7;  break;
            case '7':   bit >>= 6;  break;
            case '6':   bit >>= 5;  break;
            case '5':   bit >>= 4;  break;
            case '4':   bit >>= 3;  break;
            case '3':   bit >>= 2;  break;
            case '2':   bit >>= 1;  break;
            case '1':   break;
            case '/':   bit <<= 1;  break;
            default:    return( NULL );
            }
        }

        if( *str == ' ' ) str++;

        switch( *str++ )
        {
        case 'w':   pos.mWhiteToMove = 1; break;
        case 'b':   pos.mWhiteToMove = 0; break;
        default:    return( NULL );
        }

        if( *str == ' ' ) str++;

        while( *str && (*str != ' ') )
        {
            if( *str == 'K' )   pos.mCastlingAndEP |= SQUARE_H1; else
            if( *str == 'Q' )   pos.mCastlingAndEP |= SQUARE_A1; else
            if( *str == 'k' )   pos.mCastlingAndEP |= SQUARE_H8; else
            if( *str == 'q' )   pos.mCastlingAndEP |= SQUARE_A8; else
            if( *str != '-' )   return( NULL );

            str++;
        }

        str++;

        if( *str == '-' )
            str++;
        else
        {
            int file  = 'h' - tolower( *str++ );
            int rank  = *str++ - '1';

            if( (file | rank) & ~7 )
                return( NULL );

            int idx = (rank * 8) + file;
            pos.mCastlingAndEP |= SquareBit( idx );
        }

        str++;

        pos.mHalfmoveClock = 0;
        while( isdigit( *str ) )
            pos.mHalfmoveClock = (pos.mHalfmoveClock * 10) + (*str++ - '0');

        str++;

        pos.mFullmoveNum = 0;
        while( isdigit( *str ) )
            pos.mFullmoveNum = (pos.mFullmoveNum * 10) + (*str++ - '0');

        if( !pos.mWhiteToMove )
            pos.FlipInPlace();

        while( *str && (*str == ' ') )
            str++;
            
        return( str );
    }



    static void MoveSpecToString( const MoveSpec& spec, char* str ) 
    {
        *str++ = "hgfedcba"[spec.mSrc & 7];
        *str++ = "12345678"[spec.mSrc / 8];
        *str++ = "hgfedcba"[spec.mDest & 7];
        *str++ = "12345678"[spec.mDest / 8];
        *str++ = spec.GetPromoteChar();
        *str++ = 0;
    }


    static const char* StringToMoveSpec( const char* str, MoveSpec& spec )
    {
        int src_file0  = 'h' - tolower( *str++ );
        int src_rank0  = *str++ - '1';
        int dest_file0 = 'h' - tolower( *str++ );
        int dest_rank0 = *str++ - '1';

        if( (src_file0 | src_rank0 | dest_file0 | dest_rank0) & ~7 )
            return( NULL );

        spec.mSrc  = (src_rank0  * 8) + src_file0;
        spec.mDest = (dest_rank0 * 8) + dest_file0;
        spec.mType = MOVE;

        switch( tolower( *str ) )
        {
        case 'n':   spec.mType = PROMOTE_KNIGHT; str++; break;
        case 'b':   spec.mType = PROMOTE_BISHOP; str++; break;
        case 'r':   spec.mType = PROMOTE_ROOK;   str++; break;
        case 'q':   spec.mType = PROMOTE_QUEEN;  str++; break;
        }

        while( *str && strchr( "-+#!?", *str ) ) 
            str++;

        if( *str && !isspace( *str++ ) )    
            return( NULL );

        return( str );
    }


    static void PrintPosition( const Position& pos )
    {
        char fen[MAX_FEN_LENGTH];
        FEN::PositionToString( pos, fen );

        printf( fen );
    }
                                      

    static void PrintMoveSpec( const MoveSpec& spec )
    {
        char movetext[MAX_MOVETEXT];
        FEN::MoveSpecToString( spec, movetext );

        printf( movetext );
    }


    static void PrintMoveList( const MoveList& moves )
    {
        for( int i = 0; i < moves.mCount; i++ )
        {
            if( i > 0 )
                printf( " " );

            FEN::PrintMoveSpec( moves.mMove[i] );
        }
    }

};


#endif // PIGEON_FEN_H__
};
