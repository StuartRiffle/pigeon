// tb-gaviota.h - PIGEON CHESS ENGINE (c) 2012-2016 Stuart Riffle

#ifndef PIGEON_TB_GAVIOTA_H__
#define PIGEON_TB_GAVIOTA_H__

extern "C" 
{
    // Gaviota tablebase probing code (c) 2010 Miguel A. Ballicora
    // Using version 0.4.5, obtained from https://github.com/michiguel/Gaviota-Tablebases
    // Thank you Miguel!

    #include "gaviota-tb-0.4.5/gtb-probe.h"  
    #include "gaviota-tb-0.4.5/gtb-dec.h"
    #include "gaviota-tb-0.4.5/gtb-att.h"
    #include "gaviota-tb-0.4.5/gtb-types.h"
};

namespace Pigeon {

struct TablebaseProbe
{
    enum
    {
        FLAG_DRAW       = (1 << 0),
        FLAG_WHITE_MATE = (1 << 1),
        FLAG_BLACK_MATE = (1 << 2)
    };

    u16     mFlags;
    i16     mPliesToMate;
};

inline int SquarePigeonToGaviota( int idx )
{
    // Pigeon squares are ordered H1-A1, H2-A2, etc.
    // Gaviota squares are ordered A1-H1, A2-H2, etc.

    return( (7 - (idx & 0x7)) | (idx & 0x38) );
}

class GaviotaTablebase
{
    struct ProbeSpec
    {
        u32     stm;        /* side to move */
        u32     epsquare;   /* target square for an en passant capture */
        u32     castling;   /* castling availability, 0 => no castles */
        u32     ws[17];     /* list of squares for white */
        u32     bs[17];     /* list of squares for black */
        u8      wp[17];     /* what white pieces are on those squares */
        u8      bp[17];     /* what black pieces are on those squares */

        void SetPosition( const Position& pos )
        {
            u64     whitePawns      = SelectWithMask( pos.mBoardFlipped, ByteSwap( pos.mBlackPawns    ), pos.mWhitePawns    );
            u64     whiteKnights    = SelectWithMask( pos.mBoardFlipped, ByteSwap( pos.mBlackKnights  ), pos.mWhiteKnights  );
            u64     whiteBishops    = SelectWithMask( pos.mBoardFlipped, ByteSwap( pos.mBlackBishops  ), pos.mWhiteBishops  );
            u64     whiteRooks      = SelectWithMask( pos.mBoardFlipped, ByteSwap( pos.mBlackRooks    ), pos.mWhiteRooks    );
            u64     whiteQueens     = SelectWithMask( pos.mBoardFlipped, ByteSwap( pos.mBlackQueens   ), pos.mWhiteQueens   );
            u64     whiteKing       = SelectWithMask( pos.mBoardFlipped, ByteSwap( pos.mBlackKing     ), pos.mWhiteKing     );
            u64     blackPawns      = SelectWithMask( pos.mBoardFlipped, ByteSwap( pos.mWhitePawns    ), pos.mBlackPawns    );
            u64     blackKnights    = SelectWithMask( pos.mBoardFlipped, ByteSwap( pos.mWhiteKnights  ), pos.mBlackKnights  );
            u64     blackBishops    = SelectWithMask( pos.mBoardFlipped, ByteSwap( pos.mWhiteBishops  ), pos.mBlackBishops  );
            u64     blackRooks      = SelectWithMask( pos.mBoardFlipped, ByteSwap( pos.mWhiteRooks    ), pos.mBlackRooks    );
            u64     blackQueens     = SelectWithMask( pos.mBoardFlipped, ByteSwap( pos.mWhiteQueens   ), pos.mBlackQueens   );
            u64     blackKing       = SelectWithMask( pos.mBoardFlipped, ByteSwap( pos.mWhiteKing     ), pos.mBlackKing     );
            u64     castlingAndEP   = SelectWithMask( pos.mBoardFlipped, ByteSwap( pos.mCastlingAndEP ), pos.mCastlingAndEP );

            u32*    whiteSquare     = ws;
            u32*    blackSquare     = bs;
            u8*     whitePiece      = wp;
            u8*     blackPiece      = bp;

            for( u64 i = whitePawns;   i != 0; *whitePiece++ = tb_PAWN   ) *whiteSquare++ = SquarePigeonToGaviota( (int) ConsumeLowestBitIndex( i ) );
            for( u64 i = whiteKnights; i != 0; *whitePiece++ = tb_KNIGHT ) *whiteSquare++ = SquarePigeonToGaviota( (int) ConsumeLowestBitIndex( i ) );
            for( u64 i = whiteBishops; i != 0; *whitePiece++ = tb_BISHOP ) *whiteSquare++ = SquarePigeonToGaviota( (int) ConsumeLowestBitIndex( i ) );
            for( u64 i = whiteRooks;   i != 0; *whitePiece++ = tb_ROOK   ) *whiteSquare++ = SquarePigeonToGaviota( (int) ConsumeLowestBitIndex( i ) );
            for( u64 i = whiteQueens;  i != 0; *whitePiece++ = tb_QUEEN  ) *whiteSquare++ = SquarePigeonToGaviota( (int) ConsumeLowestBitIndex( i ) );

            for( u64 i = blackPawns;   i != 0; *blackPiece++ = tb_PAWN   ) *blackSquare++ = SquarePigeonToGaviota( (int) ConsumeLowestBitIndex( i ) );
            for( u64 i = blackKnights; i != 0; *blackPiece++ = tb_KNIGHT ) *blackSquare++ = SquarePigeonToGaviota( (int) ConsumeLowestBitIndex( i ) );
            for( u64 i = blackBishops; i != 0; *blackPiece++ = tb_BISHOP ) *blackSquare++ = SquarePigeonToGaviota( (int) ConsumeLowestBitIndex( i ) );
            for( u64 i = blackRooks;   i != 0; *blackPiece++ = tb_ROOK   ) *blackSquare++ = SquarePigeonToGaviota( (int) ConsumeLowestBitIndex( i ) );
            for( u64 i = blackQueens;  i != 0; *blackPiece++ = tb_QUEEN  ) *blackSquare++ = SquarePigeonToGaviota( (int) ConsumeLowestBitIndex( i ) );

            *whiteSquare++  = SquarePigeonToGaviota( (int) LowestBitIndex( whiteKing ) );
            *whitePiece++   = tb_KING;
            *whiteSquare    = tb_NOSQUARE;
            *whitePiece     = tb_NOPIECE;

            *blackSquare++  = SquarePigeonToGaviota( (int) LowestBitIndex( blackKing ) );
            *blackPiece++   = tb_KING;
            *blackSquare    = tb_NOSQUARE;
            *blackPiece     = tb_NOPIECE;

            stm             = pos.mWhiteToMove? tb_WHITE_TO_MOVE : tb_BLACK_TO_MOVE;
            epsquare        = tb_NOSQUARE;
            castling        = 0;

            if( castlingAndEP & SQUARE_H1 )  castling |= tb_WOO;
            if( castlingAndEP & SQUARE_A1 )  castling |= tb_WOOO;
            if( castlingAndEP & SQUARE_H8 )  castling |= tb_BOO;
            if( castlingAndEP & SQUARE_A8 )  castling |= tb_BOOO;

            if( castlingAndEP & EP_SQUARES ) epsquare = SquarePigeonToGaviota( (int) LowestBitIndex( castlingAndEP & EP_SQUARES ) );
        }
    };

public:

    enum
    {
        GAVIOTA_TB_VERBOSE  = 1,
        GAVIOTA_TB_WDL_FRAC = 96,
    };

    const char**    mPaths;
    char*           mInitInfo;
    size_t          mCacheSize;
    bool            mPieceDataAvail[32];
    bool            mInitialized;


    GaviotaTablebase()
    {
        this->Clear();
    }

    ~GaviotaTablebase()
    {
        this->Shutdown();
    }

    bool IsInitialized() const
    {
        return( mInitialized );
    }

    void Init( const char* path, int cacheMegs )
    {
        this->Shutdown();

        mPaths      = tbpaths_init();
        mPaths      = tbpaths_add( mPaths, path );
        mInitInfo   = tb_init( GAVIOTA_TB_VERBOSE, tb_CP4, mPaths );
        mCacheSize  = cacheMegs * 1024 * 1024;

        tbcache_init( mCacheSize, GAVIOTA_TB_WDL_FRAC );
        tbstats_reset();

        if( mInitInfo )
            printf( "info string GAVIOTA: %s\n", mInitInfo );

        unsigned av = tb_availability();

        if( av & (0x01 | 0x02) )  mPieceDataAvail[3] = true;
        if( av & (0x04 | 0x08) )  mPieceDataAvail[4] = true;
        if( av & (0x0F | 0x10) )  mPieceDataAvail[5] = true;

        mInitialized = true;
    }

    void Shutdown()
    {
        if( mPaths )
        {
            tbcache_done();
            tb_done();
            tbpaths_done( mPaths );
        }

        this->Clear();
    }

    template< int POPCNT >
    bool Probe( const Position& pos, TablebaseProbe& result, bool dtmCheck = false )
    {
        if( pos.mFullmoveNum < TABLEBASE_MIN_MOVES )
            return( false );

        int nonKingPieces = (int) CountBits< POPCNT >(
            pos.mWhitePawns   |
            pos.mWhiteKnights |
            pos.mWhiteBishops |
            pos.mWhiteRooks   |
            pos.mWhiteQueens  |
            pos.mBlackPawns   |
            pos.mBlackKnights |
            pos.mBlackBishops |
            pos.mBlackRooks   |
            pos.mBlackQueens  );

        int totalPieces = nonKingPieces + 2;
        if( !mPieceDataAvail[totalPieces] )
            return( false );

        if( this->LookupByHash( pos.mHash, result ) )
            return( true );

        ProbeSpec spec;
        spec.SetPosition( pos );

        u32 info        = tb_UNKNOWN;
        u32 pliesToMate = 0;
        int tbAvail     = 0;

        if( dtmCheck )
            tbAvail = tb_probe_hard( spec.stm, spec.epsquare, spec.castling, spec.ws, spec.bs, spec.wp, spec.bp, &info, &pliesToMate );
        else
            tbAvail = tb_probe_WDL_hard( spec.stm, spec.epsquare, spec.castling, spec.ws, spec.bs, spec.wp, spec.bp, &info );

        if( tbAvail )
        {
            result.mPliesToMate = pliesToMate;
            result.mFlags       = 0;

            switch( info )
            {
            case tb_DRAW:   result.mFlags |= TablebaseProbe::FLAG_DRAW;       break;
            case tb_WMATE:  result.mFlags |= TablebaseProbe::FLAG_WHITE_MATE; break;
            case tb_BMATE:  result.mFlags |= TablebaseProbe::FLAG_BLACK_MATE; break;
            }

            this->CacheResult( pos.mHash, result );
            return( true );
        }

        return( false );
    }

private:

    void Clear()
    {
        mPaths          = NULL;
        mInitInfo       = NULL;
        mCacheSize      = 0;
        mInitialized    = false;

        PlatClearMemory( mPieceDataAvail, sizeof( mPieceDataAvail ) );
    }

    bool LookupByHash( u64 hash, TablebaseProbe& result )
    {
        return( false );
    }

    void CacheResult( u64 hash, const TablebaseProbe& result )
    {
    }
};

};
#endif // PIGEON_TB_GAVIOTA_H__
