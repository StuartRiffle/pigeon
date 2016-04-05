// eval.h - PIGEON CHESS ENGINE (c) 2012-2016 Stuart Riffle

namespace Pigeon {
#ifndef PIGEON_EVAL_H__
#define PIGEON_EVAL_H__

enum
{
    #define DECLARE_EVAL_TERM( _TERM )  _TERM,
    #include "terms.inc"

    EVAL_TERMS
};




class Evaluator
{
    EvalTerm        mWeightsOpening[EVAL_TERMS];
    EvalTerm        mWeightsMidgame[EVAL_TERMS];
    EvalTerm        mWeightsEndgame[EVAL_TERMS];

public:
    Evaluator()
    {
        PlatClearMemory( mWeightsOpening, sizeof( mWeightsOpening ) );
        PlatClearMemory( mWeightsMidgame, sizeof( mWeightsMidgame ) );
        PlatClearMemory( mWeightsEndgame, sizeof( mWeightsEndgame ) );

        this->SetWeight( EVAL_KINGS,                   0,      0,      0 );
        this->SetWeight( EVAL_QUEENS,                900,    900,    900 );
        this->SetWeight( EVAL_ROOKS,                 500,    500,    500 );
        this->SetWeight( EVAL_BISHOPS,               300,    300,    300 );
        this->SetWeight( EVAL_KNIGHTS,               400,    300,    300 );
        this->SetWeight( EVAL_PAWNS,                 100,    100,    100 );
        this->SetWeight( EVAL_MOBILITY,               10,      7,      7 );
        this->SetWeight( EVAL_ATTACKING,              50,     10,      5 );
        this->SetWeight( EVAL_DEFENDING,              10,      5,     10 );
        this->SetWeight( EVAL_ENEMY_TERRITORY,        20,     10,      0 );
        this->SetWeight( EVAL_CENTER_PAWNS,           20,     10,      0 );
        this->SetWeight( EVAL_CENTER_PIECES,          50,     10,      0 );
        this->SetWeight( EVAL_CENTER_CONTROL,         10,      5,      0 );
        this->SetWeight( EVAL_KNIGHTS_DEVEL,          50,      0,      0 );
        this->SetWeight( EVAL_BISHOPS_DEVEL,          50,      0,      0 );
        this->SetWeight( EVAL_ROOKS_DEVEL,           100,      0,      0 );
        this->SetWeight( EVAL_QUEEN_DEVEL,            20,     20,      0 );
        this->SetWeight( EVAL_PROMOTING_SOON,          0,    200,    400 );
        this->SetWeight( EVAL_PROMOTING_IMMED,         0,    300,    500 );
        this->SetWeight( EVAL_CHAINED_PAWNS,          10,      5,      0 );
        this->SetWeight( EVAL_PASSED_PAWNS,           50,     10,     20 );
        this->SetWeight( EVAL_KNIGHTS_FIRST,          50,      0,      0 );
        this->SetWeight( EVAL_KNIGHTS_NOT_RIM,        50,     50,     30 );
        this->SetWeight( EVAL_BOTH_BISHOPS,            0,     50,    100 );
        this->SetWeight( EVAL_ROOK_ON_RANK_7,          0,    100,      0 );
        this->SetWeight( EVAL_ROOKS_CONNECTED,        50,     50,     20 );
        this->SetWeight( EVAL_ROOKS_OPEN_FILE,         0,    100,      0 );
        this->SetWeight( EVAL_KING_CASTLED,           50,     50,      0 );
        this->SetWeight( EVAL_PAWNS_GUARD_KING,       20,     10,      0 );
    }

    const char* GetWeightName( int idx ) const
    {
        static const char* weightTermNames[] = 
        {
            #define DECLARE_EVAL_TERM( _TERM )  #_TERM,
            #include "terms.inc"
        };

        if( (idx >= 0) && (idx < EVAL_TERMS) )
            return( weightTermNames[idx] );

        return( nullptr );
    }

    int GetWeightIdx( const char* name ) const 
    {
        for( int idx = 0; idx < EVAL_TERMS; idx++ )
        {
            const char* weightName = this->GetWeightName( idx );

            if( weightName && (stricmp( name, weightName ) == 0) )
                return( idx );
        }

        return( -1 );
    }

    void SetWeight( int idx, int openingVal, int midgameVal, int endgameVal )
    {
        mWeightsOpening[idx] = openingVal;
        mWeightsMidgame[idx] = midgameVal;
        mWeightsEndgame[idx] = endgameVal;
    }

    float CalcGamePhase( const Position& pos ) const
    {
        // "openingness" starts at 1, then reduces to 0 over at most EVAL_OPENING_PLIES
        // "endingness" starts at 0, then increases as minor/major pieces are taken

        int     ply                 = pos.GetPlyZeroBased();
        int     whitePawnCount      = (int) CountBits( pos.mWhitePawns );
        int     whiteMinorCount     = (int) CountBits( pos.mWhiteKnights | pos.mWhiteBishops );
        int     whiteMajorCount     = (int) CountBits( pos.mWhiteRooks   | pos.mWhiteQueens );
        int     whitePieceCount     = whitePawnCount + whiteMinorCount + whiteMajorCount;
        int     blackPawnCount      = (int) CountBits( pos.mBlackPawns ); 
        int     blackMinorCount     = (int) CountBits( pos.mBlackKnights | pos.mBlackBishops );
        int     blackMajorCount     = (int) CountBits( pos.mBlackRooks   | pos.mBlackQueens );
        int     blackPieceCount     = blackPawnCount + blackMinorCount + blackMajorCount;
        int     lowestPieceCount    = Min( whitePieceCount, blackPieceCount );
        float   fightingSpirit      = lowestPieceCount / 15.0f; // (king not counted)
        float   openingness         = Max( 0.0f, fightingSpirit - (ply * 1.0f / EVAL_OPENING_PLIES) );
        int     bigCaptures         = 14 - (blackMinorCount + blackMajorCount + whiteMinorCount + whiteMajorCount);
        float   endingness          = Max( 0.0f, bigCaptures / 14.0f );

        return( (openingness > 0)? (1 - openingness) : (1 + endingness) );
    }


    void GenerateWeights( EvalTerm* weights, float gamePhase ) const
    {
        float   openingPct  = 1 - Max( 0.0f, Min( 1.0f, gamePhase ) );
        float   endgamePct  = Max( 0.0f, Min( 1.0f, gamePhase - 1 ) );
        float   midgamePct  = 1 - (openingPct + endgamePct);

        for( int i = 0; i < EVAL_TERMS; i++ )
            weights[i] = (EvalTerm) ((mWeightsOpening[i] * openingPct) + (mWeightsMidgame[i] * midgamePct) + (mWeightsEndgame[i] * endgamePct));
    }


    template< typename SIMD >
    SIMD Evaluate( const PositionT< SIMD >& pos, const EvalTerm* weights ) const
    {
        PositionT< SIMD > flipped;
        flipped.FlipFrom( pos );

        SIMD evalAsWhite = this->EvalSide( pos,     weights );
        SIMD evalAsBlack = this->EvalSide( flipped, weights );

        return( evalAsWhite - evalAsBlack );
    }


    template< typename SIMD >
    SIMD EvalSide( const PositionT< SIMD >& pos, const EvalTerm* weights ) const
    {
        SIMD    whitePawns          = pos.mWhitePawns;    
        SIMD    whiteKnights        = pos.mWhiteKnights;  
        SIMD    whiteBishops        = pos.mWhiteBishops;  
        SIMD    whiteRooks          = pos.mWhiteRooks;    
        SIMD    whiteQueens         = pos.mWhiteQueens;   
        SIMD    whiteKing           = pos.mWhiteKing;     
        SIMD    blackPawns          = pos.mBlackPawns;    
        SIMD    blackKnights        = pos.mBlackKnights;  
        SIMD    blackBishops        = pos.mBlackBishops;  
        SIMD    blackRooks          = pos.mBlackRooks;    
        SIMD    blackQueens         = pos.mBlackQueens;   
        SIMD    blackKing           = pos.mBlackKing;     
        SIMD    whiteDiag           = whiteBishops | whiteQueens;
        SIMD    whiteOrtho          = whiteRooks | whiteQueens;
        SIMD    whitePieces         = whitePawns | whiteKnights | whiteBishops | whiteRooks | whiteQueens | whiteKing;
        SIMD    blackPieces         = blackPawns | blackKnights | blackBishops | blackRooks | blackQueens | blackKing;
        SIMD    allPieces           = blackPieces | whitePieces;
        SIMD    empty               = ~allPieces;
        SIMD    pawnsMobility       = StepN( whitePawns ) & empty;
        SIMD    pawnsChained        = (StepNW( whitePawns ) | StepSW( whitePawns ) | StepSE( whitePawns ) | StepNE( whitePawns )) & whitePawns;
        SIMD    pawnsControl        = MaskOut( StepNW( whitePawns ) | StepNE( whitePawns ), whitePieces );
        SIMD    knightsControl      = StepKnights( whiteKnights );
        SIMD    diagControl         = SlideIntoExDiag( whiteDiag, empty, allPieces );
        SIMD    orthoControl        = SlideIntoExDiag( whiteOrtho, empty, allPieces );
        SIMD    kingControl         = StepOut( whiteKing );
        SIMD    whiteControl        = pawnsControl | knightsControl | diagControl | orthoControl | kingControl;
        SIMD    whiteMobility       = whiteControl & empty;
        SIMD    whiteAttacking      = whiteControl & blackPieces;
        SIMD    whiteDefending      = whiteControl & whitePieces;
        SIMD    inEnemyTerritory    = whitePieces & (RANK_5 | RANK_6 | RANK_7 | RANK_8);
        SIMD    evalKnightsDevel    = CountBits( whiteKnights & ~(SQUARE_B1 | SQUARE_G1) );
        SIMD    evalBishopsDevel    = CountBits( whiteBishops & ~(SQUARE_C1 | SQUARE_F1) );
        SIMD    evalKnightsFirst    = Max( (SIMD) 0, evalKnightsDevel - evalBishopsDevel );
        SIMD    evalBothBishops     = SelectIfNotZero( whiteBishops & LIGHT_SQUARES, (SIMD) 1 ) & SelectIfNotZero( whiteBishops & DARK_SQUARES, (SIMD) 1 );
        SIMD    evalRooksConnected  = CountBits( PropExOrtho( whiteRooks, empty ) & whiteRooks );
        SIMD    evalPawnsGuardKing  = CountBits( whitePawns & (StepNW( whiteKing ) | StepN( whiteKing ) | StepNE( whiteKing )) );
        SIMD    score               = MulLow32( CountBits( whitePawns ),                                weights[EVAL_PAWNS]            ) +
                                      MulLow32( CountBits( whiteKnights ),                              weights[EVAL_KNIGHTS]          ) +
                                      MulLow32( CountBits( whiteBishops ),                              weights[EVAL_BISHOPS]          ) +
                                      MulLow32( CountBits( whiteRooks ),                                weights[EVAL_ROOKS]            ) +
                                      MulLow32( CountBits( whiteQueens ),                               weights[EVAL_QUEENS]           ) +
                                      MulLow32( CountBits( whiteKing ),                                 weights[EVAL_KINGS]            ) +
                                      MulLow32( CountBits( whiteMobility ),                             weights[EVAL_MOBILITY]         ) +
                                      MulLow32( CountBits( whiteAttacking ),                            weights[EVAL_ATTACKING]        ) +
                                      MulLow32( CountBits( whiteDefending ),                            weights[EVAL_DEFENDING]        ) +
                                      MulLow32( CountBits( inEnemyTerritory ),                          weights[EVAL_ENEMY_TERRITORY]  ) +
                                      MulLow32( CountBits( whitePawns   & CENTER_SQUARES ),             weights[EVAL_CENTER_PAWNS]     ) +
                                      MulLow32( CountBits( whitePieces  & CENTER_SQUARES ),             weights[EVAL_CENTER_PIECES]    ) +
                                      MulLow32( CountBits( whiteControl & CENTER_SQUARES ),             weights[EVAL_CENTER_CONTROL]   ) +
                                      MulLow32( evalKnightsDevel,                                       weights[EVAL_KNIGHTS_DEVEL]    ) +
                                      MulLow32( evalBishopsDevel,                                       weights[EVAL_BISHOPS_DEVEL]    ) +
                                      MulLow32( CountBits( whiteRooks   & ~(SQUARE_A1 | SQUARE_H1) ),   weights[EVAL_ROOKS_DEVEL]      ) +
                                      MulLow32( CountBits( whiteRooks   & ~(SQUARE_D1) ),               weights[EVAL_QUEEN_DEVEL]      ) +
                                      MulLow32( CountBits( whitePawns   & RANK_6 ),                     weights[EVAL_PROMOTING_SOON]   ) +
                                      MulLow32( CountBits( whitePawns   & RANK_7 ),                     weights[EVAL_PROMOTING_IMMED]  ) +
                                      MulLow32( CountBits( pawnsChained ),                              weights[EVAL_CHAINED_PAWNS]    ) +
                                      MulLow32( CountBits( PropN( whitePawns, ~blackPawns ) & RANK_8 ), weights[EVAL_PASSED_PAWNS]     ) +
                                      MulLow32( evalKnightsFirst,                                       weights[EVAL_KNIGHTS_FIRST]    ) +
                                      MulLow32( CountBits( whiteKnights & ~EDGE_SQUARES ),              weights[EVAL_KNIGHTS_NOT_RIM]  ) +
                                      MulLow32( evalBothBishops,                                        weights[EVAL_BOTH_BISHOPS]     ) +
                                      MulLow32( CountBits( whiteRooks & RANK_7 ),                       weights[EVAL_ROOK_ON_RANK_7]   ) +
                                      MulLow32( evalRooksConnected,                                     weights[EVAL_ROOKS_CONNECTED]  ) +
                                      MulLow32( CountBits( PropN( whiteRooks, empty ) & RANK_8 ),       weights[EVAL_ROOKS_OPEN_FILE]  ) +
                                      MulLow32( CountBits( whiteKing & RANK_1 & ~SQUARE_E1 ),           weights[EVAL_KING_CASTLED]     ) +
                                      MulLow32( evalPawnsGuardKing,                                     weights[EVAL_PAWNS_GUARD_KING] );

        return( score );
    }
};

#endif // PIGEON_EVAL_H__
};
