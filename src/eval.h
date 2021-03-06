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
    float       mWeightsOpening[EVAL_TERMS];
    float       mWeightsMidgame[EVAL_TERMS];
    float       mWeightsEndgame[EVAL_TERMS];
    float       mWeightsTuning[EVAL_TERMS];
    bool        mEnableOpening;
    bool        mEnableTuning;

public:
    Evaluator()
    {
        PlatClearMemory( mWeightsOpening, sizeof( mWeightsOpening ) );
        PlatClearMemory( mWeightsMidgame, sizeof( mWeightsMidgame ) );
        PlatClearMemory( mWeightsEndgame, sizeof( mWeightsEndgame ) );

        mEnableOpening  = true;
        mEnableTuning   = false;

        this->SetWeight( EVAL_PAWNS,                100.00,    100.00,    100.00 );   //    100,    100.00
        this->SetWeight( EVAL_CENTER_PAWNS,          10.00,      0.00,      0.00 );   //     10,      0.00
        this->SetWeight( EVAL_CHAINED_PAWNS,         10.00,      2.75,      2.75 );   //     30,      2.75
        this->SetWeight( EVAL_PASSED_PAWNS,           0.01,      0.01,      0.01 );   //     20,      0.01
        this->SetWeight( EVAL_PAWNS_GUARD_KING,      10.00,      5.62,      5.62 );   //     10,      5.62
        this->SetWeight( EVAL_PROMOTING_SOON,        32.98,     32.98,     50.00 );   //     10,     32.98
        this->SetWeight( EVAL_PROMOTING_IMMED,       80.09,     80.09,    100.00 );   //     20,     80.09
        this->SetWeight( EVAL_KNIGHTS,              262.62,    262.62,    262.62 );   //    300,    262.62
        this->SetWeight( EVAL_KNIGHTS_DEVEL,         15.00,      0.00,      0.00 );   //     10,      0.03
        this->SetWeight( EVAL_KNIGHTS_FIRST,         10.00,      0.00,      0.00 );   //      0,      0.09
        this->SetWeight( EVAL_KNIGHTS_INTERIOR,      15.10,     15.10,     15.10 );   //     20,     15.10
        this->SetWeight( EVAL_KNIGHTS_CENTRAL,       20.17,     20.17,     20.17 );   //     20,     20.17
        this->SetWeight( EVAL_BISHOPS,              282.00,    282.00,    282.00 );   //    320,    275.25
        this->SetWeight( EVAL_BISHOPS_DEVEL,         10.00,      0.00,      0.00 );   //     20,      6.36
        this->SetWeight( EVAL_BOTH_BISHOPS,          33.25,     33.25,     33.25 );   //     20,     33.25
        this->SetWeight( EVAL_BISHOPS_INTERIOR,       6.26,      6.26,      6.26 );   //     20,      6.26
        this->SetWeight( EVAL_BISHOPS_CENTRAL,       16.72,     16.72,     16.72 );   //     20,     16.72
        this->SetWeight( EVAL_ROOKS,                453.00,    453.00,    453.00 );   //    500,    433.85
        this->SetWeight( EVAL_ROOKS_DEVEL,           10.00,      0.00,      0.00 );   //     20,     19.17
        this->SetWeight( EVAL_ROOK_ON_RANK_7,        15.69,     15.69,     15.69 );   //     30,     15.69
        this->SetWeight( EVAL_ROOKS_CONNECTED,        0.00,      0.00,      0.00 );   //     30,      0.00
        this->SetWeight( EVAL_ROOKS_OPEN_FILE,       33.30,     33.30,     33.30 );   //     40,     33.30
        this->SetWeight( EVAL_QUEENS,               920.00,    920.00,    920.00 );   //    900,    918.40
        this->SetWeight( EVAL_QUEEN_DEVEL,           10.00,      0.00,      0.00 );   //     10,      0.11
        this->SetWeight( EVAL_QUEENS_INTERIOR,        4.57,      4.57,      4.57 );   //     20,      4.57
        this->SetWeight( EVAL_QUEENS_CENTRAL,         8.72,      8.72,      8.72 );   //     20,      8.72
        this->SetWeight( EVAL_KINGS,              20000.00,  20000.00,  20000.00 );   //  10000,  20000.00
        this->SetWeight( EVAL_KING_CASTLED,          30.00,     10.00,      0.00 );   //     30,      0.14
        this->SetWeight( EVAL_MOBILITY,               2.13,      2.13,      2.13 );   //      5,      2.13
        this->SetWeight( EVAL_ATTACKING,              8.14,      8.14,      8.14 );   //     10,      8.14
        this->SetWeight( EVAL_DEFENDING,              0.02,      0.02,      0.02 );   //     10,      0.02
        this->SetWeight( EVAL_ENEMY_TERRITORY,       10.00,     24.09,     10.00 );   //     10,     24.09
        this->SetWeight( EVAL_CENTER_PIECES,         10.00,      0.00,      0.00 );   //     20,      0.01
        this->SetWeight( EVAL_CENTER_CONTROL,         5.00,      1.69,      0.00 );   //     30,      1.69
    }

    void EnableOpening( bool enable )
    {
        mEnableOpening = enable;
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

        return( NULL );
    }

    int GetWeightIdx( const char* name ) const 
    {
        for( int idx = 0; idx < EVAL_TERMS; idx++ )
        {
            const char* weightName = this->GetWeightName( idx );

            if( weightName && (strnicmp( name, weightName, strlen( weightName ) ) == 0) )
                return( idx );
        }

        return( -1 );
    }

    void SetWeight( int idx, float openingVal, float midgameVal, float endgameVal )
    {
        mWeightsOpening[idx] = openingVal;
        mWeightsMidgame[idx] = midgameVal;
        mWeightsEndgame[idx] = endgameVal;
    }

    template< int POPCNT >
    PDECL float CalcGamePhase( const Position& pos ) const
    {
        // "openingness" starts at 1, then reduces to 0 over at most EVAL_OPENING_PLIES
        // "endingness" starts at 0, then increases as minor/major pieces are taken

        int     ply                 = pos.GetPlyZeroBased();
        int     whitePawnCount      = (int) CountBits< POPCNT >( pos.mWhitePawns );
        int     whiteMinorCount     = (int) CountBits< POPCNT >( pos.mWhiteKnights | pos.mWhiteBishops );
        int     whiteMajorCount     = (int) CountBits< POPCNT >( pos.mWhiteRooks   | pos.mWhiteQueens );
        int     whitePieceCount     = whitePawnCount + whiteMinorCount + whiteMajorCount;
        int     blackPawnCount      = (int) CountBits< POPCNT >( pos.mBlackPawns ); 
        int     blackMinorCount     = (int) CountBits< POPCNT >( pos.mBlackKnights | pos.mBlackBishops );
        int     blackMajorCount     = (int) CountBits< POPCNT >( pos.mBlackRooks   | pos.mBlackQueens );
        int     blackPieceCount     = blackPawnCount + blackMinorCount + blackMajorCount;
        int     lowestPieceCount    = Min( whitePieceCount, blackPieceCount );
        float   fightingSpirit      = lowestPieceCount / 15.0f; // (king not counted)
        float   openingness         = Max( 0.0f, fightingSpirit - (ply * 1.0f / EVAL_OPENING_PLIES) );
        int     bigCaptures         = 14 - (blackMinorCount + blackMajorCount + whiteMinorCount + whiteMajorCount);
        float   endingness          = Max( 0.0f, bigCaptures / 14.0f );

        if( !mEnableOpening )
            openingness = 0;

        return( (openingness > 0)? (1 - openingness) : (1 + endingness) );
    }


    template< int POPCNT >
    void AdjustWeights( const Position& pos0, EvalTerm score0, const Position& pos1, EvalTerm score1, float learningRate )
    {
        /*
        float       phase0                  = this->CalcGamePhase< POPCNT >( pos0 );
        float       phase1                  = this->CalcGamePhase< POPCNT >( pos1 );
        MoveMap     mmap0;                  pos0.CalcMoveMap( &mmap0 );
        MoveMap     mmap1;                  pos1.CalcMoveMap( &mmap1 );
        MoveList    moves0;                 moves0.UnpackMoveMap( pos0, mmap0 );
        MoveList    moves1;                 moves1.UnpackMoveMap( pos1, mmap1 );

        EvalWeight  weights0[EVAL_TERMS];   this->GenerateWeights( weights0, phase0 );
        EvalWeight  weights1[EVAL_TERMS];   this->GenerateWeights( weights1, phase1 );
        u64         eval0[EVAL_TERMS];      this->CalcEvalTerms< POPCNT, u64 >( pos0, mmap0, eval0 );
        u64         eval1[EVAL_TERMS];      this->CalcEvalTerms< POPCNT, u64 >( pos1, mmap1, eval1 );

        float       advantage0              = 1.0f / (1.0f + exp( -(score0 * 0.1f) ));
        float       advantage1              = 1.0f / (1.0f + exp( -(score1 * 0.1f) ));
        float       error                   = advantage1 - advantage0;
        float       openingPct              = 1 - Max( 0.0f, Min( 1.0f, phase0 ) );
        float       endgamePct              = Max( 0.0f, Min( 1.0f, phase0 - 1 ) );
        float       midgamePct              = 1 - (openingPct + endgamePct);

        for( int i = 0; i < EVAL_TERMS; i++ )
        {
            float gradient = (float) ((i32) ((eval0[i] * weights0[i]) >> WEIGHT_SHIFT));
            float adjust   = learningRate * error * (1 - error) * -gradient;

            mWeightsOpening[i] += Max( 0.0f, adjust * openingPct );
            mWeightsMidgame[i] += Max( 0.0f, adjust * midgamePct );
            mWeightsEndgame[i] += Max( 0.0f, adjust * endgamePct );
        }
        */

        //FILE* dump = fopen( "autotune.coo", "w" );
        //for( int i = 0; i < EVAL_TERMS; i++ )
        //    fprintf( dump, "tune %-25s %14.7f %14.7f %14.7f\n", this->GetWeightName( i ),  mWeightsOpening[i],  mWeightsMidgame[i],  mWeightsEndgame[i] );
        //fclose( dump );
    }


    PDECL void GenerateWeights( EvalWeight* weights, float gamePhase ) const
    {
        if( mEnableTuning )
        {
            for( int i = 0; i < EVAL_TERMS; i++ )
                weights[i] = (EvalWeight) (mWeightsTuning[i] * WEIGHT_SCALE);
        }
        else
        {
            float   openingPct  = 1 - Max( 0.0f, Min( 1.0f, gamePhase ) );
            float   endgamePct  = Max( 0.0f, Min( 1.0f, gamePhase - 1 ) );
            float   midgamePct  = 1 - (openingPct + endgamePct);

            for( int i = 0; i < EVAL_TERMS; i++ )
                weights[i] = (EvalWeight) (((mWeightsOpening[i] * openingPct) + (mWeightsMidgame[i] * midgamePct) + (mWeightsEndgame[i] * endgamePct)) * WEIGHT_SCALE);
        }
    }


    template< int POPCNT, typename SIMD >
    PDECL SIMD Evaluate( const PositionT< SIMD >& pos, const MoveMapT< SIMD >& mmap, const EvalWeight* weights ) const
    {
        SIMD    eval[EVAL_TERMS];

        this->CalcEvalTerms< POPCNT, SIMD >( pos, mmap, eval );

        SIMD    score               = this->ApplyWeights( eval, weights );
        SIMD    moveTargets         = mmap.CalcMoveTargets();
        SIMD    inCheck             = mmap.IsInCheck();
        SIMD    mateFlavor          = SelectIfNotZero( inCheck, (SIMD) EVAL_CHECKMATE, (SIMD) EVAL_STALEMATE );
        SIMD    evalConsideringMate = SelectIfNotZero( moveTargets, score, mateFlavor );    

        return( evalConsideringMate );
    }


    template< typename SIMD >
    PDECL SIMD ApplyWeights( const SIMD* eval, const EvalWeight* weights ) const
    {
        SIMD score = MulLow32( eval[0], weights[0] );
        for( int i = 1; i < EVAL_TERMS; i++ )
            score += MulLow32( eval[i], weights[i] );

        return( score >> WEIGHT_SHIFT );
    }


    template< int POPCNT, typename SIMD >
    PDECL void CalcEvalTerms( const PositionT< SIMD >& pos, const MoveMapT< SIMD >& mmap, SIMD* eval ) const
    {
        PositionT< SIMD > flipped;
        flipped.FlipFrom( pos );

        SIMD    evalWhite[EVAL_TERMS];
        SIMD    evalBlack[EVAL_TERMS];

        this->CalcSideEval< POPCNT, SIMD >( pos,     mmap, evalWhite );
        this->CalcSideEval< POPCNT, SIMD >( flipped, mmap, evalBlack );

        for( int i = 0; i < EVAL_TERMS; i++ )
            eval[i] = evalWhite[i] - evalBlack[i];
    }


    template< int POPCNT, typename SIMD >
    PDECL void CalcSideEval( const PositionT< SIMD >& pos, const MoveMapT< SIMD >& mmap, SIMD* eval ) const
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

        eval[EVAL_PAWNS]            = CountBits< POPCNT >( whitePawns );                                
        eval[EVAL_CENTER_PAWNS]     = CountBits< POPCNT >( whitePawns   & CENTER_SQUARES );             
        eval[EVAL_CHAINED_PAWNS]    = CountBits< POPCNT >( pawnsChained );                              
        eval[EVAL_PASSED_PAWNS]     = CountBits< POPCNT >( PropN( whitePawns, ~blackPawns ) & RANK_8 ); 
        eval[EVAL_PAWNS_GUARD_KING] = CountBits< POPCNT >( whitePawns & (StepNW( whiteKing ) | StepN( whiteKing ) | StepNE( whiteKing )) );                                               
        eval[EVAL_PROMOTING_SOON]   = CountBits< POPCNT >( whitePawns & RANK_6 );                     
        eval[EVAL_PROMOTING_IMMED]  = CountBits< POPCNT >( whitePawns & RANK_7 );                     

        eval[EVAL_KNIGHTS]          = CountBits< POPCNT >( whiteKnights );                              
        eval[EVAL_KNIGHTS_DEVEL]    = CountBits< POPCNT >( whiteKnights & ~(SQUARE_B1 | SQUARE_G1) );                                                 
        eval[EVAL_KNIGHTS_FIRST]    = SubClampZero( eval[EVAL_KNIGHTS_DEVEL], eval[EVAL_BISHOPS_DEVEL] );                                                 
        eval[EVAL_KNIGHTS_INTERIOR] = CountBits< POPCNT >( whiteKnights & ~EDGE_SQUARES );              
        eval[EVAL_KNIGHTS_CENTRAL]  = CountBits< POPCNT >( whiteKnights & CENTER_SQUARES );              

        eval[EVAL_BISHOPS]          = CountBits< POPCNT >( whiteBishops );                              
        eval[EVAL_BISHOPS_DEVEL]    = CountBits< POPCNT >( whiteBishops & ~(SQUARE_C1 | SQUARE_F1) );    
        eval[EVAL_BISHOPS_INTERIOR] = CountBits< POPCNT >( whiteBishops & ~EDGE_SQUARES );              
        eval[EVAL_BISHOPS_CENTRAL]  = CountBits< POPCNT >( whiteBishops & CENTER_SQUARES );              
        eval[EVAL_BOTH_BISHOPS]     = SelectIfNotZero( whiteBishops & LIGHT_SQUARES, (SIMD) 1 ) & SelectIfNotZero( whiteBishops & DARK_SQUARES, (SIMD) 1 );                                                  

        eval[EVAL_ROOKS]            = CountBits< POPCNT >( whiteRooks );                                
        eval[EVAL_ROOKS_DEVEL]      = CountBits< POPCNT >( whiteRooks & ~(SQUARE_A1 | SQUARE_H1) );   
        eval[EVAL_ROOK_ON_RANK_7]   = CountBits< POPCNT >( whiteRooks & RANK_7 );                       
        eval[EVAL_ROOKS_CONNECTED]  = CountBits< POPCNT >( PropExOrtho( whiteRooks, empty ) & whiteRooks );                                               
        eval[EVAL_ROOKS_OPEN_FILE]  = CountBits< POPCNT >( PropN( whiteRooks, empty ) & RANK_8 );       

        eval[EVAL_QUEENS]           = CountBits< POPCNT >( whiteQueens );                               
        eval[EVAL_QUEEN_DEVEL]      = CountBits< POPCNT >( whiteQueens & ~(SQUARE_D1) );               
        eval[EVAL_QUEENS_INTERIOR]  = CountBits< POPCNT >( whiteQueens & ~EDGE_SQUARES );              
        eval[EVAL_QUEENS_CENTRAL]   = CountBits< POPCNT >( whiteQueens & CENTER_SQUARES );              

        eval[EVAL_KINGS]            = CountBits< POPCNT >( whiteKing );                                 
        eval[EVAL_KING_CASTLED]     = CountBits< POPCNT >( whiteKing & RANK_1 & ~SQUARE_E1 );           

        eval[EVAL_MOBILITY]         = CountBits< POPCNT >( whiteMobility );                             
        eval[EVAL_ATTACKING]        = CountBits< POPCNT >( whiteAttacking );                            
        eval[EVAL_DEFENDING]        = CountBits< POPCNT >( whiteDefending );                            
        eval[EVAL_ENEMY_TERRITORY]  = CountBits< POPCNT >( inEnemyTerritory );                          
        eval[EVAL_CENTER_PIECES]    = CountBits< POPCNT >( whitePieces  & CENTER_SQUARES );             
        eval[EVAL_CENTER_CONTROL]   = CountBits< POPCNT >( whiteControl & CENTER_SQUARES );             
    }
};

#endif // PIGEON_EVAL_H__
};
