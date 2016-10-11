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

        assert( (CENTER_DIST_0 | CENTER_DIST_1 | CENTER_DIST_2 | CENTER_DIST_3 | CENTER_DIST_4 | CENTER_DIST_5 | CENTER_DIST_6) == ALL_SQUARES );
        assert( (CENTER_RING_0 | CENTER_RING_1 | CENTER_RING_2 | CENTER_RING_3) == ALL_SQUARES );

        //this->SetWeight( EVAL_PAWN,                100.00000,   100.00000,   100.00000 );  
        this->SetWeight( EVAL_PAWN_CHAINED,          0.43266,     0.14918,     2.24521 );  
        this->SetWeight( EVAL_PAWN_PASSED,           3.78850,     0.01971,     0.03716 );  
        this->SetWeight( EVAL_PAWN_CLEAR,           30.58570,     1.60770,     1.84895 );  
        this->SetWeight( EVAL_PAWN_GUARD_KING,      10.34970,     4.51141,     8.51882 );  
        //this->SetWeight( EVAL_PAWN_RANK_4,          23.39273,     1.58597,     0.00939 );  
        //this->SetWeight( EVAL_PAWN_RANK_5,           6.35503,     8.21703,     8.83197 );  
        //this->SetWeight( EVAL_PAWN_RANK_6,           0.98910,    45.78074,    44.54308 );  
        //this->SetWeight( EVAL_PAWN_RANK_7,          69.08373,    90.58780,    82.52876 );  
        //this->SetWeight( EVAL_KNIGHT_DEVEL,         14.51974,     0.00000,     0.00000 );  
        //this->SetWeight( EVAL_KNIGHT_EDGE,         250.63967,   248.32707,   262.98608 );  
        //this->SetWeight( EVAL_KNIGHT_CORNER,       270.38517,   165.01715,   201.47506 );  
        //this->SetWeight( EVAL_KNIGHT_BACK_RANK,      1.03804,    20.51836,    15.11453 );  
        //this->SetWeight( EVAL_KNIGHT_RING_0,       226.77008,   276.40084,   308.13009 );  
        //this->SetWeight( EVAL_KNIGHT_RING_1,       253.39178,   275.03958,   295.75897 );  
        //this->SetWeight( EVAL_KNIGHT_RING_2,       246.52185,   257.37457,   275.84324 );  
        //this->SetWeight( EVAL_BISHOP_DEVEL,          8.91278,     0.00000,     0.00000 );  
        this->SetWeight( EVAL_BISHOP_PAIR,           5.62908,    38.73975,    36.48099 );  
        //this->SetWeight( EVAL_BISHOP_EDGE,         272.70089,   272.81248,   283.37436 );  
        //this->SetWeight( EVAL_BISHOP_CORNER,       238.47599,   263.00823,   271.29253 );  
        //this->SetWeight( EVAL_BISHOP_RING_0,       217.41415,   277.96289,   294.04214 );  
        //this->SetWeight( EVAL_BISHOP_RING_1,       257.83660,   281.61956,   299.62788 );  
        //this->SetWeight( EVAL_BISHOP_RING_2,       271.12501,   273.75466,   291.77377 );  
        //this->SetWeight( EVAL_ROOK_DEVEL,           12.49096,     0.00000,     0.00000 );  
        this->SetWeight( EVAL_ROOK_CONNECTED,        1.00000,     1.00000,     1.00000 );  
        this->SetWeight( EVAL_ROOK_OPEN_FILE,        2.95221,    28.51943,    29.97842 );  
        //this->SetWeight( EVAL_ROOK_RING_0,         721.16189,   422.53224,   436.92308 );  
        //this->SetWeight( EVAL_ROOK_RING_1,         321.85842,   429.82509,   445.87453 );  
        //this->SetWeight( EVAL_ROOK_RING_2,         380.50480,   433.02880,   449.08436 );  
        //this->SetWeight( EVAL_ROOK_RING_3,         502.39359,   445.52623,   460.86179 );  
        //this->SetWeight( EVAL_ROOK_BACK_RANK,        4.40942,    23.79803,    26.06871 );  
        //this->SetWeight( EVAL_QUEEN,               851.78743,   899.75277,   932.38865 );  
        //this->SetWeight( EVAL_QUEEN_DEVEL,          10.66272,     0.00000,     0.00000 );  
        //this->SetWeight( EVAL_KING,              20000.00000, 20000.00000, 20000.00000 );  
        this->SetWeight( EVAL_KING_CASTLED,          4.13342,     9.07229,     2.52904 );  
        //this->SetWeight( EVAL_KING_EDGE,            53.47085,    16.78878,    48.47494 );  
        //this->SetWeight( EVAL_KING_CORNER,           1.73069,     7.68011,    39.87398 );  
        //this->SetWeight( EVAL_KING_RING_0,          61.22929,    21.64032,    70.32323 );  
        //this->SetWeight( EVAL_KING_RING_1,           9.26425,    27.93207,    63.21985 );  
        //this->SetWeight( EVAL_KING_RING_2,          11.74106,    21.27715,    50.78455 );  
        this->SetWeight( EVAL_MOBILITY,              0.17624,     0.26167,     0.02061 );  
        this->SetWeight( EVAL_ATTACKING,             0.05187,     6.05545,    10.79501 );  
        this->SetWeight( EVAL_DEFENDING,             0.00618,     0.35709,     0.17124 );  
        this->SetWeight( EVAL_CONTROL_RANK_5,        0.69690,     4.13653,     1.31370 );  
        this->SetWeight( EVAL_CONTROL_RANK_6,        0.21382,     3.59474,     5.42215 );  
        this->SetWeight( EVAL_CONTROL_RANK_7,        0.02038,     9.02705,    10.99075 );  
        this->SetWeight( EVAL_CONTROL_RANK_8,        0.06104,    11.76331,    12.37675 );  
        this->SetWeight( EVAL_CONTROL_FILE_AH,       2.11963,     1.98983,     2.82921 );  
        this->SetWeight( EVAL_CONTROL_FILE_BG,       0.22629,     0.03941,     0.00875 );  
        this->SetWeight( EVAL_CONTROL_FILE_CF,       6.90315,     0.11079,     0.10983 );  
        this->SetWeight( EVAL_CONTROL_FILE_DE,       0.29918,     0.01624,     0.00138 );  
        this->SetWeight( EVAL_KNIGHT_FIRST,          0.20367,     0.00000,     0.00000 );  

    }

    void EnableOpening( bool enable )
    {
        mEnableOpening = enable;
    }

    static const char* GetWeightName( int idx ) 
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

    static int GetWeightIdx( const char* name )  
    {
        for( int idx = 0; idx < EVAL_TERMS; idx++ )
        {
            const char* weightName = Evaluator::GetWeightName( idx );

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
        //return( endingness );
    }




    template< int POPCNT, typename SIMD >
    PDECL SIMD Evaluate( const PositionT< SIMD >& pos, const MoveMapT< SIMD >& mmap, const EvalWeight* weights ) const
    {
        SIMD    eval[EVAL_TERMS];   

        this->CalcEvalTerms< POPCNT, SIMD >( pos, mmap, eval );

        SIMD    evalScore           = this->ApplyWeights( eval, weights );
        SIMD    materialScore       = (pos.mWhiteMaterial - pos.mBlackMaterial) >> WEIGHT_SHIFT;
        SIMD    score               = (evalScore + pos.mWhiteMaterial - pos.mBlackMaterial) >> WEIGHT_SHIFT;//materialScore;
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

        return( score );//>> WEIGHT_SHIFT );
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

        eval[EVAL_PAWN]             = CountBits< POPCNT >( whitePawns );                                
        eval[EVAL_PAWN_CHAINED]     = CountBits< POPCNT >( pawnsChained );                              
        eval[EVAL_PAWN_PASSED]      = CountBits< POPCNT >( PropN( whitePawns, ~blackPawns ) & RANK_8 ); 
        eval[EVAL_PAWN_CLEAR]       = CountBits< POPCNT >( PropN( whitePawns, empty ) & RANK_8 ); 
        eval[EVAL_PAWN_GUARD_KING]  = CountBits< POPCNT >( whitePawns & (StepNW( whiteKing ) | StepN( whiteKing ) | StepNE( whiteKing )) );
        eval[EVAL_PAWN_RANK_4]      = CountBits< POPCNT >( whitePawns & RANK_4 );
        eval[EVAL_PAWN_RANK_5]      = CountBits< POPCNT >( whitePawns & RANK_5 );
        eval[EVAL_PAWN_RANK_6]      = CountBits< POPCNT >( whitePawns & RANK_6 );
        eval[EVAL_PAWN_RANK_7]      = CountBits< POPCNT >( whitePawns & RANK_7 );
        eval[EVAL_KNIGHT_DEVEL]     = CountBits< POPCNT >( whiteKnights & ~(SQUARE_B1 | SQUARE_G1) );    
        eval[EVAL_KNIGHT_EDGE]      = CountBits< POPCNT >( whiteKnights & EDGE_SQUARES );
        eval[EVAL_KNIGHT_CORNER]    = CountBits< POPCNT >( whiteKnights & CORNER_SQUARES );
        eval[EVAL_KNIGHT_BACK_RANK] = CountBits< POPCNT >( whiteKnights & (RANK_5 | RANK_6 | RANK_7 | RANK_7) );
        eval[EVAL_KNIGHT_RING_0]    = CountBits< POPCNT >( whiteKnights & CENTER_RING_0 );
        eval[EVAL_KNIGHT_RING_1]    = CountBits< POPCNT >( whiteKnights & CENTER_RING_1 );
        eval[EVAL_KNIGHT_RING_2]    = CountBits< POPCNT >( whiteKnights & CENTER_RING_2 );
        eval[EVAL_BISHOP_DEVEL]     = CountBits< POPCNT >( whiteBishops & ~(SQUARE_C1 | SQUARE_F1) ); 
        eval[EVAL_BISHOP_PAIR]      = SelectIfNotZero( whiteBishops & LIGHT_SQUARES, (SIMD) 1 ) & SelectIfNotZero( whiteBishops & DARK_SQUARES, (SIMD) 1 ); 
        eval[EVAL_BISHOP_EDGE]      = CountBits< POPCNT >( whiteBishops & EDGE_SQUARES );
        eval[EVAL_BISHOP_CORNER]    = CountBits< POPCNT >( whiteBishops & CORNER_SQUARES );
        eval[EVAL_BISHOP_RING_0]    = CountBits< POPCNT >( whiteBishops & CENTER_RING_0 );
        eval[EVAL_BISHOP_RING_1]    = CountBits< POPCNT >( whiteBishops & CENTER_RING_1 );
        eval[EVAL_BISHOP_RING_2]    = CountBits< POPCNT >( whiteBishops & CENTER_RING_2 );
        eval[EVAL_ROOK_DEVEL]       = CountBits< POPCNT >( whiteRooks & ~(SQUARE_A1 | SQUARE_H1) );   
        eval[EVAL_ROOK_CONNECTED]   = CountBits< POPCNT >( PropExOrtho( whiteRooks, empty ) & whiteRooks ); 
        eval[EVAL_ROOK_OPEN_FILE]   = CountBits< POPCNT >( PropN( whiteRooks, empty ) & RANK_8 );       
        eval[EVAL_ROOK_RING_0]      = CountBits< POPCNT >( whiteRooks & CENTER_RING_0 );
        eval[EVAL_ROOK_RING_1]      = CountBits< POPCNT >( whiteRooks & CENTER_RING_1 );
        eval[EVAL_ROOK_RING_2]      = CountBits< POPCNT >( whiteRooks & CENTER_RING_2 );
        eval[EVAL_ROOK_RING_3]      = CountBits< POPCNT >( whiteRooks & CENTER_RING_3 );
        eval[EVAL_ROOK_BACK_RANK]   = CountBits< POPCNT >( whiteRooks & (RANK_5 | RANK_6 | RANK_7 | RANK_7) );
        eval[EVAL_QUEEN]            = CountBits< POPCNT >( whiteQueens );
        eval[EVAL_QUEEN_DEVEL]      = CountBits< POPCNT >( whiteQueens & ~(SQUARE_D1) );
        eval[EVAL_KING]             = CountBits< POPCNT >( whiteKing );                      
        eval[EVAL_KING_CASTLED]     = CountBits< POPCNT >( whiteKing & RANK_1 & ~SQUARE_E1 );
        eval[EVAL_KING_EDGE]        = CountBits< POPCNT >( whiteKing & EDGE_SQUARES );
        eval[EVAL_KING_CORNER]      = CountBits< POPCNT >( whiteKing & CORNER_SQUARES );
        eval[EVAL_KING_RING_0]      = CountBits< POPCNT >( whiteKing & CENTER_RING_0 );
        eval[EVAL_KING_RING_1]      = CountBits< POPCNT >( whiteKing & CENTER_RING_1 );
        eval[EVAL_KING_RING_2]      = CountBits< POPCNT >( whiteKing & CENTER_RING_2 );
        eval[EVAL_MOBILITY]         = CountBits< POPCNT >( whiteMobility ); 
        eval[EVAL_ATTACKING]        = CountBits< POPCNT >( whiteAttacking );
        eval[EVAL_DEFENDING]        = CountBits< POPCNT >( whiteDefending );
        eval[EVAL_CONTROL_RANK_5]   = CountBits< POPCNT >( whiteControl & RANK_5 );
        eval[EVAL_CONTROL_RANK_6]   = CountBits< POPCNT >( whiteControl & RANK_6 );
        eval[EVAL_CONTROL_RANK_7]   = CountBits< POPCNT >( whiteControl & RANK_7 );
        eval[EVAL_CONTROL_RANK_8]   = CountBits< POPCNT >( whiteControl & RANK_8 );
        eval[EVAL_CONTROL_FILE_AH]  = CountBits< POPCNT >( whiteControl & (FILE_A | FILE_H) );
        eval[EVAL_CONTROL_FILE_BG]  = CountBits< POPCNT >( whiteControl & (FILE_B | FILE_G) );
        eval[EVAL_CONTROL_FILE_CF]  = CountBits< POPCNT >( whiteControl & (FILE_C | FILE_F) );
        eval[EVAL_CONTROL_FILE_DE]  = CountBits< POPCNT >( whiteControl & (FILE_D | FILE_E) );
        eval[EVAL_KNIGHT_FIRST]     = SubClampZero( eval[EVAL_KNIGHT_DEVEL], eval[EVAL_BISHOP_DEVEL] );
    }

};

#endif // PIGEON_EVAL_H__
};
