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
    bool        mEnableOpening;

public:
    Evaluator()
    {
        PlatClearMemory( mWeightsOpening, sizeof( mWeightsOpening ) );
        PlatClearMemory( mWeightsMidgame, sizeof( mWeightsMidgame ) );
        PlatClearMemory( mWeightsEndgame, sizeof( mWeightsEndgame ) );

        mEnableOpening  = true;

        assert( (CENTER_DIST_0 | CENTER_DIST_1 | CENTER_DIST_2 | CENTER_DIST_3 | CENTER_DIST_4 | CENTER_DIST_5 | CENTER_DIST_6) == ALL_SQUARES );
        assert( (CENTER_RING_0 | CENTER_RING_1 | CENTER_RING_2 | CENTER_RING_3) == ALL_SQUARES );

        this->SetWeight( EVAL_PAWN,                100.00000f,   100.00000f,   100.00000f );  
        this->SetWeight( EVAL_PAWN_CHAINED,          0.43266f,     0.14918f,     2.24521f );  
        this->SetWeight( EVAL_PAWN_PASSED,           3.78850f,     0.01971f,     0.03716f );  
        this->SetWeight( EVAL_PAWN_CLEAR,           30.58570f,     1.60770f,     1.84895f );  
        this->SetWeight( EVAL_PAWN_GUARD_KING,      10.34970f,     4.51141f,     8.51882f );  
        this->SetWeight( EVAL_PAWN_RANK_4,          23.39273f,     1.58597f,     0.00939f );  
        this->SetWeight( EVAL_PAWN_RANK_5,           6.35503f,     8.21703f,     8.83197f );  
        this->SetWeight( EVAL_PAWN_RANK_6,           0.98910f,    45.78074f,    44.54308f );  
        this->SetWeight( EVAL_PAWN_RANK_7,          69.08373f,    90.58780f,    82.52876f );  
        this->SetWeight( EVAL_KNIGHT_DEVEL,         14.51974f,     0.00000f,     0.00000f );  
        this->SetWeight( EVAL_KNIGHT_EDGE,         250.63967f,   248.32707f,   262.98608f );  
        this->SetWeight( EVAL_KNIGHT_CORNER,       270.38517f,   165.01715f,   201.47506f );  
        this->SetWeight( EVAL_KNIGHT_BACK_RANK,      1.03804f,    20.51836f,    15.11453f );  
        this->SetWeight( EVAL_KNIGHT_RING_0,       226.77008f,   276.40084f,   308.13009f );  
        this->SetWeight( EVAL_KNIGHT_RING_1,       253.39178f,   275.03958f,   295.75897f );  
        this->SetWeight( EVAL_KNIGHT_RING_2,       246.52185f,   257.37457f,   275.84324f );  
        this->SetWeight( EVAL_BISHOP_DEVEL,          8.91278f,     0.00000f,     0.00000f );  
        this->SetWeight( EVAL_BISHOP_PAIR,           5.62908f,    38.73975f,    36.48099f );  
        this->SetWeight( EVAL_BISHOP_EDGE,         272.70089f,   272.81248f,   283.37436f );  
        this->SetWeight( EVAL_BISHOP_CORNER,       238.47599f,   263.00823f,   271.29253f );  
        this->SetWeight( EVAL_BISHOP_RING_0,       217.41415f,   277.96289f,   294.04214f );  
        this->SetWeight( EVAL_BISHOP_RING_1,       257.83660f,   281.61956f,   299.62788f );  
        this->SetWeight( EVAL_BISHOP_RING_2,       271.12501f,   273.75466f,   291.77377f );  
        this->SetWeight( EVAL_ROOK_DEVEL,           12.49096f,     0.00000f,     0.00000f );  
        this->SetWeight( EVAL_ROOK_CONNECTED,        1.00000f,     1.00000f,     1.00000f );  
        this->SetWeight( EVAL_ROOK_OPEN_FILE,        2.95221f,    28.51943f,    29.97842f );  
        this->SetWeight( EVAL_ROOK_RING_0,         721.16189f,   422.53224f,   436.92308f );  
        this->SetWeight( EVAL_ROOK_RING_1,         321.85842f,   429.82509f,   445.87453f );  
        this->SetWeight( EVAL_ROOK_RING_2,         380.50480f,   433.02880f,   449.08436f );  
        this->SetWeight( EVAL_ROOK_RING_3,         502.39359f,   445.52623f,   460.86179f );  
        this->SetWeight( EVAL_ROOK_BACK_RANK,        4.40942f,    23.79803f,    26.06871f );  
        this->SetWeight( EVAL_QUEEN,               851.78743f,   899.75277f,   932.38865f );  
        this->SetWeight( EVAL_QUEEN_DEVEL,          10.66272f,     0.00000f,     0.00000f );  
        this->SetWeight( EVAL_KING,              20000.00000f, 20000.00000f, 20000.00000f );  
        this->SetWeight( EVAL_KING_CASTLED,          4.13342f,     9.07229f,     2.52904f );  
        this->SetWeight( EVAL_KING_EDGE,            53.47085f,    16.78878f,    48.47494f );  
        this->SetWeight( EVAL_KING_CORNER,           1.73069f,     7.68011f,    39.87398f );  
        this->SetWeight( EVAL_KING_RING_0,          61.22929f,    21.64032f,    70.32323f );  
        this->SetWeight( EVAL_KING_RING_1,           9.26425f,    27.93207f,    63.21985f );  
        this->SetWeight( EVAL_KING_RING_2,          11.74106f,    21.27715f,    50.78455f );  
        this->SetWeight( EVAL_MOBILITY,              0.17624f,     0.26167f,     0.02061f );  
        this->SetWeight( EVAL_ATTACKING,             0.05187f,     6.05545f,    10.79501f );  
        this->SetWeight( EVAL_DEFENDING,             0.00618f,     0.35709f,     0.17124f );  
        this->SetWeight( EVAL_CONTROL_RANK_5,        0.69690f,     4.13653f,     1.31370f );  
        this->SetWeight( EVAL_CONTROL_RANK_6,        0.21382f,     3.59474f,     5.42215f );  
        this->SetWeight( EVAL_CONTROL_RANK_7,        0.02038f,     9.02705f,    10.99075f );  
        this->SetWeight( EVAL_CONTROL_RANK_8,        0.06104f,    11.76331f,    12.37675f );  
        this->SetWeight( EVAL_CONTROL_FILE_AH,       2.11963f,     1.98983f,     2.82921f );  
        this->SetWeight( EVAL_CONTROL_FILE_BG,       0.22629f,     0.03941f,     0.00875f );  
        this->SetWeight( EVAL_CONTROL_FILE_CF,       6.90315f,     0.11079f,     0.10983f );  
        this->SetWeight( EVAL_CONTROL_FILE_DE,       0.29918f,     0.01624f,     0.00138f );  
        this->SetWeight( EVAL_KNIGHT_FIRST,          0.20367f,     0.00000f,     0.00000f );  

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
        float   openingPct  = 1 - Max( 0.0f, Min( 1.0f, gamePhase ) );
        float   endgamePct  = Max( 0.0f, Min( 1.0f, gamePhase - 1 ) );
        float   midgamePct  = 1 - (openingPct + endgamePct);

        for( int i = 0; i < EVAL_TERMS; i++ )
            weights[i] = (EvalWeight) (((mWeightsOpening[i] * openingPct) + (mWeightsMidgame[i] * midgamePct) + (mWeightsEndgame[i] * endgamePct)) * WEIGHT_SCALE);
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

    PDECL void CalcMaterialTable( const Position& pos, MaterialTable* mat ) const
    {
        float gamePhase = this->CalcGamePhase< 0 >( pos );

        for( int piece = 0; piece < PIECE_TYPES; piece++ )
        {
            i32* dest = mat->mValue[piece];

            
        }

        mat->CalcCastlingFixup();
    }

};

#endif // PIGEON_EVAL_H__
};
