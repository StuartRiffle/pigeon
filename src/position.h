// position.h - PIGEON CHESS ENGINE (c) 2012-2017 Stuart Riffle

namespace Pigeon {
#ifndef PIGEON_POSITION_H__
#define PIGEON_POSITION_H__


/// A map of valid move target squares

template< typename SIMD >
struct PIGEON_ALIGN_SIMD MoveMapT
{
    SIMD        mSlidingMovesNW;
    SIMD        mSlidingMovesNE;
    SIMD        mSlidingMovesSW;
    SIMD        mSlidingMovesSE;
    SIMD        mSlidingMovesN;
    SIMD        mSlidingMovesW;
    SIMD        mSlidingMovesE;
    SIMD        mSlidingMovesS;

    SIMD        mKnightMovesNNW;
    SIMD        mKnightMovesNNE;
    SIMD        mKnightMovesWNW;
    SIMD        mKnightMovesENE;
    SIMD        mKnightMovesWSW;
    SIMD        mKnightMovesESE;
    SIMD        mKnightMovesSSW;
    SIMD        mKnightMovesSSE;

    SIMD        mPawnMovesN;
    SIMD        mPawnDoublesN;
    SIMD        mPawnAttacksNE;
    SIMD        mPawnAttacksNW;

    SIMD        mCastlingMoves;
    SIMD        mKingMoves;
    SIMD        mBlackControl;
    SIMD        mCheckMask;

    INLINE PDECL SIMD CalcMoveTargets() const
    {
        SIMD slidingMoves   = mSlidingMovesNW | mSlidingMovesNE | mSlidingMovesSW | mSlidingMovesSE | mSlidingMovesN  | mSlidingMovesW  | mSlidingMovesE  | mSlidingMovesS;
        SIMD knightMoves    = mKnightMovesNNW | mKnightMovesNNE | mKnightMovesWNW | mKnightMovesENE | mKnightMovesWSW | mKnightMovesESE | mKnightMovesSSW | mKnightMovesSSE;
        SIMD otherMoves     = mPawnMovesN | mPawnDoublesN | mPawnAttacksNE | mPawnAttacksNW | mCastlingMoves | mKingMoves;
        SIMD targets        = (slidingMoves & mCheckMask) | knightMoves | otherMoves;

        return( targets );
    }

    INLINE PDECL SIMD IsInCheck() const
    {
        return( ~CmpEqual( mCheckMask, MaskAllSet< SIMD >() ) );
    }
};
                

/// A piece-square table for incremental update of material values

struct MaterialTable
{
    i32     mValue[6][64];              /// Piece-square value in centipawns, as 16.16 fixed point
    i64     mCastlingQueenside;         /// Account for the rook when castling queenside
    i64     mCastlingKingside;          /// Account for the rook when castling kingside

    void CalcCastlingFixup()
    {
        mCastlingQueenside = mValue[ROOK][D1] - mValue[ROOK][A1];
        mCastlingKingside  = mValue[ROOK][F1] - mValue[ROOK][H1];
    }
};


/// A snapshot of the game state

template< typename SIMD >
struct PIGEON_ALIGN_SIMD PositionT
{
    SIMD        mWhitePawns;        /// Bitmask of the white pawns 
    SIMD        mWhiteKnights;      /// Bitmask of the white knights    
    SIMD        mWhiteBishops;      /// Bitmask of the white bishops    
    SIMD        mWhiteRooks;        /// Bitmask of the white rooks 
    SIMD        mWhiteQueens;       /// Bitmask of the white queens    
    SIMD        mWhiteKing;         /// Bitmask of the white king 
                                      
    SIMD        mBlackPawns;        /// Bitmask of the black pawns     
    SIMD        mBlackKnights;      /// Bitmask of the black knights
    SIMD        mBlackBishops;      /// Bitmask of the black bishops    
    SIMD        mBlackRooks;        /// Bitmask of the black rooks     
    SIMD        mBlackQueens;       /// Bitmask of the black queens    
    SIMD        mBlackKing;         /// Bitmask of the black king 
                                     
    SIMD        mCastlingAndEP;     /// Bitmask of EP capture targets and castling targets
    SIMD        mHash;              /// Board position hash calculated from all the fields preceding this one     
    SIMD        mBoardFlipped;      /// A mask which is ~0 when this structure is white/black flipped from the actual position
    SIMD        mWhiteToMove;       /// 1 if it's white to play, 0 if black
    SIMD        mHalfmoveClock;     /// Number of halfmoves since the last capture or pawn move
    SIMD        mFullmoveNum;       /// Starts at 1, increments after black moves
    SIMD        mWhiteMaterial;     /// Total white material, updated incrementally from piece-square tables
    SIMD        mBlackMaterial;     /// Total black material, updated incrementally from piece-square tables


    /// Reset the fields to the start-of-game position.
    
    PDECL void Reset()
    {
        mWhitePawns                 = RANK_2;
        mWhiteKnights               = SQUARE_B1 | SQUARE_G1;
        mWhiteBishops               = SQUARE_C1 | SQUARE_F1;
        mWhiteRooks                 = SQUARE_A1 | SQUARE_H1;
        mWhiteQueens                = SQUARE_D1;        
        mWhiteKing                  = SQUARE_E1;

        mBlackPawns                 = RANK_7;              
        mBlackKnights               = SQUARE_B8 | SQUARE_G8;
        mBlackBishops               = SQUARE_C8 | SQUARE_F8;
        mBlackRooks                 = SQUARE_A8 | SQUARE_H8;
        mBlackQueens                = SQUARE_D8;
        mBlackKing                  = SQUARE_E8;

        mCastlingAndEP              = SQUARE_A1 | SQUARE_H1 | SQUARE_A8 | SQUARE_H8;
        mHash                       = this->CalcHash();
        mBoardFlipped               = 0;
        mWhiteToMove                = 1;
        mHalfmoveClock              = 0;
        mFullmoveNum                = 1;
        mWhiteMaterial              = 0;
        mBlackMaterial              = 0;
    }


    //--------------------------------------------------------------------------
    /// Duplicate member values across SIMD lanes.
    ///
    /// \param  src     Scalar instance to broadcast
    
    template< typename SCALAR >
    PDECL void Broadcast( const PositionT< SCALAR >& src )
    {
        mWhitePawns                 = src.mWhitePawns;   
        mWhiteKnights               = src.mWhiteKnights; 
        mWhiteBishops               = src.mWhiteBishops; 
        mWhiteRooks                 = src.mWhiteRooks;   
        mWhiteQueens                = src.mWhiteQueens;  
        mWhiteKing                  = src.mWhiteKing;

        mBlackPawns                 = src.mBlackPawns;   
        mBlackKnights               = src.mBlackKnights; 
        mBlackBishops               = src.mBlackBishops; 
        mBlackRooks                 = src.mBlackRooks;   
        mBlackQueens                = src.mBlackQueens;  
        mBlackKing                  = src.mBlackKing;

        mCastlingAndEP              = src.mCastlingAndEP;
        mHash                       = src.mHash;         
        mBoardFlipped               = src.mBoardFlipped; 
        mWhiteToMove                = src.mWhiteToMove;  
        mHalfmoveClock              = src.mHalfmoveClock;
        mFullmoveNum                = src.mFullmoveNum;  
        mWhiteMaterial              = src.mWhiteMaterial;
        mBlackMaterial              = src.mBlackMaterial;
    }


    /// Update the game state by applying a valid move.
    
    PDECL void Step( 
        const MoveSpecT< SIMD >&    move,              /// Move to apply
        const MaterialTable*        matWhite = NULL,   /// Material value table for white (optional)
        const MaterialTable*        matBlack = NULL )  /// Material value table for black (optional)
    {
        SIMD    moveSrc             = SelectWithMask( mBoardFlipped,  FlipSquareIndex( move.mSrc ), move.mSrc );
        SIMD    srcBit              = SquareBit( moveSrc );
        SIMD    isPawnMove          = SelectIfNotZero( srcBit & mWhitePawns, MaskAllSet< SIMD >() );
        SIMD    isCapture           = CmpEqual( move.mType, (SIMD) CAPTURE_LOSING )
                                    | CmpEqual( move.mType, (SIMD) CAPTURE_EQUAL )
                                    | CmpEqual( move.mType, (SIMD) CAPTURE_WINNING )
                                    | CmpEqual( move.mType, (SIMD) CAPTURE_PROMOTE_KNIGHT )
                                    | CmpEqual( move.mType, (SIMD) CAPTURE_PROMOTE_BISHOP )
                                    | CmpEqual( move.mType, (SIMD) CAPTURE_PROMOTE_ROOK )
                                    | CmpEqual( move.mType, (SIMD) CAPTURE_PROMOTE_QUEEN );

        this->ApplyMove( move.mSrc, move.mDest, move.mType, matWhite, matBlack );
        this->FlipInPlace();

        mWhiteToMove               ^= 1;
        mFullmoveNum               += mWhiteToMove;
        mHalfmoveClock              = (mHalfmoveClock + 1) & ~(isPawnMove | isCapture);
    }


    PDECL static INLINE SIMD CalcPieceType( const SIMD& knights, const SIMD& bishops, const SIMD& rooks, const SIMD& queens, const SIMD& king )
    {
        SIMD result =
            SelectIfNotZero( knights,   (SIMD) KNIGHT ) |
            SelectIfNotZero( bishops,   (SIMD) BISHOP ) |
            SelectIfNotZero( rooks,     (SIMD) ROOK   ) |
            SelectIfNotZero( queens,    (SIMD) QUEEN  ) |
            SelectIfNotZero( king,      (SIMD) KING   );

        return( result );
    }


    /// Update the piece positions and white/black material values.
      
    PDECL void ApplyMove( 
        const SIMD&                 srcIdx,     /// Source square index
        const SIMD&                 destIdx,    /// Destination square index
        const SIMD&                 moveType,   /// Move type (includes extra info required for promotion)
        const MaterialTable*        matWhite,   /// Material value table for white (optional)
        const MaterialTable*        matBlack )  /// Material value table for black (optional)
    {
        SIMD    whitePawns          = mWhitePawns;    
        SIMD    whiteKnights        = mWhiteKnights;  
        SIMD    whiteBishops        = mWhiteBishops;  
        SIMD    whiteRooks          = mWhiteRooks;    
        SIMD    whiteQueens         = mWhiteQueens;   
        SIMD    whiteKing           = mWhiteKing;     
        SIMD    blackPawns          = mBlackPawns;    
        SIMD    blackKnights        = mBlackKnights;  
        SIMD    blackBishops        = mBlackBishops;  
        SIMD    blackRooks          = mBlackRooks;    
        SIMD    blackQueens         = mBlackQueens;   
        SIMD    blackKing           = mBlackKing;
        SIMD    castlingAndEP       = mCastlingAndEP;

        SIMD    moveSrc             = SelectWithMask( mBoardFlipped, FlipSquareIndex( srcIdx ),  srcIdx  );
        SIMD    moveDest            = SelectWithMask( mBoardFlipped, FlipSquareIndex( destIdx ), destIdx );
        SIMD    srcBit              = SquareBit( moveSrc );
        SIMD    destBit             = SquareBit( moveDest );

        SIMD    srcPawn             = srcBit & whitePawns;
        SIMD    srcKnight           = srcBit & whiteKnights;
        SIMD    srcBishop           = srcBit & whiteBishops;
        SIMD    srcRook             = srcBit & whiteRooks;
        SIMD    srcQueen            = srcBit & whiteQueens;
        SIMD    srcKing             = srcBit & whiteKing;

        SIMD    promotedKnight      = destBit & (CmpEqual( moveType, (SIMD) PROMOTE_KNIGHT ) | CmpEqual( moveType, (SIMD) CAPTURE_PROMOTE_KNIGHT ));
        SIMD    promotedBishop      = destBit & (CmpEqual( moveType, (SIMD) PROMOTE_BISHOP ) | CmpEqual( moveType, (SIMD) CAPTURE_PROMOTE_BISHOP ));
        SIMD    promotedRook        = destBit & (CmpEqual( moveType, (SIMD) PROMOTE_ROOK   ) | CmpEqual( moveType, (SIMD) CAPTURE_PROMOTE_ROOK   ));
        SIMD    promotedQueen       = destBit & (CmpEqual( moveType, (SIMD) PROMOTE_QUEEN  ) | CmpEqual( moveType, (SIMD) CAPTURE_PROMOTE_QUEEN  ));

        SIMD    formerPawn          = SelectIfNotZero( promotedKnight | promotedBishop | promotedRook | promotedQueen, srcPawn );
        SIMD    destPawn            = SelectIfNotZero( MaskOut( srcPawn, formerPawn ), destBit );
        SIMD    destKnight          = SelectIfNotZero( srcKnight | promotedKnight,     destBit );
        SIMD    destBishop          = SelectIfNotZero( srcBishop | promotedBishop,     destBit );
        SIMD    destRook            = SelectIfNotZero( srcRook   | promotedRook,       destBit );
        SIMD    destQueen           = SelectIfNotZero( srcQueen  | promotedQueen,      destBit );
        SIMD    destKing            = SelectIfNotZero( srcKing,                        destBit );

        SIMD    epTargetNext        = Shift< SHIFT_S >( Shift< SHIFT_N * 2 >( srcPawn ) & destPawn );
        SIMD    epVictimNow         = blackPawns & Shift< SHIFT_S >( destPawn & castlingAndEP & EP_SQUARES );
        SIMD    castleRookKing      = CmpEqual( (srcKing | destBit), (SIMD) (SQUARE_E1 | SQUARE_G1) ) & SQUARE_H1;
        SIMD    castleRookQueen     = CmpEqual( (srcKing | destBit), (SIMD) (SQUARE_E1 | SQUARE_C1) ) & SQUARE_A1;
        SIMD    disableCastleBit    = 0;                 

        srcRook                    |= castleRookKing | castleRookQueen;
        destRook                   |= SelectIfNotZero( castleRookKing,  (SIMD) SQUARE_F1 );
        destRook                   |= SelectIfNotZero( castleRookQueen, (SIMD) SQUARE_D1 );
        disableCastleBit           |= (srcRook & (SQUARE_A1 | SQUARE_H1));
        disableCastleBit           |= SelectIfNotZero( srcKing, (SIMD) (SQUARE_A1 | SQUARE_H1) );

        mWhitePawns                 = MaskOut( whitePawns,   srcPawn )   | destPawn;
        mWhiteKnights               = MaskOut( whiteKnights, srcKnight ) | destKnight; 
        mWhiteBishops               = MaskOut( whiteBishops, srcBishop ) | destBishop; 
        mWhiteRooks                 = MaskOut( whiteRooks,   srcRook )   | destRook; 
        mWhiteQueens                = MaskOut( whiteQueens,  srcQueen )  | destQueen; 
        mWhiteKing                  = MaskOut( whiteKing,    srcKing )   | destKing; 
        mBlackPawns                 = MaskOut( MaskOut( blackPawns, destBit ), epVictimNow );
        mBlackKnights               = MaskOut( blackKnights, destBit );
        mBlackBishops               = MaskOut( blackBishops, destBit );
        mBlackRooks                 = MaskOut( blackRooks,   destBit );
        mBlackQueens                = MaskOut( blackQueens,  destBit );
        mCastlingAndEP              = MaskOut( castlingAndEP, disableCastleBit | EP_SQUARES ) | epTargetNext;
        mHash                       = this->CalcHash();

        if( matWhite && matBlack )
        {
            // Incremental update of mWhiteMaterial and mBlackMaterial using tables

            SIMD    moveSrcType     = CalcPieceType( srcKnight, srcBishop,  srcRook, srcQueen, srcKing );
            SIMD    moveSrcValue    = LoadIndirect32( (const i32*) matWhite->mValue, Shift< 6 >( moveSrcType )  + moveSrc );

            SIMD    moveDestType    = CalcPieceType( destKnight, destBishop,  destRook, destQueen, destKing );
            SIMD    moveDestValue   = LoadIndirect32( (const i32*) matWhite->mValue, Shift< 6 >( moveDestType ) + moveDest );

            SIMD    captType        = CalcPieceType( blackKnights & destBit, blackBishops & destBit, blackRooks & destBit, blackQueens & destBit, blackKing & destBit );
            SIMD    captTypeMask    = SelectIfNotZero( (blackPawns & destBit) | epVictimNow | captType, MaskAllSet< SIMD >() );
            SIMD    captSquareIndex = moveDest - SelectIfNotZero( epVictimNow, (SIMD) 8 );
            SIMD    captValue       = LoadIndirectMasked32( (const i32*) matBlack->mValue, Shift< 6 >( captType ) + FlipSquareIndex( captSquareIndex ), captTypeMask );

            SIMD    castleFixKing   = SelectIfNotZero( castleRookKing,  (SIMD) matWhite->mCastlingKingside );
            SIMD    castleFixQueen  = SelectIfNotZero( castleRookQueen, (SIMD) matWhite->mCastlingQueenside );

            mWhiteMaterial          = mWhiteMaterial + moveDestValue - moveSrcValue + castleFixKing + castleFixQueen; 
            mBlackMaterial          = mBlackMaterial - captValue;
        }
    }


    /// Calculate the "position hash".
    
    PDECL SIMD CalcHash() const
    {
        SIMD    whitePawns          = SelectWithMask( mBoardFlipped, ByteSwap( mBlackPawns    ), mWhitePawns    );
        SIMD    whiteKnights        = SelectWithMask( mBoardFlipped, ByteSwap( mBlackKnights  ), mWhiteKnights  );
        SIMD    whiteBishops        = SelectWithMask( mBoardFlipped, ByteSwap( mBlackBishops  ), mWhiteBishops  );
        SIMD    whiteRooks          = SelectWithMask( mBoardFlipped, ByteSwap( mBlackRooks    ), mWhiteRooks    );
        SIMD    whiteQueens         = SelectWithMask( mBoardFlipped, ByteSwap( mBlackQueens   ), mWhiteQueens   );
        SIMD    whiteKing           = SelectWithMask( mBoardFlipped, ByteSwap( mBlackKing     ), mWhiteKing     );
        SIMD    blackPawns          = SelectWithMask( mBoardFlipped, ByteSwap( mWhitePawns    ), mBlackPawns    );
        SIMD    blackKnights        = SelectWithMask( mBoardFlipped, ByteSwap( mWhiteKnights  ), mBlackKnights  );
        SIMD    blackBishops        = SelectWithMask( mBoardFlipped, ByteSwap( mWhiteBishops  ), mBlackBishops  );
        SIMD    blackRooks          = SelectWithMask( mBoardFlipped, ByteSwap( mWhiteRooks    ), mBlackRooks    );
        SIMD    blackQueens         = SelectWithMask( mBoardFlipped, ByteSwap( mWhiteQueens   ), mBlackQueens   );
        SIMD    blackKing           = SelectWithMask( mBoardFlipped, ByteSwap( mWhiteKing     ), mBlackKing     );
        SIMD    castlingAndEP       = SelectWithMask( mBoardFlipped, ByteSwap( mCastlingAndEP ), mCastlingAndEP );

        SIMD    allPawnsEtc         = (whitePawns | blackPawns) ^ mWhiteToMove;
        SIMD    hash0               = XorShiftA( XorShiftB( XorShiftC( (SIMD) HASH_SEED0 ^ allPawnsEtc )   ^ blackKnights ) ^ whiteRooks  );              
        SIMD    hash1               = XorShiftA( XorShiftC( XorShiftB( (SIMD) HASH_SEED1 ^ castlingAndEP ) ^ whiteBishops ) ^ blackQueens );             
        SIMD    hash2               = XorShiftD( XorShiftC( XorShiftB( (SIMD) HASH_SEED2 ^ whiteKing )     ^ blackBishops ) ^ whiteQueens );             
        SIMD    hash3               = XorShiftD( XorShiftB( XorShiftC( (SIMD) HASH_SEED3 ^ blackKing )     ^ whiteKnights ) ^ blackRooks  );        
        SIMD    hash                = XorShiftB( hash0 - hash2 ) ^ XorShiftC( hash1 - hash3 );

        return( hash );
    }


    /// Generate a map of valid moves from the current position.
    ///
    /// \param  dest    Target move map for storing result
    
    PDECL void CalcMoveMap( MoveMapT< SIMD >* RESTRICT dest ) const
    {
        SIMD    whitePawns          = mWhitePawns;    
        SIMD    whiteKnights        = mWhiteKnights;  
        SIMD    whiteBishops        = mWhiteBishops;  
        SIMD    whiteRooks          = mWhiteRooks;    
        SIMD    whiteQueens         = mWhiteQueens;   
        SIMD    whiteKing           = mWhiteKing;     
        SIMD    blackPawns          = mBlackPawns;    
        SIMD    blackKnights        = mBlackKnights;  
        SIMD    blackBishops        = mBlackBishops;  
        SIMD    blackRooks          = mBlackRooks;    
        SIMD    blackQueens         = mBlackQueens;   
        SIMD    blackKing           = mBlackKing;     
        SIMD    castlingAndEP       = mCastlingAndEP;

        SIMD    whiteDiag           = whiteBishops | whiteQueens;
        SIMD    whiteOrtho          = whiteRooks | whiteQueens;
        SIMD    whitePieces         = whitePawns | whiteKnights | whiteBishops | whiteRooks | whiteQueens | whiteKing;
        SIMD    blackDiag           = blackBishops | blackQueens;
        SIMD    blackOrtho          = blackRooks | blackQueens;
        SIMD    blackPieces         = blackPawns | blackKnights | blackBishops | blackRooks | blackQueens | blackKing;
        SIMD    allPieces           = blackPieces | whitePieces;
        SIMD    empty               = ~allPieces;

        SIMD    kingViewN           = MaskOut( SlideIntoN(  SlideIntoN(  whiteKing, empty, whitePieces ), empty, blackOrtho ), whiteKing );
        SIMD    kingViewNW          = MaskOut( SlideIntoNW( SlideIntoNW( whiteKing, empty, whitePieces ), empty, blackDiag  ), whiteKing );
        SIMD    kingViewW           = MaskOut( SlideIntoW(  SlideIntoW(  whiteKing, empty, whitePieces ), empty, blackOrtho ), whiteKing );
        SIMD    kingViewSW          = MaskOut( SlideIntoSW( SlideIntoSW( whiteKing, empty, whitePieces ), empty, blackDiag  ), whiteKing );
        SIMD    kingViewS           = MaskOut( SlideIntoS(  SlideIntoS(  whiteKing, empty, whitePieces ), empty, blackOrtho ), whiteKing );
        SIMD    kingViewSE          = MaskOut( SlideIntoSE( SlideIntoSE( whiteKing, empty, whitePieces ), empty, blackDiag  ), whiteKing );
        SIMD    kingViewE           = MaskOut( SlideIntoE(  SlideIntoE(  whiteKing, empty, whitePieces ), empty, blackOrtho ), whiteKing );
        SIMD    kingViewNE          = MaskOut( SlideIntoNE( SlideIntoNE( whiteKing, empty, whitePieces ), empty, blackDiag  ), whiteKing );

        SIMD    kingDangerN         = SelectIfNotZero( (kingViewN  & blackPieces), kingViewN  );
        SIMD    kingDangerNW        = SelectIfNotZero( (kingViewNW & blackPieces), kingViewNW ) | StepNW( whiteKing, blackPawns );
        SIMD    kingDangerW         = SelectIfNotZero( (kingViewW  & blackPieces), kingViewW  );
        SIMD    kingDangerSW        = SelectIfNotZero( (kingViewSW & blackPieces), kingViewSW );
        SIMD    kingDangerS         = SelectIfNotZero( (kingViewS  & blackPieces), kingViewS  );
        SIMD    kingDangerSE        = SelectIfNotZero( (kingViewSE & blackPieces), kingViewSE );
        SIMD    kingDangerE         = SelectIfNotZero( (kingViewE  & blackPieces), kingViewE  );
        SIMD    kingDangerNE        = SelectIfNotZero( (kingViewNE & blackPieces), kingViewNE ) | StepNE( whiteKing, blackPawns );
        SIMD    kingDangerKnights   = StepKnights( whiteKing, blackKnights );

        SIMD    pinnedLineN         = SelectIfNotZero( (kingDangerN  & whitePieces), kingDangerN  );
        SIMD    pinnedLineNW        = SelectIfNotZero( (kingDangerNW & whitePieces), kingDangerNW );
        SIMD    pinnedLineW         = SelectIfNotZero( (kingDangerW  & whitePieces), kingDangerW  );
        SIMD    pinnedLineSW        = SelectIfNotZero( (kingDangerSW & whitePieces), kingDangerSW );
        SIMD    pinnedLineS         = SelectIfNotZero( (kingDangerS  & whitePieces), kingDangerS  );
        SIMD    pinnedLineSE        = SelectIfNotZero( (kingDangerSE & whitePieces), kingDangerSE );
        SIMD    pinnedLineE         = SelectIfNotZero( (kingDangerE  & whitePieces), kingDangerE  );
        SIMD    pinnedLineNE        = SelectIfNotZero( (kingDangerNE & whitePieces), kingDangerNE );
        SIMD    pinnedNS            = pinnedLineN  | pinnedLineS; 
        SIMD    pinnedNWSE          = pinnedLineNW | pinnedLineSE;
        SIMD    pinnedWE            = pinnedLineW  | pinnedLineE; 
        SIMD    pinnedSWNE          = pinnedLineSW | pinnedLineNE;
        SIMD    notPinned           = ~(pinnedNS | pinnedNWSE | pinnedWE | pinnedSWNE);

        SIMD    maskAllSet          = MaskAllSet< SIMD >();
        SIMD    checkMaskN          = SelectIfNotZero( (kingDangerN  ^ pinnedLineN ), kingDangerN , maskAllSet );
        SIMD    checkMaskNW         = SelectIfNotZero( (kingDangerNW ^ pinnedLineNW), kingDangerNW, maskAllSet );
        SIMD    checkMaskW          = SelectIfNotZero( (kingDangerW  ^ pinnedLineW ), kingDangerW , maskAllSet );
        SIMD    checkMaskSW         = SelectIfNotZero( (kingDangerSW ^ pinnedLineSW), kingDangerSW, maskAllSet );
        SIMD    checkMaskS          = SelectIfNotZero( (kingDangerS  ^ pinnedLineS ), kingDangerS , maskAllSet );
        SIMD    checkMaskSE         = SelectIfNotZero( (kingDangerSE ^ pinnedLineSE), kingDangerSE, maskAllSet );
        SIMD    checkMaskE          = SelectIfNotZero( (kingDangerE  ^ pinnedLineE ), kingDangerE , maskAllSet );
        SIMD    checkMaskNE         = SelectIfNotZero( (kingDangerNE ^ pinnedLineNE), kingDangerNE, maskAllSet );
        SIMD    checkMaskKnights    = SelectIfNotZero( kingDangerKnights, kingDangerKnights, maskAllSet );
        SIMD    checkMask           = checkMaskN & checkMaskNW & checkMaskW & checkMaskSW & checkMaskS & checkMaskSE & checkMaskE & checkMaskNE & checkMaskKnights;

        SIMD    slidingMovesN       = MaskOut( SlideIntoN(  whiteOrtho & (notPinned | pinnedNS  ), empty, blackPieces ), whiteOrtho );
        SIMD    slidingMovesNW      = MaskOut( SlideIntoNW( whiteDiag  & (notPinned | pinnedNWSE), empty, blackPieces ), whiteDiag  );
        SIMD    slidingMovesW       = MaskOut( SlideIntoW(  whiteOrtho & (notPinned | pinnedWE  ), empty, blackPieces ), whiteOrtho );
        SIMD    slidingMovesSW      = MaskOut( SlideIntoSW( whiteDiag  & (notPinned | pinnedSWNE), empty, blackPieces ), whiteDiag  );
        SIMD    slidingMovesS       = MaskOut( SlideIntoS(  whiteOrtho & (notPinned | pinnedNS  ), empty, blackPieces ), whiteOrtho );
        SIMD    slidingMovesSE      = MaskOut( SlideIntoSE( whiteDiag  & (notPinned | pinnedNWSE), empty, blackPieces ), whiteDiag  );
        SIMD    slidingMovesE       = MaskOut( SlideIntoE(  whiteOrtho & (notPinned | pinnedWE  ), empty, blackPieces ), whiteOrtho );
        SIMD    slidingMovesNE      = MaskOut( SlideIntoNE( whiteDiag  & (notPinned | pinnedSWNE), empty, blackPieces ), whiteDiag  );

        SIMD    epTarget            = castlingAndEP & EP_SQUARES;
        SIMD    epVictim            = Shift< SHIFT_S >( epTarget );
        SIMD    epCaptor1           = whitePawns & Shift< SHIFT_W >( epVictim );
        SIMD    epCaptor2           = whitePawns & Shift< SHIFT_E >( epVictim );

        SIMD    epDiscCheckNW       = StepNW( PropNW( whiteKing, empty | epVictim ) ) & blackDiag;
        SIMD    epDiscCheckSW       = StepSW( PropSW( whiteKing, empty | epVictim ) ) & blackDiag;
        SIMD    epDiscCheckSE       = StepSE( PropSE( whiteKing, empty | epVictim ) ) & blackDiag;
        SIMD    epDiscCheckNE       = StepNE( PropNE( whiteKing, empty | epVictim ) ) & blackDiag;
        SIMD    epDiscCheckW1       = StepW(  PropW(  whiteKing, empty | epVictim | epCaptor1 ) );
        SIMD    epDiscCheckW2       = StepW(  PropW(  whiteKing, empty | epVictim | epCaptor2 ) );
        SIMD    epDiscCheckE1       = StepE(  PropE(  whiteKing, empty | epVictim | epCaptor1 ) );
        SIMD    epDiscCheckE2       = StepE(  PropE(  whiteKing, empty | epVictim | epCaptor2 ) );
        SIMD    epDiscCheckW        = (epDiscCheckW1 | epDiscCheckW2) & blackOrtho;
        SIMD    epDiscCheckE        = (epDiscCheckE1 | epDiscCheckE2) & blackOrtho;
        SIMD    epDiscCheck         = epDiscCheckNW | epDiscCheckW | epDiscCheckSW | epDiscCheckSE | epDiscCheckE | epDiscCheckNE;
        SIMD    epValidTarget       = SelectIfZero( epDiscCheck, epTarget );

        SIMD    pawnCheckMask       = checkMask | (epValidTarget & Shift< SHIFT_N >( checkMask ));
        SIMD    pawnAttacksNW       = StepNW( whitePawns & (notPinned | pinnedNWSE), blackPieces | epValidTarget ) & pawnCheckMask;
        SIMD    pawnAttacksNE       = StepNE( whitePawns & (notPinned | pinnedSWNE), blackPieces | epValidTarget ) & pawnCheckMask;
        SIMD    pawnClearN          = StepN(  whitePawns & (notPinned | pinnedNS), empty );
        SIMD    pawnDoublesN        = StepN(  pawnClearN & RANK_3, empty ) & checkMask;
        SIMD    pawnMovesN          = pawnClearN & checkMask;

        SIMD    mobileKnights       = whiteKnights & notPinned;
        SIMD    knightTargets       = ~whitePieces & checkMask;
        SIMD    knightMovesNNW      = Shift< SHIFT_N + SHIFT_NW >( mobileKnights & (~FILE_A)           ) & knightTargets;
        SIMD    knightMovesWNW      = Shift< SHIFT_W + SHIFT_NW >( mobileKnights & (~FILE_A & ~FILE_B) ) & knightTargets;
        SIMD    knightMovesWSW      = Shift< SHIFT_W + SHIFT_SW >( mobileKnights & (~FILE_A & ~FILE_B) ) & knightTargets;
        SIMD    knightMovesSSW      = Shift< SHIFT_S + SHIFT_SW >( mobileKnights & (~FILE_A)           ) & knightTargets;
        SIMD    knightMovesSSE      = Shift< SHIFT_S + SHIFT_SE >( mobileKnights & (~FILE_H)           ) & knightTargets;
        SIMD    knightMovesESE      = Shift< SHIFT_E + SHIFT_SE >( mobileKnights & (~FILE_H & ~FILE_G) ) & knightTargets;
        SIMD    knightMovesENE      = Shift< SHIFT_E + SHIFT_NE >( mobileKnights & (~FILE_H & ~FILE_G) ) & knightTargets;
        SIMD    knightMovesNNE      = Shift< SHIFT_N + SHIFT_NE >( mobileKnights & (~FILE_H)           ) & knightTargets;

        SIMD    blackPawnsCon       = StepSW( blackPawns ) | StepSE( blackPawns ); 
        SIMD    blackKnightsCon     = StepKnights( blackKnights );
        SIMD    blackDiagCon        = SlideIntoExDiag(  blackDiag,  empty | whiteKing, blackPieces );
        SIMD    blackOrthoCon       = SlideIntoExOrtho( blackOrtho, empty | whiteKing, blackPieces );
        SIMD    blackKingCon        = StepOut( blackKing );
        SIMD    blackControl        = blackPawnsCon | blackKnightsCon | blackDiagCon | blackOrthoCon | blackKingCon;

        SIMD    castleKingBlocks    = allPieces    & (SQUARE_F1 | SQUARE_G1);                
        SIMD    castleQueenBlocks   = allPieces    & (SQUARE_B1 | SQUARE_C1 | SQUARE_D1); 
        SIMD    castleKingThreats   = blackControl & (SQUARE_E1 | SQUARE_F1 | SQUARE_G1); 
        SIMD    castleQueenThreats  = blackControl & (SQUARE_C1 | SQUARE_D1 | SQUARE_E1); 
        SIMD    castleKingUnavail   = MaskOut( (SIMD) SQUARE_H1, castlingAndEP & whiteRooks );
        SIMD    castleQueenUnavail  = MaskOut( (SIMD) SQUARE_A1, castlingAndEP & whiteRooks );
        SIMD    castleKing          = SelectIfZero( (castleKingBlocks  | castleKingThreats  | castleKingUnavail),  (SIMD) SQUARE_G1 );
        SIMD    castleQueen         = SelectIfZero( (castleQueenBlocks | castleQueenThreats | castleQueenUnavail), (SIMD) SQUARE_C1 );
        SIMD    castlingMoves       = castleKing | castleQueen;
        SIMD    kingMoves           = StepOut( whiteKing, ~whitePieces & ~blackControl );

        dest->mSlidingMovesNW       = slidingMovesNW;
        dest->mSlidingMovesNE       = slidingMovesNE;
        dest->mSlidingMovesSW       = slidingMovesSW;
        dest->mSlidingMovesSE       = slidingMovesSE;
        dest->mSlidingMovesN        = slidingMovesN;
        dest->mSlidingMovesW        = slidingMovesW;
        dest->mSlidingMovesE        = slidingMovesE;
        dest->mSlidingMovesS        = slidingMovesS;

        dest->mKnightMovesNNW       = knightMovesNNW;
        dest->mKnightMovesNNE       = knightMovesNNE;
        dest->mKnightMovesWNW       = knightMovesWNW;
        dest->mKnightMovesENE       = knightMovesENE;
        dest->mKnightMovesWSW       = knightMovesWSW;
        dest->mKnightMovesESE       = knightMovesESE;
        dest->mKnightMovesSSW       = knightMovesSSW;
        dest->mKnightMovesSSE       = knightMovesSSE;

        dest->mPawnMovesN           = pawnMovesN;
        dest->mPawnDoublesN         = pawnDoublesN;
        dest->mPawnAttacksNE        = pawnAttacksNE;
        dest->mPawnAttacksNW        = pawnAttacksNW;

        dest->mCastlingMoves        = castlingMoves;
        dest->mKingMoves            = kingMoves;
        dest->mCheckMask            = checkMask;
        dest->mBlackControl         = blackControl;
    }


    //--------------------------------------------------------------------------
    /// Flip the white/black pieces to view the board "from the other side".
    ///
    /// Note that this doesn't change the side to move, or any other game state.
    ///
    /// \param  prev    Position to flip
    
    PDECL void FlipFrom( const PositionT< SIMD >& prev )
    {
        SIMD    newWhitePawns       = ByteSwap( prev.mBlackPawns   );
        SIMD    newWhiteKnights     = ByteSwap( prev.mBlackKnights );
        SIMD    newWhiteBishops     = ByteSwap( prev.mBlackBishops );
        SIMD    newWhiteRooks       = ByteSwap( prev.mBlackRooks   );
        SIMD    newWhiteQueens      = ByteSwap( prev.mBlackQueens  );
        SIMD    newWhiteKing        = ByteSwap( prev.mBlackKing    );
        SIMD    newBlackPawns       = ByteSwap( prev.mWhitePawns   );
        SIMD    newBlackKnights     = ByteSwap( prev.mWhiteKnights );
        SIMD    newBlackBishops     = ByteSwap( prev.mWhiteBishops );
        SIMD    newBlackRooks       = ByteSwap( prev.mWhiteRooks   );
        SIMD    newBlackQueens      = ByteSwap( prev.mWhiteQueens  );
        SIMD    newBlackKing        = ByteSwap( prev.mWhiteKing    );
        SIMD    newWhiteMaterial    = prev.mBlackMaterial;
        SIMD    newBlackMaterial    = prev.mWhiteMaterial;

        mWhitePawns                 = newWhitePawns;  
        mWhiteKnights               = newWhiteKnights;
        mWhiteBishops               = newWhiteBishops;
        mWhiteRooks                 = newWhiteRooks;  
        mWhiteQueens                = newWhiteQueens; 
        mWhiteKing                  = newWhiteKing;   

        mBlackPawns                 = newBlackPawns;  
        mBlackKnights               = newBlackKnights;
        mBlackBishops               = newBlackBishops;
        mBlackRooks                 = newBlackRooks;  
        mBlackQueens                = newBlackQueens; 
        mBlackKing                  = newBlackKing;   

        mCastlingAndEP              = ByteSwap( prev.mCastlingAndEP );
        mHash                       = prev.mHash;
        mBoardFlipped               = ~prev.mBoardFlipped;
        mWhiteToMove                = prev.mWhiteToMove;
        mHalfmoveClock              = prev.mHalfmoveClock;
        mFullmoveNum                = prev.mFullmoveNum;
        mWhiteMaterial              = newWhiteMaterial;
        mBlackMaterial              = newBlackMaterial;
    }


    PDECL void  CalcMaterial( const MaterialTable* matWhite, const MaterialTable* matBlack ) {}

    PDECL void  FlipInPlace()           { this->FlipFrom( *this ); }
    PDECL int   GetPlyZeroBased() const { return( (int) ((mFullmoveNum - 1) * 2 + (1 - mWhiteToMove)) ); }
};


// Material tables 

template<> 
PDECL void PositionT< u64 >::CalcMaterial( const MaterialTable* matWhite, const MaterialTable* matBlack )
{
    mWhiteMaterial = 0;
    mBlackMaterial = 0;

    for( u64 i = mWhitePawns;               i != 0; mWhiteMaterial += matWhite->mValue[PAWN  ][ConsumeLowestBitIndex( i )] ) {}
    for( u64 i = mWhiteKnights;             i != 0; mWhiteMaterial += matWhite->mValue[KNIGHT][ConsumeLowestBitIndex( i )] ) {}
    for( u64 i = mWhiteBishops;             i != 0; mWhiteMaterial += matWhite->mValue[BISHOP][ConsumeLowestBitIndex( i )] ) {}
    for( u64 i = mWhiteRooks;               i != 0; mWhiteMaterial += matWhite->mValue[ROOK  ][ConsumeLowestBitIndex( i )] ) {}
    for( u64 i = mWhiteQueens;              i != 0; mWhiteMaterial += matWhite->mValue[QUEEN ][ConsumeLowestBitIndex( i )] ) {}
    for( u64 i = mWhiteKing;                i != 0; mWhiteMaterial += matWhite->mValue[KING  ][ConsumeLowestBitIndex( i )] ) {}
                                                                                            
    for( u64 i = ByteSwap( mBlackPawns );   i != 0; mBlackMaterial += matBlack->mValue[PAWN  ][ConsumeLowestBitIndex( i )] ) {}
    for( u64 i = ByteSwap( mBlackKnights ); i != 0; mBlackMaterial += matBlack->mValue[KNIGHT][ConsumeLowestBitIndex( i )] ) {}
    for( u64 i = ByteSwap( mBlackBishops ); i != 0; mBlackMaterial += matBlack->mValue[BISHOP][ConsumeLowestBitIndex( i )] ) {}
    for( u64 i = ByteSwap( mBlackRooks );   i != 0; mBlackMaterial += matBlack->mValue[ROOK  ][ConsumeLowestBitIndex( i )] ) {}
    for( u64 i = ByteSwap( mBlackQueens );  i != 0; mBlackMaterial += matBlack->mValue[QUEEN ][ConsumeLowestBitIndex( i )] ) {}
    for( u64 i = ByteSwap( mBlackKing );    i != 0; mBlackMaterial += matBlack->mValue[KING  ][ConsumeLowestBitIndex( i )] ) {}
}



#endif // PIGEON_POSITION_H__
};
