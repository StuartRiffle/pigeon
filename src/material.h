// material.h - PIGEON CHESS ENGINE (c) 2012-2016 Stuart Riffle

namespace Pigeon {
#ifndef PIGEON_MATERIAL_H__
#define PIGEON_MATERIAL_H__


/// A piece-square table for evaluation.
//
struct MaterialTable
{
    i32     mValue[6][64];
    i64     mCastlingQueenside;
    i64     mCastlingKingside;

    void CalcCastlingFixup()
    {
        mCastlingQueenside = mValue[ROOK][D1] - mValue[ROOK][A1];
        mCastlingKingside  = mValue[ROOK][F1] - mValue[ROOK][H1];
    }
};


#endif // PIGEON_MATERIAL_H__
};
