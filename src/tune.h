// tune.h - PIGEON CHESS ENGINE (c) 2012-2016 Stuart Riffle
        
namespace Pigeon {
#ifndef PIGEON_TUNE_H__
#define PIGEON_TUNE_H__


class AutoTuner : public AmoebaOptimizer
{
    struct TestPos
    {
        std::vector< u64 >  mCoeff;
        double              mExpected;
    };


    bool IsValidTestPos( const Position& pos )
    {
        return( true );
    }

    bool LoadGameLine( const char* str )
    {
        Tokenizer tok( str );

        double expected = 0;

        if( tok.Consume( "W" ) )
            expected = 1.0;
        else if( tok.Consume( "L" ) )
            expected = 0.0;
        else if( tok.Consume( "D" ) )
            expected = 0.5;
        else
            return( false );

        Position pos;
        pos.Reset();

        for( const char* movetext = tokens.ConsumeNext(); movetext; movetext = tokens.ConsumeNext() )
        {
            MoveSpec spec;
            FEN::StringToMoveSpec( movetext, spec );

            pos.Step( spec );
            expected = 1.0 - expected;

            if( this->IsValidTestPos( pos ) )
            {
                TestPos testPos;

                testPos.mExpected = expected;
                testPos.mCoeff.resize( EVAL_TERMS );

                MoveMap mmap;
                pos.CalcMoveMap( &mmap );
                pos.CalcEvalTerms< 1, u64 >( pos, mmap, &testPos.mCoeff[0] );

                mTestPos.push_back( testPos );
            }
        }

        return( true );
    }

protected:
    virtual void CalcError( ParameterSet& point )
    {
        double  accum   = 0;
        u64     count   = 0;

        for( std::list< TestPos >::iterator iter = mTestPos.begin(); iter != mTestPos.end(); ++iter )
        {
            const TestPos& testPos = *iter;

            double score = 0;
            for( size_t i = 0; i < point.mElem.size(); i++ )
                score += point.mElem[i] * testPos.mCoeff[i];

            double sig = 1.0 / (1.0 + exp( -score / 400.0 ));
            double err = (testPos.mExpected - sig);

            accum += (err * err);
            count++;
        }

        point.mError = accum / count;
    }
};


#endif // PIGEON_TUNE_H__
};

