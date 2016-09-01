// tune.h - PIGEON CHESS ENGINE (c) 2012-2016 Stuart Riffle
        
namespace Pigeon {
#ifndef PIGEON_TUNE_H__
#define PIGEON_TUNE_H__


class AutoTuner : public AmoebaOptimizer
{
    struct TestPos
    {
        i8      mCoeff[EVAL_TERMS];
        float   mExpected;
        float   mWeight;
    };

    std::vector< TestPos >  mTestPos;
    Evaluator               mEvaluator;

public:
    AutoTuner()
    {
        ParameterSet init(  EVAL_TERMS );
        ParameterSet range( EVAL_TERMS );
        EvalWeight   weights[EVAL_TERMS];

        mEvaluator.GenerateWeights( weights, 1.0f );

        for( int i = 0; i < EVAL_TERMS; i++ )
        {
            float weight = weights[i] * 1.0f / WEIGHT_SCALE;

            init.mElem[i]  = weight;
            range.mElem[i] = (weight < 100)? (weight * 0.5f) : 100.0f;
        }

        range.mElem[EVAL_PAWNS] = 0;
        range.mElem[EVAL_KINGS] = 0;
        range.mElem[EVAL_ROOKS_CONNECTED] = 0;

        this->Initialize( init, range );

        mTestPos.reserve( 100000 );
    }

    bool LoadGameLine( const char* str )
    {
        Tokenizer tok( str );

        float expected = 0;

        if( tok.Consume( "W" ) )
            expected = 1.0f;
        else if( tok.Consume( "L" ) )
            expected = 0.0f;
        else if( tok.Consume( "D" ) )
            expected = 0.5f;
        else
            return( false );

        Position pos;
        pos.Reset();

        for( const char* movetext = tok.ConsumeNext(); movetext; movetext = tok.ConsumeNext() )
        {
            MoveSpec spec;
            FEN::StringToMoveSpec( movetext, spec );
            if( spec.IsPromotion() )
                break;

            pos.Step( spec );
            expected = 1.0f - expected;

            MoveMap mmap;
            pos.CalcMoveMap( &mmap );

            if( this->IsValidTestPos( pos, mmap ) )
            {
                TestPos testPos;

                testPos.mExpected = expected;

                float phase = mEvaluator.CalcGamePhase< 1 >( pos );
                testPos.mWeight = 1.0f;//1.0f - abs( 1.0f - phase );

                //if( (phase >= 1.0f) && (phase < 1.9f) )//testPos.mWeight > 0.3f )
                //if( phase >= 1.0f )
                {
                    u64 coeff[EVAL_TERMS];
                    mEvaluator.CalcEvalTerms< 1, u64 >( pos, mmap, coeff );

                    for( int i = 0; i < EVAL_TERMS; i++ )
                        testPos.mCoeff[i] = (int) coeff[i];

                    mTestPos.push_back( testPos );
                }
            }
        }

        return( true );
    }

    void Dump()
    {
        const ParameterSet& best = this->GetBest();

        printf( "\n" );
        printf( "%" PRId64 " iterations, error %.15f\n", this->GetIterCount(), best.mError );

        for( size_t i = 0; i < best.mElem.size(); i++ )
            printf( "%-22s %9.2f\n", mEvaluator.GetWeightName( (int) i ), best.mElem[i] );

    }

protected:
    bool IsValidTestPos( const Position& pos, const MoveMap& mmap )
    {
        MoveList moves;
        moves.UnpackMoveMap( pos, mmap );
        moves.DiscardMovesBelow( CAPTURE_WINNING );

        if( moves.mCount == 0 )
            return( true );

        return( false );
    }

    virtual void CalcError( ParameterSet& point )
    {
        double  accum   = 0;
        double  divisor = 0;
        double  factor  = -1.0 / 400.0;

        #pragma omp parallel for reduction(+: accum,divisor) schedule(static)
        for( int i = 0; i < (int) mTestPos.size(); i++ )
        {
            //const TestPos& testPos = *iter;
            const TestPos& testPos = mTestPos[i];

            double score = 0;
            for( size_t i = 0; i < EVAL_TERMS; i++ )
                score += point.mElem[i] * testPos.mCoeff[i];

            double sig = 1.0 / (1.0 + pow( 10.0, score * factor ));
            double err = (sig - testPos.mExpected);

            accum   = accum + (err * err) * testPos.mWeight;
            divisor = divisor + testPos.mWeight;
        }

        point.mError = (accum / divisor);
        printf( "." );
    }
};


#endif // PIGEON_TUNE_H__
};

