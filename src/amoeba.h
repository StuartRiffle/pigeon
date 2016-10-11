// amoeba.h - PIGEON CHESS ENGINE (c) 2012-2016 Stuart Riffle
        
namespace Pigeon {
#ifndef PIGEON_AMOEBA_H__
#define PIGEON_AMOEBA_H__


/// A point in n-space, with associated error value
///
/// This is just for use with AmoebaOptimizer, and only implements the
/// operators needed for that.
//
struct ParameterSet
{
    std::vector< double >       mElem;          //< A point in parameter space
    double                      mError;         //< The error calue calculated for this point

    ParameterSet( size_t n = 0 ) : mElem( n ), mError( 0 ) {}
    ParameterSet( const ParameterSet& rhs ) : mElem( rhs.mElem ), mError( rhs.mError ) {}

    ParameterSet&   operator -=( const ParameterSet& rhs )          { for( size_t i = 0; i < mElem.size(); i++ ) mElem[i] -= rhs.mElem[i]; return( *this ); }
    ParameterSet&   operator +=( const ParameterSet& rhs )          { for( size_t i = 0; i < mElem.size(); i++ ) mElem[i] += rhs.mElem[i]; return( *this ); }
    ParameterSet&   operator *=( double scale )                      { for( size_t i = 0; i < mElem.size(); i++ ) mElem[i] *= scale;        return( *this ); }
    bool            operator  <( const ParameterSet& rhs ) const    { return( mError < rhs.mError ); }
};


/// This is a dirty implementation of Nelder-Mead optimization. It tries 
/// to find a minimum in n-space by flopping a simplex of n+1 vertices
/// around (that's the "amoeba"). I like it because it doesn't require a 
/// gradient, and it looks like fun for the simplex.
//
class AmoebaOptimizer
{
    std::vector< ParameterSet > mPoint;         //< The vertices of the simplex, sorted by increasing error
    size_t                      mIterations;    //< Iteration counter

protected:
    virtual void CalcError( ParameterSet& point ) = 0;

    ParameterSet GeneratePoint( const ParameterSet& centroid, float factor )
    {
        const ParameterSet& worst = mPoint[mPoint.size() - 1];

        ParameterSet result( centroid );
        result -= worst;
        result *= factor;
        result += centroid;

        //for( size_t i = 0; i < result.mElem.size(); i++ )
        //    if( result.mElem[i] < 0 )
        //        result.mElem[i] = 0;

        //assert( result.mElem[EVAL_PAWN] == 100.0f );
        //assert( result.mElem[EVAL_KING] == 20000.0f );

        this->CalcError( result );
        return( result );
    }

    ParameterSet CalcCentroid()
    {
        ParameterSet centroid( mPoint[0] );
        for( size_t i = 1; i < mPoint.size() - 1; i++ )
            centroid += mPoint[i];

        centroid *= 1.0 / (mPoint.size() - 1);
        return( centroid );
    }

    void ReplaceWorst( const ParameterSet& point )
    {
        mPoint.resize( mPoint.size() - 1 );
        mPoint.insert( std::upper_bound( mPoint.begin(), mPoint.end(), point ), point );
    }

    void ShrinkSimplex()
    {
        const ParameterSet& best = mPoint[0];

        for( size_t i = 1; i < mPoint.size(); i++ )
        {
            mPoint[i] -= best;
            mPoint[i] *= 0.5;
            mPoint[i] += best;

            printf( "*" );
            this->CalcError( mPoint[i] );
        }

        //this->CalcError( mPoint[0] );
        std::sort( mPoint.begin(), mPoint.end() );
    }


    void UpdateSimplex()
    {
        ParameterSet&   best        = mPoint[0];
        ParameterSet&   worst       = mPoint[mPoint.size() - 1];
        ParameterSet&   secondWorst = mPoint[mPoint.size() - 2];
        ParameterSet    centroid    = this->CalcCentroid();
        ParameterSet    reflected   = this->GeneratePoint( centroid, 1.0 );

        if( (reflected.mError < secondWorst.mError) && (reflected.mError >= best.mError) )
        {
            printf( ":" );
            this->ReplaceWorst( reflected );
        }
        else if( reflected.mError < best.mError )
        //if( reflected.mError < secondWorst.mError )
        {
            ParameterSet expanded = this->GeneratePoint( centroid, 2.0 );

            printf( (expanded.mError < reflected.mError)? "+" : "." );
            this->ReplaceWorst( (expanded.mError < reflected.mError)? expanded : reflected );
        }
        else
        {
            ParameterSet contracted = this->GeneratePoint( centroid, 0.5 );

            if( (contracted.mError < worst.mError) || (reflected.mError < worst.mError) )
            {
                printf( (contracted.mError < reflected.mError)? "-" : "." );
                this->ReplaceWorst( (contracted.mError < reflected.mError)? contracted : reflected );
            }
            else
            {
                double tryThis[] = { 0.25, 0.75, 1.25, 1.75, -0.5, 3 };
                size_t tryCount = sizeof( tryThis ) / sizeof( tryThis[0] );

                for( size_t i = 0; i < tryCount; i++ )
                {
                    printf( "%d", i );
                    ParameterSet guess = this->GeneratePoint( centroid, (float) tryThis[i] );
                    if( guess.mError < worst.mError )
                    {
                        this->ReplaceWorst( guess );
                        return;
                    }
                }

                this->ShrinkSimplex();
            }

            //if( contracted.mError < worst.mError )
            //{
            //    printf( "-" );
            //    this->ReplaceWorst( contracted );
            //}
            //else
            //{
            //    this->ShrinkSimplex();
            //}
        }
    }

public:

    AmoebaOptimizer()
    {
    }

    void Initialize( const ParameterSet& init, const ParameterSet& range ) 
    {
        mIterations = 0;

        mPoint.clear();
        mPoint.reserve( init.mElem.size() + 1 );

        std::default_random_engine generator;
        std::normal_distribution< double > normalDist;

        mPoint.push_back( init );
        for( size_t i = 0; i < init.mElem.size(); i++ )
        {
            ParameterSet extent( init );

            for( size_t j = 0; j < extent.mElem.size(); j++ )
                extent.mElem[j] += range.mElem[j] * normalDist( generator );

            //extent.mElem[i] += range.mElem[i];

            mPoint.push_back( extent );
        }
    }

    void Reset()
    {
        printf( "*" );
        for( size_t i = 0; i < mPoint.size(); i++ )
            this->CalcError( mPoint[i] );

        std::sort( mPoint.begin(), mPoint.end() );
    }

    void Step( int count = 1 )
    {
        while( count-- )
        {
            this->UpdateSimplex();
            mIterations++;
        }
    }

    const ParameterSet& GetBest()
    {
        return( mPoint[0] );
    }

    u64 GetIterCount()
    {
        return( mIterations );
    }
};


#endif // PIGEON_AMOEBA_H__
};

