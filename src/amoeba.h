// amoeba.h - PIGEON CHESS ENGINE (c) 2012-2016 Stuart Riffle
        
namespace Pigeon {
#ifndef PIGEON_AMOEBA_H__
#define PIGEON_AMOEBA_H__

struct ParameterSet
{
    std::vector< float >    mElem;
    double                  mError;

    ParameterSet( size_t n = 0 ) : mElem( n ), mError( 0 ) {}
    ParameterSet( const ParameterSet& rhs ) : mElem( rhs.mElem ), mError( rhs.mError ) {}

    ParameterSet&   operator -=( const ParameterSet& rhs )          { for( size_t i = 0; i < mElem.size(); i++ ) mElem[i] -= rhs.mElem[i]; return( *this ); }
    ParameterSet&   operator +=( const ParameterSet& rhs )          { for( size_t i = 0; i < mElem.size(); i++ ) mElem[i] += rhs.mElem[i]; return( *this ); }
    ParameterSet&   operator *=( float scale )                      { for( size_t i = 0; i < mElem.size(); i++ ) mElem[i] *= scale;        return( *this ); }
    bool            operator  <( const ParameterSet& rhs ) const    { return( mError < rhs.mError ); }
};

class AmoebaOptimizer
{
    std::vector< ParameterSet > mPoint;
    size_t                      mIterations;

protected:
    virtual void CalcError( ParameterSet& point ) = 0;

    ParameterSet GeneratePoint( const ParameterSet& centroid, float factor )
    {
        const ParameterSet& worst = mPoint[mPoint.size() - 1];

        ParameterSet result( centroid );
        result -= worst;
        result *= factor;
        result += centroid;

        for( size_t i = 0; i < result.mElem.size(); i++ )
            if( result.mElem[i] < 0 )
                result.mElem[i] = 0;

        this->CalcError( result );
        return( result );
    }

    ParameterSet CalcCentroid()
    {
        ParameterSet centroid( mPoint[0] );
        for( size_t i = 1; i < mPoint.size() - 1; i++ )
            centroid += mPoint[i];

        centroid *= 1.0f / (mPoint.size() - 1);
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
            mPoint[i] *= 0.5f;
            mPoint[i] += best;

            this->CalcError( mPoint[i] );
        }

        std::sort( mPoint.begin(), mPoint.end() );
    }


    void UpdateSimplex()
    {
        ParameterSet&   best        = mPoint[0];
        ParameterSet&   worst       = mPoint[mPoint.size() - 1];
        ParameterSet&   secondWorst = mPoint[mPoint.size() - 2];
        ParameterSet    centroid    = this->CalcCentroid();
        ParameterSet    reflected   = this->GeneratePoint( centroid, 1.0f );

        if( (reflected.mError < secondWorst.mError) && (reflected.mError >= best.mError) )
        {
            this->ReplaceWorst( reflected );
        }
        else if( reflected.mError < best.mError )
        {
            ParameterSet expanded = this->GeneratePoint( centroid, 2.0f );

            this->ReplaceWorst( (expanded.mError < reflected.mError)? expanded : reflected );
        }
        else
        {
            ParameterSet contracted = this->GeneratePoint( centroid, 0.5f );

            if( contracted.mError < worst.mError )
            {
                this->ReplaceWorst( contracted );
            }
            else
            {
                this->ShrinkSimplex();
            }
        }

    }

public:

    AmoebaOptimizer()
    {
    }

    void Initialize( const ParameterSet& init, const ParameterSet& range ) 
    {
        mIterations = 0;
        mPoint.reserve( init.mElem.size() + 1 );

        mPoint.push_back( init );
        for( size_t i = 0; i < init.mElem.size(); i++ )
        {
            ParameterSet extent( init );
            extent.mElem[i] += range.mElem[i];

            mPoint.push_back( extent );
        }
    }

    void Reset()
    {
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

