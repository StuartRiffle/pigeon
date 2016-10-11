F// neural.h - PIGEON CHESS ENGINE (c) 2012-2016 Stuart Riffle

#ifndef PIGEON_NEURAL_H__
#define PIGEON_NEURAL_H__


#define NN_DEFAULT_DECAY_RATE       (1.0f)
#define NN_DEFAULT_LEARNING_RATE    (0.1f)
#define NN_DEFAULT_MOMENTUM         (0.1f)
#define NN_DEFAULT_WEIGHT_CLAMP     (50.0f)
#define NN_BIAS_TERM                (1)


/// Neuron activation function
//
enum ActivationFunc
{
    ACTIVATION_TANH,                                ///< tanh( x )
    ACTIVATION_LOGISTIC,                            ///< Logistic sigmoid function, 1 / (1 - e^-x) == tanh( x * 0.5 ) * 0.5 + 0.5
    ACTIVATION_LINEAR,                              ///< Passthrough
};


struct LayerHeader
{
    i32         mTag;                       ///< Must be SERIALIZED_LAYER_TAG
    i32         mNumInputs;                 ///< Number of layer inputs, including bias term
    i32         mNumOutputs;                ///< Number of layer outputs
    i32         mActivation;                ///< Activation function

    LayerHeader() :
        mTag           (SERIALIZED_LAYER_TAG),
        mNumInputs     (0),
        mNumOutputs    (0),
        mActivation    (ACTIVATION_LINEAR)
    {
    }

    int CalcSerializedSize()
    {
        return( sizeof( LayerHeader ) + (mNumInputs * mNumOutputs) * sizeof( float ) );
    }

    int CalcRuntimeDataSlots( bool allowLearning )
    {
        if( mTag != SERIALIZED_LAYER_TAG )
            return( 0 );

        int runSlots = 
            mNumInputs +                    // inputs
            mNumOutputs +                   // outputs
            (mNumInputs * mNumOutputs);     // weights

        int learnSlots =
            mNumInputs +                    // input errors
            (mNumInputs * mNumOutputs) +    // weight deltas
            (mNumInputs * mNumOutputs) );   // previous weight deltas

        return( runSlots + (allowLearning? learnSlots : 0) );
    }
};

/// Neuron layer
//
struct Layer : public LayerHeader
{
    bool        mLearningEnabled;
    float*      mData;                      ///< One big block to hold all the different buffers described below (mOfs*)
    int         mOfsInput;                  ///< Copy of input data (with a bias term appended)
    int         mOfsOutput;                 ///< Most recent activations
    int         mOfsWeight;                 ///< Weight matrix between inputs and outputs
    int         mOfsInputError;             ///< If learning enabled, the accumulated error for each input term
    int         mOfsDelta;                  ///< If learning enabled, the accumulated updates for the weight matrix
    int         mOfsDeltaPrev;              ///< If learning enabled, the previous iteration's delta (used for momentum)

public:

    Layer() : LayerHeader(), 
        mData              (NULL),
        mOfsInput          (0),
        mOfsInputError     (0),
        mOfsOutput         (0),
        mOfsWeight         (0),
        mOfsDelta          (0),
        mOfsDeltaPrev      (0) 
    {
    }

    Layer( int numInputs, int numOutputs ) :
        mActivationFunc( ACTIVATION_TANH )
    {
        this->SetSize( numInputs, numOutputs );
    }

    void* Serialize( void* dest ) 
    {
        i32* header = (i32*) dest;

        *header++ = SERIALIZED_LAYER_TAG;
        *header++ = mNumInputs;
        *header++ = mNumOutputs;
        *header++ = mActivationFunc;

        float*  dataSer = (float*) header;
        float*  weight  = mData + mOfsWeight;

        for( int idx = 0; idx < mNumInputs * mNumOutputs; idx++ )
            *dataSer++ = weight[idx];

        assert( ((intptr_t) dataSer) - ((intptr_t) dest) == this->CalcSerializedSize() );
        return( dataSer );
    }


    void* Deserialize( void* src, float* dataBlock )
    {
        i32* header = (i32*) src;

        if( *header++ != SERIALIZED_LAYER_TAG )
            return( src );

        mNumInputs  = *header++;
        mNumOutputs = *header++;
        mActivation = *header++;

        this->SetSize( mNumInputs, mNumOutputs );
        this->SetData( dataBlock );

        float*  dataSer = (float*) header;
        float*  weight  = mData + mOfsWeight;

        for( int idx = 0; idx < mNumInputs * mNumOutputs; idx++ )
            weight[idx] = *dataSer++;

        assert( ((intptr_t) dataSer) - ((intptr_t) src) == this->CalcSerializedSize() );
        return( dataSer );
    }

    int SetSize( int numInputs, int numOutputs )
    {
        int count = 0;

        mNumInputs      = numInputs;
        mNumOutputs     = numOutputs;
        mOfsInput       = count;        count += mNumInputs;
        mOfsOutput      = count;        count += mNumOutputs;
        mOfsWeight      = count;        count += (mNumInputs * mNumOutputs);
        mOfsInputError  = count;        count += mNumInputs;
        mOfsDelta       = count;        count += (mNumInputs * mNumOutputs);
        mOfsDeltaPrev   = count;        count += (mNumInputs * mNumOutputs);

        return( count );
    }

    void SetData( float* dataBlock )
    {
        mData = dataBlock;
    }

    void FeedForward( const float* inputSrc )
    {        
        this->LoadInputs( inputSrc );
        this->ApplyWeights();
        this->ApplyActivation();
    }

    void BackpropError( const float* outputErr )
    {
        assert( mAllowLearning );

        this->UpdateInputError( outputErr );
        this->UpdateWeightDeltas( outputErr );
    }

    void UpdateWeights( float learningRate, float decay, float momentum, float weightLimit )
    {
        assert( mAllowLearning );

        float*  weight      = mData + mOfsWeight;
        float*  delta       = mData + mOfsDelta;
        float*  deltaPrev   = mData + mOfsDeltaPrev;
        int     numWeights  = mNumInputs * mNumOutputs;

        for( int idx = 0; idx < numWeights; idx++ )
        {
            float   deltaCurr    = (delta[idx] * learningRate) + (deltaPrev[idx] * momentum);
            float   weightNew    = (weight[idx] * decay) + deltaCurr;
            float   weightClamp  = Min( Max( weightNew, -weightLimit ), weightLimit );

            weight[idx]     = weightClamp;
            deltaPrev[Idx]  = deltaCurr;
            delta[idx]      = 0;
        }
    }    

    void RandomizeWeights()
    {
        float* weight      = mData + mOfsWeight;
        int numWeights  = mNumInputs * mNumOutputs;
        float  scale       = float( 0.1 ) / sqrt( (float) mNumInputs );

        for( int idx = 0; idx < numWeights; idx++ )
            weight[idx] = (rand() / RAND_MAX) * scale;
    }

private:

    void LoadInputs( const float* inputSrc, int count )
    {
        assert( mNumInputs == (count + 1) );

        float* input = mData + mOfsInput;

        for( int in = 0; in < mNumInputs - 1; in++ )
            input[in] = inputSrc[in];

        input[mNumInputs - 1] = 1.0f; // bias term
    }

    void ApplyWeights()
    {
        const float*    input   = mData + mOfsInput;
        const float*    weight  = mData + mOfsWeight;
        float*          output  = mData + mOfsOutput;

        for( int out = 0; out < mNumOutputs; out++ )
        {
            const float* weightRow = weight + (out * mNumInputs);

            float acc = 0;
            for( int in = 0; in < mNumInputs; in++ )
                acc += input[in] * weightRow[in];

            output[out] = acc;
        }
    }

    void ApplyActivation()
    {
        float* output = mData + mOfsOutput;

        switch( mActivation )
        {
        case ACTIVATION_TANH:  
            for( int out = 0; out < mNumOutputs; out++ )
                output[out] = tanh( output[out] );  
            break;

        case ACTIVATION_LOGISTIC:           
            for( int out = 0; out < mNumOutputs; out++ )
                output[out] = 1.0f / (1.0f - exp( -output[out] ));
            break;

        case ACTIVATION_LINEAR: 
        default:    
            break;
        }
    }

    void UpdateInputError( const float* outputErr )
    {
        assert( mAllowLearning );

        const float*    input       = mData + mOfsInput;
        const float*    weight      = mData + mOfsWeight;
        float*          inputErr    = mData + mOfsInputErr;

        for( int in = 0; in < mNumInputs; in++ )
        {
            float acc = 0;
            for( int out = 0; out < mNumOutputs; out++ )
                acc += outputErr[out] * weight[(out * mNumInputs) + in];

            inputErr[in] += input[in] * (1 - input[in]) * acc;
        }
    }

    void UpdateWeightDeltas( const float* outputErr )
    {
        assert( mAllowLearning );

        const float*    input       = mData + mOfsInput;
        float*          delta       = mData + mOfsDelta;

        for( int out = 0; out < mNumOutputs; out++ )
        {
            float* deltaRow = delta + (out * mNumInputs);

            for( int in = 0; in < mNumInputs; in++ )
                deltaRow[in] += outputErr[out] * input[in];
        }
    }    
};


/// Feedforward neural network
//
class Backprop
{
    int                             mNumLayers;     ///< Number of weight layers
    int                             mNumInputs;     ///< Number of network inputs, not including bias term
    int                             mNumOutputs;    ///< Number of network outputs
    float                           mDecayRate;     ///< Weight decay rate, set to 1.0f to disable decay
    float                           mLearningRate;  ///< Learning rate
    float                           mMomentum;      ///< Momentum 
    float                           mWeightClamp;   ///< Valid range for weight values
    bool                            mBatchTraining; ///< True when a batch training session is in progress
    unique_ptr< gvec >              mCurrInput;     ///< Most recent network input
    unique_ptr< gvec >              mCurrTarget;    ///< Desired output for most recent input
    unique_ptr< gvec >              mAccumError;    ///< Accumulated output error
    vector< unique_ptr< Layer > >   mLayer;         ///< Weight layers (mNumLayers of them)

public:
    Backprop( const vector< int >& layerSizes ) :
        mNumLayers(     layerSizes.size() - 1 ),
        mLastLayer(     mNumLayers - 1 ),
        mDecayRate(     NN_DEFAULT_DECAY_RATE ),
        mLearningRate(  NN_DEFAULT_LEARNING_RATE ),
        mMomentum(      NN_DEFAULT_MOMENTUM ),
        mWeightClamp(   NN_DEFAULT_WEIGHT_CLAMP ),
        mBatchTraining( false )
    {
        for( int i = 0; i < mNumLayers; i++ )
        {
            int layerInputs  = layerSizes[i] + NN_BIAS_TERM;
            int layerOutputs = layerSizes[i + 1];

            mLayer.push_back( unique_ptr< Layer >( new Layer( layerInputs, layerOutputs ) ) );
            mLayer[i]->RandomizeWeights();
        }

        this->AllocBuffers();
    }



            void* Serialize( void* dest ) 
            {
                i32* header = (i32*) dest;

                *header++ = SERIALIZED_PERCEPTRON_TAG;
                *header++ = mNumLayers;
                *header++ = mNumOutputs;
                *header++ = mActivationFunc;
                *header++ = mNumInputs;
                *header++ = mNumOutputs;
                *header++ = mActivationFunc;
                        
                void* layerSer = header;

                for( int i = 0; i < mNumLayers; i++ )
                    layerSer = mLayer[i].Serialize( layerSer );
            }


            void* Deserialize( void* src, float* dataBlock )
            {
                i32* header = (i32*) src;

                if( *header++ != SERIALIZED_PERCEPTRON_TAG )
                    return( src );

                mNumInputs  = *header++;
                mNumOutputs = *header++;
                mActivation = *header++;

                this->SetSize( mNumInputs, mNumOutputs );
                this->SetData( dataBlock );

                float*  dataSer = (float*) header;
                float*  weight  = mData + mOfsWeight;

                for( int idx = 0; idx < mNumInputs * mNumOutputs; idx++ )
                    weight[idx] = *dataSer++;

                assert( ((intptr_t) dataSer) - ((intptr_t) src) == this->CalcSerializedSize() );
                return( dataSer );
            }

    Backprop( SerializedReader& reader )
    {
        mNumLayers      = reader.Read< int32_t >();
        mLastLayer      = mNumLayers - 1;
        mDecayRate      = reader.Read< float >();
        mLearningRate   = reader.Read< float >();
        mMomentum       = reader.Read< float >();
        mWeightClamp    = reader.Read< float >();

        for( int i = 0; i < mNumLayers; i++ )
            mLayer.push_back( unique_ptr< Layer >( new Layer( reader ) ) );

        this->AllocBuffers();
    }

    void AllocBuffers()
    {
        mNumInputs      = mLayer[0]->GetNumInputs() - NN_BIAS_TERM;
        mNumOutputs     = mLayer[mLastLayer]->GetNumOutputs();

        ASSERT( mNumInputs > 0 );
        ASSERT( mNumOutputs > 0 );

        mCurrInput      = std::move( unique_ptr< gvec >( new gvec( mNumInputs ) ) );
        mCurrTarget     = std::move( unique_ptr< gvec >( new gvec( mNumOutputs ) ) );
        mAccumError     = std::move( unique_ptr< gvec >( new gvec( mNumOutputs ) ) );
    
        mCurrInput->clear();
        mCurrTarget->clear();
        mAccumError->clear();
    }

    void Serialize( SerializedWriter& writer )
    {
        writer.Write< int32_t >( mNumLayers );
        writer.Write< float >( mDecayRate    );
        writer.Write< float >( mLearningRate );
        writer.Write< float >( mMomentum     );
        writer.Write< float >( mWeightClamp  );

        for( int i = 0; i < mNumLayers; i++ )
            mLayer[i]->Serialize( writer );
    }

    void Run( const float* input )
    {        
        for( int i = 0; i < mNumLayers; i++ )
        {
            mLayer[i].LoadInputs( (i == 0)? input : mLayer[i - 1].GetOutput() );
            mLayer[i].ApplyWeights();
            mLayer[i].ApplyActivation();
        }
    }

    void Train( const float* target )
    {
        bool onlineTraining = !mBatchTraining;
        if( onlineTraining )
            this->StartBatch();

        mCurrTarget->load( target, mNumOutputs );

        gvec_const_view goal    = mCurrTarget->view;
        gvec_const_view output  = mLayer[mLastLayer]->GetOutput();
        gvec_view       error   = mAccumError->view;

        parallel_for_each( error.extent, [=]( gvec_idx idx ) restrict( amp )
        {
            float act  = output[idx];
            float diff = goal[idx] - act;

            error[idx] += act * (1 - act) * diff;
        } );

        for( int i = mLastLayer; i >= 0; i-- )
            mLayer[i]->BackpropError( (i == mLastLayer)? mAccumError->view : mLayer[i + 1]->GetInputError() );

        if( onlineTraining )
            this->EndBatch();
    }

    void SyncReadOutput( float* dest )
    {
        gvec_const_view output = mLayer[mLastLayer]->GetOutput();

        for( int i = 0; i < mNumOutputs; i++ )
            dest[i] = output[i];
    }

    float SyncGetError( const float* target )
    {
        gvec_const_view output = mLayer[mLastLayer]->GetOutput();

        float acc = 0;
        for( int i = 0; i < mNumOutputs; i++ )
        {
            float err = target[i] - output[i];
            acc += (err * err);
        }

        return( acc );
    }

    void StartBatch()
    {
        ASSERT( !mBatchTraining );
        mBatchTraining = true;

        for( int i = 0; i < mNumLayers; i++ )
            mLayer[i]->ClearError();

        mAccumError->clear();
    }

    void EndBatch()
    {
        ASSERT( mBatchTraining );
        mBatchTraining = false;

        for( int i = 0; i < mNumLayers; i++ )
            mLayer[i]->UpdateWeights( mLearningRate, mDecayRate, mMomentum, mWeightClamp );
    }
};


#endif // PIGEON_NEURAL_H__

