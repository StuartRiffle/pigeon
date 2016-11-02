// gpu.h - PIGEON CHESS ENGINE (c) 2012-2016 Stuart Riffle

#ifndef PIGEON_GPU_H__
#define PIGEON_GPU_H__
namespace Pigeon {

#if PIGEON_ENABLE_CUDA


#define CUDA_REQUIRE( _CALL ) \
    if( (_CALL) != cudaSuccess ) \
    { \
        fprintf( stderr, "ERROR: failure in " #_CALL "\n" ); \
        return; \
    }


class CudaChessContext
{
    bool                        mInitialized;
    int                         mDeviceIndex;
    int                         mBatchCount;
    int                         mBatchSlots;
    std::vector< SearchBatch >  mBatches;
    cudaDeviceProp              mProp;
    std::vector< cudaStream_t > mStream;
    HashTable                   mHashTableHost;
    HashTable*                  mHashTableDev;
    void*                       mHashMemoryDev;
    Evaluator                   mEvaluatorHost;
    Evaluator*                  mEvaluatorDev;
    SearchJobInput*             mInputHost;
    SearchJobOutput*            mOutputHost;
    SearchJobInput*             mInputDev;
    SearchJobOutput*            mOutputDev;

    Mutex                       mMutex;
    int                         mStreamIndex;

    CudaChessContext()
    {
        PlatClearMemory( this, sizeof( *this ) );
    }

    ~CudaChessContext()
    {
        this->Shutdown();
    }

    void Initialize( int index, int batchCount, int batchSlots, size_t hashMegs )
    {
        mDeviceIndex    = index;
        mBatchCount     = batchCount;
        mBatchSlots     = batchSlots;

        CUDA_REQUIRE(( cudaSetDevice( mDeviceIndex ) ));
        CUDA_REQUIRE(( cudaGetDeviceProperties( &mProp, mDeviceIndex ) ));

        for( int i = 0; i < CUDA_STREAM_COUNT; i++ )
        {
            cudaStream_t stream;
            CUDA_REQUIRE(( cudaStreamCreateWithFlags( &stream, cudaStreamNonBlocking ) ));

            mStream.push_back( stream );
        }

        // Device hash table

        size_t hashMemSize = hashMegs * 1024 * 1024;

        CUDA_REQUIRE(( cudaMalloc( &mHashTableDev, sizeof( HashTable ) ) ));
        CUDA_REQUIRE(( cudaMalloc( &mHashMemoryDev, hashMemSize ) ));

        mHashTableHost.CalcTableEntries( hashMemSize );
        mHashTableHost.mTable = mHashMemoryDev;

        CUDA_REQUIRE(( cudaMemcpy( mHashTableDev, &mHashTableHost, sizeof( mHashTableHost ), cudaMemcpyHostToDevice ) ));
        CUDA_REQUIRE(( cudaMemset( mHashMemoryDev, 0, hashMemSize ) ));

        // Device evaluator

        mEvaluatorHost.SetDefaultWeights();

        CUDA_REQUIRE(( cudaMalloc( &mEvaluatorDev, sizeof( Evaluator ) ) ));
        CUDA_REQUIRE(( cudaMemcpy( mEvaluatorDev, &mEvaluatorHost, sizeof( mEvaluatorHost ), cudaMemcpyHostToDevice ) ));

        // I/O buffers

        size_t inputBufSize     = mBatchCount * mBatchSlots * sizeof( SearchJobInput );
        size_t outputBufSize    = mBatchCount * mBatchSlots * sizeof( SearchJobOutput );

        CUDA_REQUIRE(( cudaMallocHost( &mInputHost,  inputBufSize ) ));
        CUDA_REQUIRE(( cudaMallocHost( &mOutputHost, outputBufSize ) ));

        CUDA_REQUIRE(( cudaMalloc( &mInputDev,  inputBufSize ) ));
        CUDA_REQUIRE(( cudaMalloc( &mOutputDev, outputBufSize ) ));

        mBatches.resize( mBatchCount );
        for( int i = 0; i < mBatchCount; i++ )
        {
            SearchBatch* batch = &mBatches[i];
            int offset = i * mBatchSlots;

            batch->mInputHost   = mInputHost  + offset;
            batch->mInputDev    = mInputDev   + offset;
            batch->mOutputHost  = mOutputHost + offset;
            batch->mOutputDev   = mOutputDev  + offset;
            batch->mState       = BATCH_UNUSED;
            batch->mCount       = 0;
            batch->mLimit       = mBatchSlots;
        }

        mInitialized = true;
    }

    void Shutdown()
    {
        if( mHashTableDev )
            cudaFree( mHashTableDev );

        if( mHashMemoryDev )
            cudaFree( mHashMemoryDev );

        if( mEvaluatorDev )
            cudaFree( mEvaluatorDev );

        if( mInputHost )
            cudaFreeHost( mInputHost );

        if( mOutputHost )
            cudaFreeHost( mOutputHost );

        if( mInputDev )
            cudaFree( mInputDev );

        if( mOutputDev )
            cudaFree( mOutputDev );

        for( size_t i = 0; i < mStream.size(); i++ )
            cudaStreamDestroy( mStream[i] );

        mStream.clear();

        PlatClearMemory( this, sizeof( *this ) );
    }

    SearchBatch* AllocBatch()
    {
        Mutex::Scope scope( mMutex );
        cudaSetDevice( mDeviceIndex );

        for( size_t i = 0; i < mBatches.size(); i++ )
        {
            SearchBatch* batch = this->CycleToNextBatch();

            if( batch->mState == BATCH_UNUSED )
            {
                cudaError_t status = cudaEventCreateWithFlags( &batch->mEvent, cudaEventDisableTiming );
                assert( status == cudaSuccess );

                batch->mState   = BATCH_HOST_FILL;
                batch->mStream  = this->CycleToNextStream();
                batch->mCount   = 0;

                return( batch );
            }
        }

        return( NULL );
    }

    void SubmitBatch( SearchBatch* batch )
    {
        Mutex::Scope scope( mMutex );
        cudaSetDevice( mDeviceIndex );

        assert( batch->mState == BATCH_HOST_FILL );
        assert( batch->mCount <= batch->mLimit );

        if( batch->mCount > 0 )
        {
            int blockSize = mProp.warpSize;
            QueueSearchBatch( batch, blockSize );
        }

        batch->mState = BATCH_DEV_RUNNING;
    }


    SearchBatch* GetCompletedBatch()
    {
        Mutex::Scope scope( mMutex );
        cudaSetDevice( mDeviceIndex );

        for( size_t i = 0; i < mBatches.size(); i++ )
        {
            SearchBatch* batch = this->CycleToNextBatch();

            if( batch->mState == BATCH_DEV_RUNNING )
            {
                if( cudaEventQuery( batch->mEvent ) == cudaSuccess )
                {
                    cudaEventDestroy( batch->mEvent );
                    PlatMemoryFence();

                    batch->mState = BATCH_HOST_POST;
                    return( batch );
                }
            }
        }

        return( NULL );
    }

    void ReleaseBatch( SearchBatch* batch )
    {
        Mutex::Scope scope( mMutex );

        int idx = batch - &mBatches[0];

        assert( idx >= 0 );
        assert( idx < (int) mBatches.size() );

        batch->mState = BATCH_UNUSED;
        batch->mCount = 0;
    }


private:
    SearchBatch* CycleToNextBatch()
    {
        mBatchCursor = (mBatchCursor + 1) % mBatches.size();
        return( &mBatches[mBatchCursor] );
    }

    cudaStream_t CycleToNextStream()
    {
        mStreamIndex = (mStreamIndex + 1) % CUDA_STREAM_COUNT;
        return( mStream[mStreamIndex] );
    }

};

class CudaSystem
{
public:
    static int GetDeviceCount()
    {
        int count;
        if( cudaGetDeviceCount( &count ) != cudaSuccess )
            count = 0;

        return( count );
    }

    void QueuePosition( const Position& pos, int depth )
    {

    }
};

#endif // PIGEON_ENABLE_CUDA

};
#endif // PIGEON_GPU_H__
