// gpu.h - PIGEON CHESS ENGINE (c) 2012-2016 Stuart Riffle

#ifndef PIGEON_GPU_H__
#define PIGEON_GPU_H__

#if PIGEON_ENABLE_CUDA

enum
{
    BATCH_UNUSED = 0,
    BATCH_HOST_FILL,
    BATCH_DEV_RUNNING,
    BATCH_HOST_POST
};

struct SearchJobInput
{
	Position            mPosition;
	int                 mSearchDepth;
};

struct SearchJobOutput
{
	MoveList			mBestLine;
    u64                 mNodes;
	EvalTerm			mScore;
};

struct SearchBatch
{
    int                 mState;
    int                 mCount;
    int                 mLimit;
    cudaEvent_t         mEvent;
    cudaStream_t        mStream;
    SearchJobInput*     mInputHost;
    SearchJobInput*     mInputDev;
    SearchJobOutput*    mOutputHost;
    SearchJobOutput*    mOutputDev;
	HashTable*          mHashTable;
	Evaluator*          mEvaluator;
};


#if PIGEON_CUDA_HOST

#define CUDA_REQUIRE( _CALL ) \
    if( (_CALL) != cudaSuccess ) \
    { \
        fprintf( stderr, "ERROR: failure in " #_CALL "\n" ); \
        return; \
    }


class CudaChessContext
{
    Mutex                       mMutex;
    bool                        mInitialized;
    int                         mDeviceIndex;
    int                         mBatchCount;
    int                         mBatchSlots;
    int                         mBatchCursor;
    std::vector< SearchBatch >  mBatches;
    cudaDeviceProp              mProp;
    std::vector< cudaStream_t > mStream;
    int                         mStreamIndex;
    HashTable                   mHashTableHost;
    HashTable*                  mHashTableDev;
    void*                       mHashMemoryDev;
    Evaluator                   mEvaluatorHost;
    Evaluator*                  mEvaluatorDev;
    SearchJobInput*             mInputHost;
    SearchJobOutput*            mOutputHost;
    SearchJobInput*             mInputDev;
    SearchJobOutput*            mOutputDev;

public:
    CudaChessContext()
    {
        this->Clear();
    }

    ~CudaChessContext()
    {
        this->Shutdown();
    }

    void Clear()
    {
        mInitialized    = false;
        mDeviceIndex    = 0;
        mBatchCount     = 0;
        mBatchSlots     = 0;
        mBatchCursor    = 0;
        mStreamIndex    = 0;
        mHashTableDev   = NULL;
        mHashMemoryDev  = NULL;
        mEvaluatorDev   = NULL;
        mInputHost      = NULL;
        mOutputHost     = NULL;
        mInputDev       = NULL;
        mOutputDev      = NULL;

        mBatches.clear();
        mStream.clear();
        mHashTableHost.Clear();
        mEvaluatorHost.SetDefaultWeights();

        PlatClearMemory( &mProp, sizeof( mProp ) );
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

        CUDA_REQUIRE(( cudaMalloc( (void**) &mHashTableDev, sizeof( HashTable ) ) ));
        CUDA_REQUIRE(( cudaMalloc( (void**) &mHashMemoryDev, hashMemSize ) ));

        mHashTableHost.CalcTableEntries( hashMemSize );
        mHashTableHost.mTable = (u64*) mHashMemoryDev;

        CUDA_REQUIRE(( cudaMemcpy( mHashTableDev, &mHashTableHost, sizeof( mHashTableHost ), cudaMemcpyHostToDevice ) ));
        CUDA_REQUIRE(( cudaMemset( mHashMemoryDev, 0, hashMemSize ) ));

        // Device evaluator

        mEvaluatorHost.SetDefaultWeights();

        CUDA_REQUIRE(( cudaMalloc( (void**) &mEvaluatorDev, sizeof( Evaluator ) ) ));
        CUDA_REQUIRE(( cudaMemcpy( mEvaluatorDev, &mEvaluatorHost, sizeof( mEvaluatorHost ), cudaMemcpyHostToDevice ) ));

        // I/O buffers

        size_t inputBufSize     = mBatchCount * mBatchSlots * sizeof( SearchJobInput );
        size_t outputBufSize    = mBatchCount * mBatchSlots * sizeof( SearchJobOutput );

        CUDA_REQUIRE(( cudaMallocHost( (void**) &mInputHost,  inputBufSize ) ));
        CUDA_REQUIRE(( cudaMallocHost( (void**) &mOutputHost, outputBufSize ) ));

        CUDA_REQUIRE(( cudaMalloc( (void**) &mInputDev,  inputBufSize ) ));
        CUDA_REQUIRE(( cudaMalloc( (void**) &mOutputDev, outputBufSize ) ));

        mBatches.resize( mBatchCount );
        for( int i = 0; i < mBatchCount; i++ )
        {
            SearchBatch* batch = &mBatches[i];
            int offset = i * mBatchSlots;

            batch->mState       = BATCH_UNUSED;
            batch->mCount       = 0;
            batch->mLimit       = mBatchSlots;
            batch->mInputHost   = mInputHost  + offset;
            batch->mInputDev    = mInputDev   + offset;
            batch->mOutputHost  = mOutputHost + offset;
            batch->mOutputDev   = mOutputDev  + offset;
            batch->mHashTable   = mHashTableDev;
            batch->mEvaluator   = mEvaluatorDev;
        }

        int coresPerSM = 0;
        switch( mProp.major )
        {
        case 1:     coresPerSM = 8; break;
        case 2:     coresPerSM = (mProp.minor > 0)? 48 : 32; break;
        case 3:     coresPerSM = 192; break;
        case 5:     coresPerSM = 128; break;
        case 6:     coresPerSM = 64; break;
        case 7:     coresPerSM = 128; break;
        default:    break;
        }

        printf( "info string CUDA %d: %s (", mDeviceIndex, mProp.name );
        printf( "CC %d.%d, ", mProp.major, mProp.minor );
        if( coresPerSM > 0 )
            printf( "%d cores, ", mProp.multiProcessorCount * coresPerSM );
        printf( "%d mHz, ", mProp.clockRate / 1000 );
        printf( "%d MB)\n", mProp.totalGlobalMem / (1024 * 1024) );

        mInitialized = true;
    }

    bool IsInitialized() const
    {
        return( mInitialized );
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

        this->Clear();
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
            extern void QueueSearchBatch( SearchBatch* batch, int blockSize );
            
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
};

#endif // PIGEON_CUDA_HOST
#endif // PIGEON_ENABLE_CUDA
#endif // PIGEON_GPU_H__
