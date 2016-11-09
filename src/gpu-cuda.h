// gpu.h - PIGEON CHESS ENGINE (c) 2012-2016 Stuart Riffle

#ifndef PIGEON_GPU_H__
#define PIGEON_GPU_H__

// This file is #included from search.h, and we're already inside namespace Pigeon

#if PIGEON_ENABLE_CUDA

enum
{
    BATCH_UNUSED = 0,
    BATCH_HOST_FILL,
    BATCH_DEV_RUNNING,
    BATCH_HOST_POST
};

struct PIGEON_ALIGN( 32 ) SearchJobInput
{
	Position            mPosition;
    MoveMap             mMoveMap;
    int                 mPly;
	int                 mDepth;
    EvalTerm            mAlpha;
    EvalTerm            mBeta;
    EvalTerm            mScore;
    MoveSpec            mPath[MAX_SEARCH_DEPTH];
    u64                 mTick;
};

struct PIGEON_ALIGN( 32 ) SearchJobOutput
{
    u64                 mNodes;
    u64                 mSteps;
	EvalTerm			mScore;
    int                 mDeepestPly;
    MoveSpec            mPath[MAX_SEARCH_DEPTH];
};

struct PIGEON_ALIGN( 32 ) SearchBatch
{
    int                 mState;                     /// Batch state (from BATCH_* enum)
    int                 mCount;                     /// Jobs in the batch
    int                 mLimit;                     /// Maximum jobs in the batch (input/output buffer sizes)
    cudaStream_t        mStream;                    /// Stream this batch was issued into
    SearchJobInput*     mInputHost;                 /// Job input buffer, host side
    SearchJobInput*     mInputDev;                  /// Job input buffer, device side
    SearchJobOutput*    mOutputHost;                /// Job output buffer, host side
    SearchJobOutput*    mOutputDev;                 /// Job output buffer, device side
	HashTable*          mHashTable;                 /// Device hash table structure
	Evaluator*          mEvaluator;                 /// Device evaluator structure (blending weights)

    u64                 mTickQueued;                /// CPU tick when the batch was queued for execution
    u64                 mTickReturned;              /// CPU tick when the completed batch was found
    float               mCpuLatency;                /// CPU time elapsed (in ms) between those two ticks, represents batch processing latency

    cudaEvent_t         mStartEvent;                /// GPU timer event to mark the start of kernel execution
    cudaEvent_t         mEndEvent;                  /// GPU timer event to mark the end of kernel execution
    cudaEvent_t         mReadyEvent;                /// GPU timer event after the results have been copied back to host memory
    float               mGpuTime;                   /// GPU time spent executing kernel (in ms)
};


#if PIGEON_CUDA_HOST

#define CUDA_REQUIRE( _CALL ) \
    if( (_CALL) != cudaSuccess ) \
    { \
        fprintf( stderr, "ERROR: failure in " #_CALL "\n" ); \
        return; \
    }

/// Cuda device manager for search jobs
/// 
class CudaChessContext
{
    Mutex                       mMutex;             /// Mutex to serialize access
    bool                        mInitialized;       /// True when the CUDA device has been successfully set up
    int                         mDeviceIndex;       /// CUDA device index
    cudaDeviceProp              mProp;              /// CUDA device properties
    int                         mBlockWarps;        /// The block size is mBlockWarps * the warp size
    int                         mBatchCount;        /// Number of batch buffers
    int                         mBatchSlots;        /// Number of jobs per batch
    int                         mBatchCursor;       /// FIXME
    std::vector< cudaStream_t > mStreamId;          /// A list of available execution streams, used round-robin
    std::vector< SearchBatch >  mBatches;           /// All the search batch structures; queues etc point into this
    std::queue< SearchBatch* >  mRunningBatches;    /// Batches that have been submitted
    std::queue< SearchBatch* >  mDoneBatches;       /// Completed batches ready for post-processing
    std::stack< SearchBatch* >  mFreeBatches;       /// Batches that are available to allocate
    int                         mStreamIndex;       /// Index of the next stream to be used for submitting a batch
    HashTable                   mHashTableHost;     /// Host-side copy of the device's hash table object
    HashTable*                  mHashTableDev;      /// Device hash table structure
    void*                       mHashMemoryDev;     /// Raw memory used by the hash table
    Evaluator                   mEvaluatorHost;     /// Host-side copy of the device's evaluator object
    Evaluator*                  mEvaluatorDev;      /// Device evaluator object
    i32*                        mExitFlagHost;      /// Hist-side flag to trigger early exit
    i32*                        mExitFlagDev;       /// Device-side flag to trigger early exit, set via DMA
    cudaStream_t                mExitFlagStream;    /// A special stream to allow the exit flag to be updated while kernels are running
    SearchJobInput*             mInputHost;         /// Host-side job input buffer (each batch uses a slice)
    SearchJobOutput*            mOutputHost;        /// Host-side job output buffer (each batch uses a slice)
    SearchJobInput*             mInputDev;          /// Device-side job input buffer (each batch uses a slice)
    SearchJobOutput*            mOutputDev;         /// Device-side job output buffer (each batch uses a slice)

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
        mBlockWarps     = 4;
        mBatchCount     = 0;
        mBatchSlots     = 0;
        mBatchCursor    = 0;
        mStreamIndex    = 0;
        mHashTableDev   = NULL;
        mHashMemoryDev  = NULL;
        mEvaluatorDev   = NULL;
        mExitFlagHost   = NULL;
        mExitFlagDev    = NULL;
        mInputHost      = NULL;
        mOutputHost     = NULL;
        mInputDev       = NULL;
        mOutputDev      = NULL;
        mExitFlagStream = (cudaStream_t) 0;

        mBatches.clear();
        mStreamId.clear();
        mHashTableHost.Clear();
        mEvaluatorHost.SetDefaultWeights();

        while( !mRunningBatches.empty() )
            mDoneBatches.pop();

        while( !mDoneBatches.empty() )
            mDoneBatches.pop();

        while( !mFreeBatches.empty() )
            mFreeBatches.pop();

        PlatClearMemory( &mProp, sizeof( mProp ) );
    }

    void Initialize( int index, int batchCount, int batchSlots, size_t hashMegs )
    {
        mDeviceIndex    = index;
        mBatchCount     = batchCount;
        mBatchSlots     = batchSlots;

        CUDA_REQUIRE(( cudaSetDevice( mDeviceIndex ) ));
        CUDA_REQUIRE(( cudaGetDeviceProperties( &mProp, mDeviceIndex ) ));

        int leastPrio    = 0;
        int greatestPrio = 0;
        CUDA_REQUIRE(( cudaDeviceGetStreamPriorityRange( &leastPrio, &greatestPrio ) ));

        for( int i = 0; i < CUDA_STREAM_COUNT; i++ )
        {
            cudaStream_t stream;
            //CUDA_REQUIRE(( cudaStreamCreateWithPriority( &stream, cudaStreamNonBlocking, greatestPrio + 1 ) ));
            CUDA_REQUIRE(( cudaStreamCreateWithFlags( &stream, cudaStreamNonBlocking ) ));

            mStreamId.push_back( stream );
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

        // Early-exit flag

        CUDA_REQUIRE(( cudaMallocHost( (void**) &mExitFlagHost, sizeof( *mExitFlagHost ) ) ));
        *mExitFlagHost = 0;

        CUDA_REQUIRE(( cudaMalloc( (void**) &mExitFlagDev, sizeof( *mExitFlagDev ) ) ));
        CUDA_REQUIRE(( cudaMemcpy( mExitFlagDev, mExitFlagHost, sizeof( *mExitFlagHost ), cudaMemcpyHostToDevice ) ));

        //CUDA_REQUIRE(( cudaStreamCreateWithPriority( &mExitFlagStream, cudaStreamNonBlocking, greatestPrio ) ));
        CUDA_REQUIRE(( cudaStreamCreateWithFlags( &mExitFlagStream, cudaStreamNonBlocking ) ));

        // I/O buffers

        size_t inputBufSize     = mBatchCount * mBatchSlots * sizeof( SearchJobInput );
        size_t outputBufSize    = mBatchCount * mBatchSlots * sizeof( SearchJobOutput );

        CUDA_REQUIRE(( cudaMallocHost( (void**) &mInputHost,  inputBufSize ) ));
        CUDA_REQUIRE(( cudaMallocHost( (void**) &mOutputHost, outputBufSize ) ));

        CUDA_REQUIRE(( cudaMalloc( (void**) &mInputDev,  inputBufSize ) ));
        CUDA_REQUIRE(( cudaMalloc( (void**) &mOutputDev, outputBufSize ) ));

        mBatches.reserve( mBatchCount );

        for( int i = 0; i < mBatchCount; i++ )
        {
            SearchBatch batch;
            int         offset = i * mBatchSlots;

            batch.mState       = BATCH_UNUSED;
            batch.mCount       = 0;
            batch.mLimit       = mBatchSlots;
            batch.mInputHost   = mInputHost  + offset;
            batch.mInputDev    = mInputDev   + offset;
            batch.mOutputHost  = mOutputHost + offset;
            batch.mOutputDev   = mOutputDev  + offset;
            batch.mHashTable   = mHashTableDev;
            batch.mEvaluator   = mEvaluatorDev;

            CUDA_REQUIRE(( cudaEventCreate( &batch.mStartEvent ) ));
            CUDA_REQUIRE(( cudaEventCreate( &batch.mEndEvent ) ));
            CUDA_REQUIRE(( cudaEventCreateWithFlags( &batch.mReadyEvent, cudaEventDisableTiming ) ));

            mBatches.push_back( batch );
            mFreeBatches.push( &mBatches.back() );
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
        else
            printf( "%d SM, ", mProp.multiProcessorCount );

        printf( "%d MHz, ", mProp.clockRate / 1000 );
        printf( "%d MB)\n", mProp.totalGlobalMem / (1024 * 1024) );

        CUDA_REQUIRE(( cudaDeviceSetCacheConfig( cudaFuncCachePreferL1 ) ));

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

        if( mExitFlagDev )
            cudaFree( mExitFlagDev );

        if( mInputHost )
            cudaFreeHost( mInputHost );

        if( mOutputHost )
            cudaFreeHost( mOutputHost );

        if( mInputDev )
            cudaFree( mInputDev );

        if( mOutputDev )
            cudaFree( mOutputDev );

        if( mExitFlagHost )
            cudaFreeHost( mExitFlagHost );

        if( mExitFlagStream )
            cudaStreamDestroy( mExitFlagStream );

        for( size_t i = 0; i < mStreamId.size(); i++ )
            cudaStreamDestroy( mStreamId[i] );

        for( int i = 0; i < mBatches.size(); i++ )
        {
            SearchBatch* batch = &mBatches[i];

            cudaEventDestroy( batch->mStartEvent );
            cudaEventDestroy( batch->mEndEvent );
            cudaEventDestroy( batch->mReadyEvent );
        }

        this->Clear();
    }

    void SetBlockWarps( int blockWarps )
    {
        Mutex::Scope scope( mMutex );

        assert( blockWarps > 0 );
        assert( IsPowerOfTwo( blockWarps ) );

        mBlockWarps = blockWarps;
    }

    SearchBatch* AllocBatch()
    {
        Mutex::Scope scope( mMutex );

        if( mFreeBatches.empty() )
            return( NULL );

        SearchBatch* batch = mFreeBatches.top();
        mFreeBatches.pop();

        assert( batch->mState == BATCH_UNUSED );

        batch->mState   = BATCH_HOST_FILL;
        batch->mStream  = (cudaStream_t) 0;
        batch->mCount   = 0;

        return( batch );
    }


    void SubmitBatch( SearchBatch* batch )
    {
        Mutex::Scope scope( mMutex );

        assert( batch->mState == BATCH_HOST_FILL );
        assert( batch->mCount <= batch->mLimit );

        if( batch->mCount < 1 )
        {
            this->ReleaseBatch( batch );
            return;
        }

        mStreamIndex = (mStreamIndex + 1) % mStreamId.size();

        batch->mStream = mStreamId[mStreamIndex];

        extern void QueueSearchBatch( SearchBatch* batch, int blockCount, int blockSize, i32* exitFlag );

        int blockSize  = mProp.warpSize * mBlockWarps;
        int blockCount = (batch->mCount + blockSize - 1) / blockSize;

        cudaSetDevice( mDeviceIndex );
        QueueSearchBatch( batch, blockCount, blockSize, mExitFlagDev );

        batch->mTickQueued  = Timer::GetTick();
        batch->mState       = BATCH_DEV_RUNNING;

        mRunningBatches.push( batch );


        //cudaDeviceSynchronize();
    }

    SearchBatch* GetCompletedBatch()
    {
        Mutex::Scope scope( mMutex );

        if( mDoneBatches.empty() )
            this->GatherCompletedBatches();

        if( mDoneBatches.empty() )
            return( NULL );

        SearchBatch* batch = mDoneBatches.front();
        mDoneBatches.pop();

        assert( batch->mState == BATCH_HOST_POST );
        return( batch );
    }

    void ReleaseBatch( SearchBatch* batch )
    {
        Mutex::Scope scope( mMutex );

        int idx = (int) (batch - &mBatches[0]);

        assert( idx >= 0 );
        assert( idx < (int) mBatches.size() );

        batch->mState = BATCH_UNUSED;
        batch->mCount = 0;

        mFreeBatches.push( batch );
    }

    void CancelAllBatchesSync()
    {
        Mutex::Scope scope( mMutex );
        cudaSetDevice( mDeviceIndex );

        this->SetExitFlagSync( 1 );

        this->GatherCompletedBatches();
        while( !mDoneBatches.empty() )
        {
            SearchBatch* batch = mDoneBatches.front();
            mDoneBatches.pop();

            this->ReleaseBatch( batch );
        }

        this->SetExitFlagSync( 0 );

        assert( mDoneBatches.empty() );
        assert( mRunningBatches.empty() );
        assert( mFreeBatches.size() == mBatches.size() );
    }


private:
    int CycleToNextStream()
    {
        assert( mStreamId.size() > 0 );

        mStreamIndex = (mStreamIndex + 1) % mStreamId.size();
        return( mStreamIndex );
    }

    void GatherCompletedBatches()
    {
        cudaSetDevice( mDeviceIndex ); // Is this required before cudaEventQuery()?

        while( !mRunningBatches.empty() )
        {
            SearchBatch* batch = mRunningBatches.front();
            assert( batch->mState == BATCH_DEV_RUNNING );

            if( cudaEventQuery( batch->mReadyEvent ) != cudaSuccess )
                break;

            mRunningBatches.pop();

            batch->mTickReturned    = Timer::GetTick();
            batch->mCpuLatency      = (batch->mTickReturned - batch->mTickQueued) * 1000.0f / Timer::GetFrequency();
            batch->mState           = BATCH_HOST_POST;

            cudaEventElapsedTime( &batch->mGpuTime, batch->mStartEvent, batch->mEndEvent );

            mDoneBatches.push( batch );
        }
    }

    void SetExitFlagSync( i32 val )
    {
        *mExitFlagHost = val;
        cudaMemcpyAsync( mExitFlagDev, mExitFlagHost, sizeof( *mExitFlagHost ), cudaMemcpyHostToDevice, mExitFlagStream );

        cudaDeviceSynchronize();
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

