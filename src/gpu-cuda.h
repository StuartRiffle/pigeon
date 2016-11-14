// gpu.h - PIGEON CHESS ENGINE (c) 2012-2016 Stuart Riffle

#ifndef PIGEON_GPU_H__
#define PIGEON_GPU_H__

// This file is #included from search.h, and we're already inside namespace Pigeon

#if PIGEON_ENABLE_CUDA


//==============================================================================
/// Async batch job state

enum
{
    BATCH_UNUSED = 0,                               /// Unused batch
    BATCH_HOST_FILL,                                /// The host-side code is adding jobs to the batch
    BATCH_DEV_RUNNING,                              /// The batch has been submitted to the device
    BATCH_HOST_POST                                 /// The host-side code is processing completed jobs
};


//==============================================================================
/// Search job input

struct PIGEON_ALIGN( 32 ) SearchJobInput
{
	Position            mPosition;                  ///
    MoveMap             mMoveMap;                   ///
    int                 mPly;                       ///
	int                 mDepth;                     ///
    EvalTerm            mAlpha;                     ///
    EvalTerm            mBeta;                      ///
    EvalTerm            mScore;                     ///
    MoveSpec            mPath[MAX_SEARCH_DEPTH];    ///
    u64                 mTick;                      ///
};


//==============================================================================
/// Search job output

struct PIGEON_ALIGN( 32 ) SearchJobOutput
{
    u64                 mNodes;                     ///
    u64                 mSteps;                     ///
	EvalTerm			mScore;                     ///
    int                 mDeepestPly;                ///
    MoveSpec            mPath[MAX_SEARCH_DEPTH];    ///
};


//==============================================================================
/// Search batch
///
/// A batch is a bunch of async search jobs

struct PIGEON_ALIGN( 32 ) SearchBatch
{
    IAsyncSearcher*     mDevice;                    /// The device this batch was allocated from
    int                 mState;                     /// Batch state (from BATCH_* enum)
    int                 mCount;                     /// Jobs in the batch
    int                 mLimit;                     /// Maximum jobs in the batch (input/output buffer sizes)
    cudaStream_t        mStream;                    /// Stream this batch was issued into
    SearchJobInput*     mInputHost;                 /// Job input buffer, host side
    SearchJobInput*     mInputDev;                  /// Job input buffer, device side
    SearchJobOutput*    mOutputHost;                /// Job output buffer, host side
    SearchJobOutput*    mOutputDev;                 /// Job output buffer, device side
	HashTable*          mHashTableDev;              /// Device hash table structure
	Evaluator*          mEvaluatorDev;              /// Device evaluator structure (blending weights)
    int*                mOptionsDev;                /// Device option settings
    u64                 mTickQueued;                /// CPU tick when the batch was queued for execution
    u64                 mTickReturned;              /// CPU tick when the completed batch was found
    float               mCpuLatency;                /// CPU time elapsed (in ms) between those two ticks, represents batch processing latency
    cudaEvent_t         mStartEvent;                /// GPU timer event to mark the start of kernel execution
    cudaEvent_t         mEndEvent;                  /// GPU timer event to mark the end of kernel execution
    cudaEvent_t         mReadyEvent;                /// GPU timer event to notify that the results have been copied back to host memory
    float               mGpuTime;                   /// GPU time spent executing kernel (in ms)
};


#if PIGEON_CUDA_HOST

#define CUDA_REQUIRE( _CALL ) \
{ \
    cudaError_t status = (_CALL); \
    if( status != cudaSuccess ) \
    { \
        fprintf( stderr, "ERROR: failure in " #_CALL " [%d]\n", status ); \
        return; \
    } \
}



//==============================================================================
/// Cuda device manager for search jobs

class CudaChessDevice : public IAsyncSearcher
{
    Mutex                       mMutex;             /// Mutex to serialize access
    bool                        mInitialized;       /// True when the CUDA device has been successfully set up
    int                         mDeviceIndex;       /// CUDA device index
    cudaDeviceProp              mProp;              /// CUDA device properties

    int                         mBlockWarps;        /// The block size is mBlockWarps * the warp size
    int                         mBatchCount;        /// Number of batch buffers
    int                         mBatchSlots;        /// Number of jobs per batch

    std::vector< cudaStream_t > mStreamId;          /// A list of available execution streams, used round-robin
    std::vector< SearchBatch >  mBatches;           /// All the search batch structures; queues etc point into this

    std::queue< SearchBatch* >  mPendingBatches;
    std::queue< SearchBatch* >  mRunningBatches;    /// Batches that have been submitted
    std::queue< SearchBatch* >  mDoneBatches;       /// Completed batches ready for post-processing
    std::stack< SearchBatch* >  mFreeBatches;       /// Batches that are available to allocate
    int                         mStreamIndex;       /// Index of the next stream to be used for submitting a batch

    HashTable                   mHashTableHost;     /// Host-side copy of the device's hash table object
    HashTable*                  mHashTableDev;      /// Device hash table structure
    void*                       mHashMemoryDev;     /// Raw memory used by the hash table

    Evaluator                   mEvaluatorHost;     /// Host-side copy of the device's evaluator object
    Evaluator*                  mEvaluatorDev;      /// Device evaluator object

    i32*                        mExitFlagHost;      /// Host-side flag to trigger early exit
    i32*                        mExitFlagDev;       /// Device-side flag to trigger early exit, set via DMA
    cudaStream_t                mExitFlagStream;    /// A special stream to allow the exit flag to be updated while kernels are running

    i32*                        mOptionsHost;       /// Host-side option array
    i32*                        mOptionsDev;        /// Device-side copy of the options

    SearchJobInput*             mInputHost;         /// Host-side job input buffer (each batch uses a slice)
    SearchJobOutput*            mOutputHost;        /// Host-side job output buffer (each batch uses a slice)
    SearchJobInput*             mInputDev;          /// Device-side job input buffer (each batch uses a slice)
    SearchJobOutput*            mOutputDev;         /// Device-side job output buffer (each batch uses a slice)

    Semaphore                   mSemaInitialized;
    Semaphore                   mSemaPending;
    Semaphore                   mThreadsDone;


//private:
//    CudaChessDevice( const CudaChessDevice& device ) {}
//    CudaChessDevice& operator=( const CudaChessDevice& device ) { return( *this ); }
//
public:
    CudaChessDevice()
    {
        this->Clear();
    }

    ~CudaChessDevice()
    {
        this->Shutdown();
    }



    //--------------------------------------------------------------------------
    /// Clear the object state

    void Clear()
    {
        mInitialized    = false;
        mDeviceIndex    = 0;
        mBlockWarps     = 4;
        mBatchCount     = 0;
        mBatchSlots     = 0;
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
        mOptionsHost    = NULL;
        mOptionsDev     = NULL;

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


    //--------------------------------------------------------------------------
    ///

    void Initialize( int index, int batchCount, int batchSlots, size_t hashBytes, i32* options )
    {
        mDeviceIndex    = index;
        mBatchCount     = batchCount;
        mBatchSlots     = batchSlots;
        mOptionsHost    = options;

        mHashTableHost.CalcTableEntries( hashBytes );

        PlatSpawnThread( &CudaThreadProc, this );

        // Block until initialization has completed (or failed)

        mSemaInitialized.Wait();
    }

    void InitCuda()
    {
        CUDA_REQUIRE(( cudaSetDevice( mDeviceIndex ) ));
        CUDA_REQUIRE(( cudaGetDeviceProperties( &mProp, mDeviceIndex ) ));

        for( int i = 0; i < CUDA_STREAM_COUNT; i++ )
        {
            cudaStream_t stream;
            CUDA_REQUIRE(( cudaStreamCreateWithFlags( &stream, cudaStreamNonBlocking ) ));

            mStreamId.push_back( stream );
        }

        // Device hash table

        size_t hashTableBytes = mHashTableHost.mEntries * sizeof( u64 );

        CUDA_REQUIRE(( cudaMalloc( (void**) &mHashTableDev, sizeof( HashTable ) ) ));
        CUDA_REQUIRE(( cudaMalloc( (void**) &mHashMemoryDev, hashTableBytes ) ));

        mHashTableHost.mTable = (u64*) mHashMemoryDev;

        CUDA_REQUIRE(( cudaMemcpy( mHashTableDev, &mHashTableHost, sizeof( HashTable ), cudaMemcpyHostToDevice ) ));
        CUDA_REQUIRE(( cudaMemset( mHashMemoryDev, 0, hashTableBytes ) ));

        // Device evaluator

        mEvaluatorHost.SetDefaultWeights();

        CUDA_REQUIRE(( cudaMalloc( (void**) &mEvaluatorDev, sizeof( Evaluator ) ) ));
        CUDA_REQUIRE(( cudaMemcpy( mEvaluatorDev, &mEvaluatorHost, sizeof( mEvaluatorHost ), cudaMemcpyHostToDevice ) ));

        // Options

        CUDA_REQUIRE(( cudaMalloc( (void**) &mOptionsDev, sizeof( i32 ) * OPTION_COUNT ) ));
        CUDA_REQUIRE(( cudaMemcpy( mOptionsDev, mOptionsHost, sizeof( i32 ) * OPTION_COUNT, cudaMemcpyHostToDevice ) ));

        // Early-exit flag

        CUDA_REQUIRE(( cudaMallocHost( (void**) &mExitFlagHost, sizeof( *mExitFlagHost ) ) ));
        *mExitFlagHost = 0;

        CUDA_REQUIRE(( cudaMalloc( (void**) &mExitFlagDev, sizeof( *mExitFlagDev ) ) ));
        CUDA_REQUIRE(( cudaMemcpy( mExitFlagDev, mExitFlagHost, sizeof( *mExitFlagHost ), cudaMemcpyHostToDevice ) ));

        CUDA_REQUIRE(( cudaStreamCreateWithFlags( &mExitFlagStream, cudaStreamNonBlocking ) ));

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
            SearchBatch& batch  = mBatches[i];
            int          offset = i * mBatchSlots;

            batch.mDevice       = this;
            batch.mState        = BATCH_UNUSED;
            batch.mCount        = 0;
            batch.mLimit        = mBatchSlots;
            batch.mStream       = (cudaStream_t) 0;
            batch.mInputHost    = mInputHost  + offset;
            batch.mInputDev     = mInputDev   + offset;
            batch.mOutputHost   = mOutputHost + offset;
            batch.mOutputDev    = mOutputDev  + offset;
            batch.mHashTableDev = mHashTableDev;
            batch.mEvaluatorDev = mEvaluatorDev;
            batch.mOptionsDev   = mOptionsDev;
            batch.mTickQueued   = 0;
            batch.mTickReturned = 0;
            batch.mCpuLatency   = 0;
            batch.mGpuTime      = 0;

            CUDA_REQUIRE(( cudaEventCreate( &batch.mStartEvent ) ));
            CUDA_REQUIRE(( cudaEventCreate( &batch.mEndEvent ) ));
            CUDA_REQUIRE(( cudaEventCreateWithFlags( &batch.mReadyEvent, cudaEventDisableTiming ) ));

            mFreeBatches.push( &batch );
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


    //--------------------------------------------------------------------------
    ///

    bool IsInitialized() const
    {
        return( mInitialized );
    }


    //--------------------------------------------------------------------------
    ///

    void Shutdown()
    {
        if( mInitialized )
        {
            this->CancelAllBatchesSync();

            this->SendToSubmissionThread( NULL );
            mThreadsDone.Wait();
        }

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

        if( mOptionsDev )
            cudaFree( mOptionsDev );

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


    //--------------------------------------------------------------------------
    ///

    void SetBlockWarps( int blockWarps )
    {
        Mutex::Scope scope( mMutex );

        assert( blockWarps > 0 );
        assert( IsPowerOfTwo( blockWarps ) );

        mBlockWarps = blockWarps;
    }


    //--------------------------------------------------------------------------
    ///

    virtual SearchBatch* AllocBatch()
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



    void SendToSubmissionThread( SearchBatch* batch )
    {
        Mutex::Scope scope( mMutex );

        mPendingBatches.push( batch );
        mSemaPending.Post();
    }

    //--------------------------------------------------------------------------
    ///

    virtual void SubmitBatch( SearchBatch* batch )
    {
        assert( batch->mState == BATCH_HOST_FILL );
        assert( batch->mCount <= batch->mLimit );

        if( batch->mCount > 0 )
        {
            this->SendToSubmissionThread( batch );
        }
        else
        {
            this->ReleaseBatch( batch );
        }
    }


    //--------------------------------------------------------------------------
    ///

    virtual SearchBatch* GetCompletedBatch()
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


    //--------------------------------------------------------------------------
    ///

    virtual void ReleaseBatch( SearchBatch* batch )
    {
        Mutex::Scope scope( mMutex );

        int idx = (int) (batch - &mBatches[0]);

        assert( idx >= 0 );
        assert( idx < (int) mBatches.size() );

        batch->mState = BATCH_UNUSED;
        batch->mCount = 0;

        mFreeBatches.push( batch );
    }


    //--------------------------------------------------------------------------
    ///

    virtual void CancelAllBatchesSync()
    {
        this->SetExitFlagSync( 1 );
        {
            Mutex::Scope scope( mMutex );

            assert( mRunningBatches.empty() );

            this->GatherCompletedBatches();
            while( !mDoneBatches.empty() )
            {
                SearchBatch* batch = mDoneBatches.front();
                mDoneBatches.pop();

                this->ReleaseBatch( batch );
            }

            assert( mDoneBatches.empty() );
            assert( mFreeBatches.size() == mBatches.size() );
        }
        this->SetExitFlagSync( 0 );
    }


private:

    //--------------------------------------------------------------------------
    ///

    int CycleToNextStream()
    {
        assert( mStreamId.size() > 0 );

        mStreamIndex = (mStreamIndex + 1) % mStreamId.size();
        return( mStreamIndex );
    }


    //--------------------------------------------------------------------------
    ///

    void GatherCompletedBatches()
    {
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


    //--------------------------------------------------------------------------
    ///

    void SetExitFlagSync( i32 val )
    {
        *mExitFlagHost = val;
        cudaMemcpyAsync( mExitFlagDev, mExitFlagHost, sizeof( *mExitFlagHost ), cudaMemcpyHostToDevice, mExitFlagStream );

        cudaDeviceSynchronize();
    }

private:


    SearchBatch* ClaimPending()
    {
        Mutex::Scope scope( mMutex );
        assert( !mPendingBatches.empty() );

        SearchBatch* batch = mPendingBatches.front();
        mPendingBatches.pop();

        if( batch )
        {
            mStreamIndex = (mStreamIndex + 1) % mStreamId.size();
            batch->mStream = mStreamId[mStreamIndex];
        }

        return( batch );
    }

    void MarkAsRunning( SearchBatch* batch )
    {
        Mutex::Scope scope( mMutex );

        batch->mState = BATCH_DEV_RUNNING;
        mRunningBatches.push( batch );
    }

    void SubmissionThread()
    {
        this->InitCuda();

        int     warpSize    = mProp.warpSize;
        int     blockWarps  = mBlockWarps;
        i32*    exitFlag    = mExitFlagDev;

        mSemaInitialized.Post();

        if( !mInitialized )
            return;

        for( ;; )
        {
            mSemaPending.Wait();

            SearchBatch* batch = this->ClaimPending();
            if( batch == NULL )
                break;

            int blockSize = warpSize * blockWarps;
            while( (blockSize >> 1) >= batch->mCount )
                blockSize >>= 1;

            int blockCount = (batch->mCount + blockSize - 1) / blockSize;

            batch->mOutputHost->mNodes = 42;
            extern void QueueSearchBatch( SearchBatch* batch, int blockCount, int blockSize, i32* exitFlag );

            QueueSearchBatch( batch, blockCount, blockSize, exitFlag );
            batch->mTickQueued = Timer::GetTick();

            this->MarkAsRunning( batch );
        }
    }

    static void* CudaThreadProc( void* param )
    {
        CudaChessDevice* device = reinterpret_cast< CudaChessDevice* >( param );

        char threadName[80];
        sprintf( threadName, "CUDA %d", device->mDeviceIndex );
        PlatSetThreadName( threadName );

        device->SubmissionThread();
        device->mThreadsDone.Post();

        return( NULL );
    }

};


//==============================================================================
///

class CudaSystem
{
public:

    //--------------------------------------------------------------------------
    ///

    static int GetDeviceCount()
    {
        int count;
        if( cudaGetDeviceCount( &count ) != cudaSuccess )
            count = 0;

        return( count );
    }
};


class CudaChessManager : public IAsyncSearcher
{
    typedef std::shared_ptr< CudaChessDevice > CudaChessDevicePtr;

    Mutex                               mMutex;
    std::vector< CudaChessDevicePtr >   mDevices;
    bool                                mInitialized;
    int                                 mAllocIndex;
    int                                 mPollIndex;

public:
    CudaChessManager() :
        mInitialized( false ),
        mAllocIndex( 0 ),
        mPollIndex( 0 )
    {
    }

    void Initialize( int* options )
    {
        int deviceCount = CudaSystem::GetDeviceCount();

        for( int i = 0; i < deviceCount; i++ )
        {
            cudaDeviceProp deviceProp = { 0 };
            int deviceIndex = i;

            cudaSetDevice( deviceIndex ); 
            if( cudaGetDeviceProperties( &deviceProp, deviceIndex ) != cudaSuccess )
                continue;

            int     batchCount  = BATCH_COUNT_DEFAULT;
            int     batchSize   = BATCH_SIZE_DEFAULT;
            size_t  hashBytes   = 1 * 1024 * 1024;

            // By default use half of the device memory for hash table

            while( 4 * hashBytes < deviceProp.totalGlobalMem )
                hashBytes *= 2;

            // TODO: use the device properties to tweak these settings 

            CudaChessDevicePtr device( new CudaChessDevice() );

            device->Initialize( deviceIndex, batchCount, batchSize, hashBytes, options );
            if( !device->IsInitialized() )
                continue;

            mDevices.push_back( device );
        }

        mInitialized = !mDevices.empty();
    }

    bool IsInitialized() const
    {
        return( mInitialized );
    }

    void SetBlockWarps( int blockWarps )
    {
        Mutex::Scope scope( mMutex );

        for( size_t i = 0; i < mDevices.size(); i++ )
            mDevices[i]->SetBlockWarps( blockWarps );
    }


    virtual SearchBatch* AllocBatch()
    {
        Mutex::Scope scope( mMutex );

        mAllocIndex++;
        if( mAllocIndex >= mDevices.size() )
            mAllocIndex = 0;

        int idx = mAllocIndex;

        for( size_t i = 0; i < mDevices.size(); i++ )
        {
            SearchBatch* batch = mDevices[idx]->AllocBatch();
            if( batch != NULL )
                return( batch );

            idx++;
            if( idx >= mDevices.size() )
                idx = 0;
        }

        return( NULL );
    }

    virtual void SubmitBatch( SearchBatch* batch )
    {
        assert( batch != NULL );
        assert( batch->mDevice != NULL );

        batch->mDevice->SubmitBatch( batch );
    }

    virtual SearchBatch* GetCompletedBatch()
    {
        Mutex::Scope scope( mMutex );

        mPollIndex++;
        if( mPollIndex >= mDevices.size() )
            mPollIndex = 0;

        int idx = mPollIndex;

        for( size_t i = 0; i < mDevices.size(); i++ )
        {
            SearchBatch* batch = mDevices[idx]->GetCompletedBatch();
            if( batch != NULL )
                return( batch );

            idx++;
            if( idx >= mDevices.size() )
                idx = 0;
        }

        return( NULL );
    }

    virtual void ReleaseBatch( SearchBatch* batch )
    {
        batch->mDevice->ReleaseBatch( batch );
    }

    virtual void CancelAllBatchesSync()
    {
        Mutex::Scope scope( mMutex );

        for( size_t i = 0; i < mDevices.size(); i++ )
            mDevices[i]->CancelAllBatchesSync();
    }

};






#endif // PIGEON_CUDA_HOST
#endif // PIGEON_ENABLE_CUDA
#endif // PIGEON_GPU_H__

