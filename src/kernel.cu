// kernel.cu - PIGEON CHESS ENGINE (c) 2012-2016 Stuart Riffle

#include "platform.h"
#include "defs.h"
#include "bits.h"
#include "simd.h"
#include "position.h"
#include "movelist.h"
#include "eval.h"
#include "table.h"
#include "search.h"

using namespace Pigeon;


/// Process a batch of search jobs
///
/// Individual searches take an unpredictable number of nodes to complete. If every thread
/// handled only one search, it would be idle until the longest search in the batch was done.
/// To reduce that effect, every thread processes a number of searches sequentially, in the 
/// hope of averaging out the total number of nodes per thread. The UCI option "GPU Job Multiple"
/// sets the number of searches per thread.

__global__ void SearchPositionsOnGPU( const SearchJobInput* inputBuf, SearchJobOutput* outputBuf, int count, int stride, HashTable* hashTable, Evaluator* evaluator )
{
    const SearchJobInput*   input   = NULL;
    SearchJobOutput*	    output  = NULL;
    SearchMetrics           metrics;
    SearchState< 1, u64 >   ss;

    HashTable hashTableLocal = *hashTable;
    Evaluator evaluatorLocal = *evaluator;

    ss.mHashTable   = &hashTableLocal;
    ss.mEvaluator   = &evaluatorLocal;
    ss.mMetrics	    = &metrics;

    int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    while( idx < count )
    {
        if( input == NULL )
        {
            input  = inputBuf  + idx;
            output = outputBuf + idx;

            ss.PrepareSearch( &input->mPosition, &input->mMoveMap, input->mDepth, input->mPly, input->mScore, input->mAlpha, input->mBeta );
            metrics.Clear();
        }

        ss.Advance();

        if( ss.IsDone() )
        {
            output->mScore          = ss.GetFinalScore();
            output->mNodes          = metrics.mNodesTotal;
            output->mDeepestPly     = ss.mDeepestPly;

            ss.ExtractBestLine( &output->mBestLine );

            input  = NULL;
            output = NULL;

            idx += stride;
        }
    }

    __threadfence();
}


void QueueSearchBatch( SearchBatch* batch, int blockCount, int blockSize )
{
    // Copy the inputs to device

    cudaMemcpyAsync( batch->mInputDev, batch->mInputHost, sizeof( SearchJobInput ) * batch->mCount, cudaMemcpyHostToDevice, batch->mStream );

    // Clear the device outputs

    cudaMemsetAsync( batch->mOutputDev, 0, sizeof( SearchJobOutput ) * batch->mCount, batch->mStream );

    // Run the search kernel

    cudaFuncSetCacheConfig( SearchPositionsOnGPU, cudaFuncCachePreferL1 );

    int stride = blockCount * blockSize;
    SearchPositionsOnGPU<<< blockCount, blockSize, 0, batch->mStream >>>( batch->mInputDev, batch->mOutputDev, batch->mCount, stride, batch->mHashTable, batch->mEvaluator );

    // Copy the outputs to host

    cudaMemcpyAsync( batch->mOutputHost, batch->mOutputDev, sizeof( SearchJobOutput ) * batch->mCount, cudaMemcpyDeviceToHost, batch->mStream );

    // Record an event we can test for completion

    cudaEventRecord( batch->mEvent, batch->mStream );
}

