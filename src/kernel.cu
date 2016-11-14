// kernel.cu - PIGEON CHESS ENGINE (c) 2012-2016 Stuart Riffle

#include "platform.h"
#include "defs.h"
#include "bits.h"
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
/// hope of averaging out the total number of nodes per thread. 

__global__ void SearchPositionsOnGPU( 
    const SearchJobInput*   inputBuf, 
    SearchJobOutput*        outputBuf, 
    int                     count, 
    int                     stride, 
    HashTable*              hashTable, 
    Evaluator*              evaluator, 
    i32*                    options,
    i32*                    exitFlag )
{
    const SearchJobInput*   input   = NULL;
    SearchJobOutput*	    output  = NULL;
    SearchMetrics           metrics;
    HistoryTable            historyTable;
    SearchState< 1, u64 >   ss;

    historyTable.Clear();

    ss.mHashTable       = hashTable;
    ss.mEvaluator       = evaluator;
    ss.mOptions         = options;
    ss.mMetrics	        = &metrics;
    ss.mHistoryTable    = &historyTable;

    int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    while( idx < count )
    {
        if( exitFlag )
        {
            //__threadfence_system();
            //
            //i32 flag = atomicAdd( exitFlag, 0 );
            //if( flag )
            //    break;

            //i32 flag = *((volatile i32*) exitFlag);
            //if( flag )
            //    break;

            //asm( "ld.global.cg.u32 %0, [%1];" : "=r"( flag ) : "l"( exitFlag ) );
            //if( flag )
            //    break;
        }

        if( input == NULL )
        {
            input  = inputBuf  + idx;
            output = outputBuf + idx;

            ss.PrepareSearch( &input->mPosition, &input->mMoveMap, input->mDepth, input->mPly, input->mScore, input->mAlpha, input->mBeta );
            metrics.Clear();
        }

        ss.Step();

        if( ss.IsDone() )
        {
            output->mScore          = ss.GetFinalScore();
            output->mNodes          = metrics.mNodesTotal;
            output->mSteps          = metrics.mSteps;
            output->mDeepestPly     = ss.mDeepestPly;

            for( int i = 0; i < input->mDepth; i++ )
                output->mPath[i] = ss.mFrames[i].bestMove;

            input  = NULL;
            output = NULL;

            idx += stride;
        }
    }

    __threadfence();
}


void QueueSearchBatch( SearchBatch* batch, int blockCount, int blockSize, i32* exitFlag )
{
    // Copy the inputs to device

    cudaMemcpyAsync( batch->mInputDev, batch->mInputHost, sizeof( SearchJobInput ) * batch->mCount, cudaMemcpyHostToDevice, batch->mStream );

    // Clear the device outputs

    cudaMemsetAsync( batch->mOutputDev, 0, sizeof( SearchJobOutput ) * batch->mCount, batch->mStream );

    // Run the search kernel

    int stride = blockCount * blockSize;

    cudaEventRecord( batch->mStartEvent, batch->mStream );
    SearchPositionsOnGPU<<< blockCount, blockSize, 0, batch->mStream >>>( 
        batch->mInputDev, 
        batch->mOutputDev, 
        batch->mCount, 
        stride, 
        batch->mHashTableDev, 
        batch->mEvaluatorDev,
        batch->mOptionsDev,
        exitFlag );
    cudaEventRecord( batch->mEndEvent, batch->mStream );

    // Copy the outputs to host

    cudaMemcpyAsync( batch->mOutputHost, batch->mOutputDev, sizeof( SearchJobOutput ) * batch->mCount, cudaMemcpyDeviceToHost, batch->mStream );

    // Record an event we can test for completion

    cudaEventRecord( batch->mReadyEvent, batch->mStream );
}

