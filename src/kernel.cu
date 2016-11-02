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


__global__ void SearchPositionsOnGPU( const Pigeon::SearchJobInput* inputBuf, Pigeon::SearchJobOutput* outputBuf, int count, Pigeon::HashTable* hashTable, Pigeon::Evaluator* evaluator )
{
	int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	if( idx >= count )
		return;
		 
	const Pigeon::SearchJobInput*	input	= inputBuf  + idx;
	Pigeon::SearchJobOutput*		output	= outputBuf + idx;
	Pigeon::SearchMetrics			metrics;

	Pigeon::SearchState< 1, Pigeon::u64 > ss;

    ss.mHashTable	= hashTable;
    ss.mEvaluator	= evaluator;
    ss.mMetrics		= &metrics;

    output->mScore = ss.RunToDepth( input->mPosition, input->mSearchDepth );
	output->mNodes = metrics.mNodesTotal;

    ss.ExtractBestLine( &output->mBestLine );

	__threadfence();
}


void QueueSearchBatch( Pigeon::SearchBatch* batch, int blockSize )
{
	// Copy the inputs to device

	cudaMemcpyAsync( batch->mInputDev, batch->mInputHost, sizeof( Pigeon::SearchJobInput ) * batch->mCount, cudaMemcpyHostToDevice, batch->mStream );

	// Clear the device outputs

	cudaMemsetAsync( batch->mOutputDev, 0, sizeof( Pigeon::SearchJobOutput ) * batch->mCount, batch->mStream );

	// Run the search kernel

	int blockCount = (batch->mCount + blockSize - 1) / blockSize;
	SearchPositionsOnGPU<<< blockCount, blockSize, 0, batch->mStream >>>( batch->mInputDev, batch->mOutputDev, batch->mCount, batch->mHashTable, batch->mEvaluator );

	// Copy the outputs to host

	cudaMemcpyAsync( batch->mOutputHost, batch->mOutputDev, sizeof( Pigeon::SearchJobOutput ) * batch->mCount, cudaMemcpyDeviceToHost, batch->mStream );

	// Record an event we can test for completion

	cudaEventRecord( batch->mEvent, batch->mStream );
}

