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


__global__ void SearchPositionsOnGPU( const Pigeon::SearchJobInput* inputBuf, Pigeon::SearchJobOutput* outputBuf, int count )
{
	int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	if( idx >= count )
		return;
		 
	const Pigeon::SearchJobInput*	input	= inputBuf  + idx;
	Pigeon::SearchJobOutput*		output	= outputBuf + idx;

	Pigeon::SearchState< 1, Pigeon::u64 > ss;

    ss.mHashTable	= input->mHashTable;
    ss.mEvaluator	= input->mEvaluator;
    ss.mMetrics		= &output->mMetrics;

    output->mScore = ss.RunToDepth( input->mPosition, input->mSearchDepth );
    ss.ExtractBestLine( &output->mBestLine  );

	__threadfence();
}


extern "C" void QueueSearchBatch( Pigeon::SearchBatch* batch )
{
	cudaMemcpyAsync( batch->mInputDev, batch->mInputHost, sizeof( Pigeon::SearchJobInput ) * batch->mCount, cudaMemcpyHostToDevice, batch->mStream );
	cudaMemsetAsync( batch->mOutputDev, 0, sizeof( Pigeon::SearchJobOutput ) * batch->mCount, batch->mStream );

	int	minGridSize;
	int blockSize;

	if( cudaOccupancyMaxPotentialBlockSize( &minGridSize, &blockSize, SearchPositionsOnGPU, 0, 0 ) != cudaSuccess )
		blockSize = 32;

	int blockCount = (batch->mCount + blockSize - 1) / blockSize;

	SearchPositionsOnGPU<<< blockCount, blockSize, 0, batch->mStream >>>( batch->mInputDev, batch->mOutputDev, batch->mCount );

	cudaMemcpyAsync( batch->mOutputHost, batch->mOutputDev, sizeof( Pigeon::SearchJobOutput ) * batch->mCount, cudaMemcpyDeviceToHost, batch->mStream );
	cudaEventRecord( batch->mEvent, batch->mStream );
}

