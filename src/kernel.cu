
//__CUDACC__ defines whether nvcc is steering compilation or not
//__CUDA_ARCH__is always undefined when compiling host code, steered by nvcc or not
//__CUDA_ARCH__is only defined for the device code trajectory of compilation steered by nvcc

#include "cuda_runtime.h"
#include "device_launch_parameters.h"



#include "platform.h"
#include "defs.h"
#include "bits.h"
#include "simd.h"
#include "position.h"
#include "movelist.h"
#include "eval.h"
#include "table.h"
#include "search.h"




#include <stdio.h>



__device__ void Foo()
{
    int i = threadIdx.x;

	Pigeon::Position pos;
	pos.Reset();

	Pigeon::SearchState< 1, Pigeon::u64 > ss;
	Pigeon::EvalTerm score = ss.RunToDepth( pos, 3 );

    printf( "Thread %d says %d\n", i, score );
}

__global__ void RunFoo()
{
    Foo();
}

using namespace Pigeon;

int main()
{
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    //cudaStatus = cudaSetDevice(0);
    //if (cudaStatus != cudaSuccess) {
    //    fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
    //    goto Error;
    //}

    RunFoo<<< 1, 1 >>>();


    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        //goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        //goto Error;
    }



    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}
