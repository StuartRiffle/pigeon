// cuda.h - PIGEON CHESS ENGINE (c) 2012-2016 Stuart Riffle

#ifndef PIGEON_GPU_H__
#define PIGEON_GPU_H__

#include <cuda_runtime_api.h>

namespace Pigeon {

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


};
#endif // PIGEON_GPU_H__
