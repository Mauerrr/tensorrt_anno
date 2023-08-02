#include <cuda_runtime.h>
#include <math.h>

__global__ void myScalarKernel(
    const float* input, float* output, 
    const float scalar, const int nElements)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= nElements) 
        return;

    output[index] = input[index] + scalar;
}

void myScalarImpl(const float* inputs, float* outputs, const float scalar, const int nElements, cudaStream_t stream)
{
    dim3 blockSize(256, 1, 1);
    dim3 gridSize(ceil(float(nElements) / 256), 1, 1);
    myScalarKernel<<<gridSize, blockSize, 0, stream>>>(inputs, outputs, scalar, nElements);
}