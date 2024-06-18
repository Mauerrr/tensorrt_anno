#include "cuda_runtime_api.h"
#include "utils.hpp"

#define BLOCKSIZE 16

/* 
    使用shared memory把计算一个tile所需要的数据分块存储到访问速度快的memory中
    M_device: 指向设备（GPU）上第一个矩阵的指针。
    N_device: 指向设备（GPU）上第二个矩阵的指针。
    P_device: 指向设备（GPU）上结果矩阵的指针。
    width: 矩阵的宽度（假设是方阵，即行数和列数相同）。
*/
__global__ void MatmulSharedStaticKernel(float *M_device, float *N_device, float *P_device, int width){
    //声明共享内存
    __shared__ float M_deviceShared[BLOCKSIZE][BLOCKSIZE];
    __shared__ float N_deviceShared[BLOCKSIZE][BLOCKSIZE];
    /* 
        对于x和y, 根据blockID, tile大小和threadID进行索引
    */
    //计算线程的索引，threadIdx.x和threadIdx.y是线程在block里的下x和y的索引，是一个二维块
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    float P_element = 0.0;
    //这里的ty和tx是为了方便后面的计算，类似于简化书写
    //ty和tx对于每个线程都是不一样的，每个线程在执行核函数时都会运行相同的代码，由于 threadIdx.x 和 threadIdx.y 不同，每个线程将从全局内存中的不同位置加载数据到共享内存中的不同位置。因为所有线程是并行执行的，所以整个矩阵乘法的计算会非常高效。

    int ty = threadIdx.y;
    int tx = threadIdx.x;
    /* 对于每一个P的元素，我们只需要循环遍历width / tile_width 次就okay了，这里有点绕，画图理解一下。tile是子块*/
    for (int m = 0; m < width / BLOCKSIZE; m ++) {
        //从global memory中写取数据到shared memory中，y*width代表当前线程所在的起始位置在全局矩阵中的索引，m代表第几个子块
        M_deviceShared[ty][tx] = M_device[y * width + (m * BLOCKSIZE + tx)];
        N_deviceShared[ty][tx] = N_device[(m * BLOCKSIZE + ty)* width + x];
        //与shared memory中的线程进行一个同步
        __syncthreads();

        for (int k = 0; k < BLOCKSIZE; k ++) {
            P_element += M_deviceShared[ty][k] * N_deviceShared[k][tx];
        }
        __syncthreads();
    }

    //最后通过把每个线程的计算结果写入到全局内存中，因为他们写不同的位置，所以不会冲突
    P_device[y * width + x] = P_element;
}

__global__ void MatmulSharedDynamicKernel(float *M_device, float *N_device, float *P_device, int width, int blockSize){
    /* 
        声明动态共享变量的时候需要加extern，同时需要是一维的 
        注意这里有个坑, 不能够像这样定义： 
            __shared__ float M_deviceShared[];
            __shared__ float N_deviceShared[];
        因为在cuda中定义动态共享变量的话，无论定义多少个他们的地址都是一样的。
        所以直接声明一个动态空间，然后用不同的指针来搞
        所以如果想要像上面这样使用的话，需要用两个指针分别指向shared memory的不同位置才行
    */

    extern __shared__ float deviceShared[];
    //stride相当于一个偏移
    int stride = blockSize * blockSize;
    /* 
        对于x和y, 根据blockID, tile大小和threadID进行索引
    */
    int x = blockIdx.x * blockSize + threadIdx.x;
    int y = blockIdx.y * blockSize + threadIdx.y;

    float P_element = 0.0;

    int ty = threadIdx.y;
    int tx = threadIdx.x;
    /* 对于每一个P的元素，我们只需要循环遍历width / tile_width 次就okay了 */
    for (int m = 0; m < width / blockSize; m ++) {
        deviceShared[ty * blockSize + tx] = M_device[y * width + (m * blockSize + tx)];
        deviceShared[stride + (ty * blockSize + tx)] = N_device[(m * blockSize + ty)* width + x];
        __syncthreads();

        for (int k = 0; k < blockSize; k ++) {
            P_element += deviceShared[ty * blockSize + k] * deviceShared[stride + (k * blockSize + tx)];
        }
        __syncthreads();
    }

    if (y < width && x < width) {
        P_device[y * width + x] = P_element;
    }
}

/*
    使用Tiling技术
    一个tile处理的就是block, 将一个矩阵分为多个小的tile，这些tile之间的执行独立，并且可以并行
*/
void MatmulSharedOnDevice(float *M_host, float *N_host, float* P_host, int width, int blockSize, bool staticMem){
    /* 设置矩阵大小 */
    int size = width * width * sizeof(float);
    long int sMemSize = blockSize * blockSize * sizeof(float) * 2;

    /* 分配M, N在GPU上的空间*/
    float *M_device;
    float *N_device;
    CUDA_CHECK(cudaMalloc((void**)&M_device, size));
    CUDA_CHECK(cudaMalloc((void**)&N_device, size));

    /* 分配M, N拷贝到GPU上*/
    CUDA_CHECK(cudaMemcpy(M_device, M_host, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(N_device, N_host, size, cudaMemcpyHostToDevice));

    /* 分配P在GPU上的空间*/
    float *P_device;
    CUDA_CHECK(cudaMalloc((void**)&P_device, size));;

    /* 调用kernel来进行matmul计算, 在这个例子中我们用的方案是：使用一个grid，一个grid里有width*width个线程 */
    dim3 dimBlock(blockSize, blockSize);
    dim3 dimGrid(width / blockSize, width / blockSize);
    if (staticMem) {
        MatmulSharedStaticKernel <<<dimGrid, dimBlock>>> (M_device, N_device, P_device, width);
    } else {
        MatmulSharedDynamicKernel <<<dimGrid, dimBlock, sMemSize, nullptr>>> (M_device, N_device, P_device, width, blockSize);
    }

    /* 将结果从device拷贝回host*/
    CUDA_CHECK(cudaMemcpy(P_host, P_device, size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());

    /* 注意要在synchronization结束之后排查kernel的错误 */
    LAST_KERNEL_CHECK(); 

    /* Free */
    cudaFree(P_device);
    cudaFree(N_device);
    cudaFree(M_device);
}
