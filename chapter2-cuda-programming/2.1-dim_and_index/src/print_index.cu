#include <cuda_runtime.h>
#include <stdio.h>

// 声明一个在GPU上运行的内核函数
//__global__ 是一个CUDA修饰符，用于声明一个在GPU上运行的内核函数。
__global__ void print_idx_kernel(){
    printf("block idx: (%3d, %3d, %3d), thread idx: (%3d, %3d, %3d)\n",
         // blockIdx 用于获取当前块的索引
         blockIdx.z, blockIdx.y, blockIdx.x,
         // threadIdx 用于获取当前线程的索引
         threadIdx.z, threadIdx.y, threadIdx.x);
}

// 打印网格维度和块维度
__global__ void print_dim_kernel(){
    printf("grid dimension: (%3d, %3d, %3d), block dimension: (%3d, %3d, %3d)\n",
          // gridDim 用于获取当前网格的维度
         gridDim.z, gridDim.y, gridDim.x,
         // blockDim 用于获取当前块的维度
         blockDim.z, blockDim.y, blockDim.x);
}

// 打印当前线程所在的块索引和线程在块内的一维索引
__global__ void print_thread_idx_per_block_kernel(){
    int index = threadIdx.z * blockDim.x * blockDim.y + \
              threadIdx.y * blockDim.x + \
              threadIdx.x;

    printf("block idx: (%3d, %3d, %3d), thread idx: %3d\n",
         blockIdx.z, blockIdx.y, blockIdx.x,
         index);
}

__global__ void print_thread_idx_per_grid_kernel(){
    int bSize  = blockDim.z * blockDim.y * blockDim.x;

    int bIndex = blockIdx.z * gridDim.x * gridDim.y + \
               blockIdx.y * gridDim.x + \
               blockIdx.x;

    int tIndex = threadIdx.z * blockDim.x * blockDim.y + \
               threadIdx.y * blockDim.x + \
               threadIdx.x;

    int index  = bIndex * bSize + tIndex;

    printf("block idx: %3d, thread idx in block: %3d, thread idx: %3d\n", 
         bIndex, tIndex, index);
}

// 计算当前线程在块内的一维索引
__global__ void print_cord_kernel(){
    // 使用三维索引计算公式，将三维索引转换为一维索引
    int index = threadIdx.z * blockDim.x * blockDim.y + \
              threadIdx.y * blockDim.x + \
              threadIdx.x;

    // 计算当前线程在网格内的二维坐标
    int x  = blockIdx.x * blockDim.x + threadIdx.x;
    int y  = blockIdx.y * blockDim.y + threadIdx.y;

    // 打印当前线程所在的块索引、线程索引和网格内的二维坐标
    printf("block idx: (%3d, %3d, %3d), thread idx: %3d, cord: (%3d, %3d)\n",
         blockIdx.z, blockIdx.y, blockIdx.x,
         index, x, y);
}

void print_one_dim(){
    int inputSize = 8;
    //就是block下有几个thread
    int blockDim = 4;
    int gridDim = inputSize / blockDim;

    dim3 block(blockDim);
    dim3 grid(gridDim);

    /* 这里建议大家吧每一函数都试一遍*/
    //如果这里不加cudaDeviceSynchronize()会导致cpu先输出printf，然后函
    //数的调用再输出，这是由于因为CUDA内核函数是异步启动的，内核函数开始执行后，CPU会立即继续执行后续代码，而不会等待GPU完成。
    /*blockIdx.x 并不是一个数组，而是一个表示当前线程所在块的索引的变量。之所以有很多输出，是因为你的内核函数在多个线程中并行执行，每个线程都
    会执行一次 printf，输出各自的块和线程索引。*/

    printf("print_idx_kernel: \n");
    print_idx_kernel<<<grid, block>>>();
    cudaDeviceSynchronize();  // 同步以确保顺序输出

    printf("print_dim_kernel: \n");
    print_dim_kernel<<<grid, block>>>();
    cudaDeviceSynchronize();  // 同步以确保顺序输出

    printf("print_thread_idx_per_block_kernel: \n");
    print_thread_idx_per_block_kernel<<<grid, block>>>();
    cudaDeviceSynchronize();  // 同步以确保顺序输出

    printf("print_thread_idx_per_grid_kernel: \n");
    print_thread_idx_per_grid_kernel<<<grid, block>>>();
    cudaDeviceSynchronize();  // 同步以确保顺序输出

    cudaDeviceSynchronize();
}

void print_two_dim(){
    int inputWidth = 4;

    int blockDim = 2;
    int gridDim = inputWidth / blockDim;

    dim3 block(blockDim, blockDim);
    dim3 grid(gridDim, gridDim);

    printf("print_idx_kernel: \n");
    print_idx_kernel<<<grid, block>>>();
    cudaDeviceSynchronize();  // 同步以确保顺序输出

    printf("print_dim_kernel: \n");
    print_dim_kernel<<<grid, block>>>();
    cudaDeviceSynchronize();  // 同步以确保顺序输出

    printf("print_thread_idx_per_block_kernel: \n");
    print_thread_idx_per_block_kernel<<<grid, block>>>();
    cudaDeviceSynchronize();  // 同步以确保顺序输出

    printf("print_thread_idx_per_grid_kernel: \n");
    print_thread_idx_per_grid_kernel<<<grid, block>>>();
    cudaDeviceSynchronize();  // 同步以确保顺序输出

    cudaDeviceSynchronize();
}

void print_cord(){
    int inputWidth = 4;

    int blockDim = 2;
    int gridDim = inputWidth / blockDim;

    dim3 block(blockDim, blockDim);
    dim3 grid(gridDim, gridDim);

    print_cord_kernel<<<grid, block>>>();

    cudaDeviceSynchronize();
}

int main() {
    /*
    synchronize是同步的意思，有几种synchronize

    cudaDeviceSynchronize: CPU与GPU端完成同步，CPU不执行之后的语句，知道这个语句以前的所有cuda操作结束
    cudaStreamSynchronize: 跟cudaDeviceSynchronize很像，但是这个是针对某一个stream的。只同步指定的stream中的cpu/gpu操作，其他的不管
    cudaThreadSynchronize: 现在已经不被推荐使用的方法
    __syncthreads:         线程块内同步
    */
   
    //print_one_dim();
    print_two_dim();
    //print_cord();
    return 0;
}
