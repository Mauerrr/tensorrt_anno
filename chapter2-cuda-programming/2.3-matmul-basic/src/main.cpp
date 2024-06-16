#include <stdio.h>
#include <cuda_runtime.h>

#include "utils.hpp"
#include "timer.hpp"
#include "matmul.hpp"


int seed;
int main(){
    Timer timer;
    int width     = 1<<10; // 1,024
    int min       = 0;
    int max       = 1;
    int size      = width * width;
    int blockSize = 1;

    //h_matM是一个指针类型，然后将void*类型的指针强制转换为float*类型的指针，后面那个是乘法运算符
    float* h_matM = (float*)malloc(size * sizeof(float));
    float* h_matN = (float*)malloc(size * sizeof(float));
    float* h_matP = (float*)malloc(size * sizeof(float));
    float* d_matP = (float*)malloc(size * sizeof(float));
    
    seed = 1;
    initMatrix(h_matM, size, min, max, seed);
    seed += 1;
    initMatrix(h_matN, size, min, max, seed);
    
    /* CPU */
    timer.start();
    MatmulOnHost(h_matM, h_matN, h_matP, width);
    timer.stop();
    timer.duration<Timer::ms>("matmul in cpu");

    /* GPU warmup */
    //第一次调用 CUDA 函数时，CUDA 运行时需要加载和初始化，这包括加载 CUDA 驱动到主机内存，以及准备 GPU 设备进行计算。此外，当 CUDA 程序第一次运行某个内核时，该内核需要从 CPU 内存加载到 GPU 内存，并可能需要一些编译或其他初始化过程。这些操作通常会在第一次执行时造成额外的延迟。
    //GPU 设备从空闲状态转换到高效率运行状态可能需要一些时间。在 "冷启动" 时，GPU 的许多部件（如缓存、内存控制器等）可能还没有完全活跃起来。通过一个预热步骤，可以使 GPU 达到一个适合高性能运行的稳态。
    timer.start();
    MatmulOnDevice(h_matM, h_matN, d_matP, width, blockSize);
    timer.stop();
    timer.duration<Timer::ms>("matmul in gpu(warmup)");

    /* GPU general implementation, bs = 16*/
    blockSize = 16;
    timer.start();
    MatmulOnDevice(h_matM, h_matN, d_matP, width, blockSize);
    timer.stop();
    timer.duration<Timer::ms>("matmul in gpu(bs = 16)");
    compareMat(h_matP, d_matP, size);

    /* GPU general implementation, bs = 1*/
    blockSize = 1;
    timer.start();
    MatmulOnDevice(h_matM, h_matN, d_matP, width, blockSize);
    timer.stop();
    timer.duration<Timer::ms>("matmul in gpu(bs = 1)");
    compareMat(h_matP, d_matP, size);

    /* GPU general implementation, bs = 32*/
    blockSize = 32;
    timer.start();
    MatmulOnDevice(h_matM, h_matN, d_matP, width, blockSize);
    timer.stop();
    timer.duration<Timer::ms>("matmul in gpu(bs = 32)");
    compareMat(h_matP, d_matP, size);
    return 0;
}
