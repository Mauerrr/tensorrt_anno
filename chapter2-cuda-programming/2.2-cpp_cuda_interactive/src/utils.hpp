#ifndef __UTILS_HPP__
#define __UTILS_HPP__

#include <cuda_runtime.h>
#include <system_error>

// 一般cuda的check都是这样写成宏
//宏就是用#define定义的一段代码，它是一种预处理指令，用来在程序编译之前处理一些代码，比如宏定义、条件编译、文件包含等。一般包括函数、变量、表达式等。通常由#define、#ifdef、#ifndef、#endif等关键字组成。
#define CUDA_CHECK(call) {                                                 \
    cudaError_t error = call;                                              \
    if (error != cudaSuccess) {                                            \
        printf("ERROR: %s:%d, ", __FILE__, __LINE__);                      \
        printf("CODE:%d, DETAIL:%s\n", error, cudaGetErrorString(error));  \
        exit(1);                                                           \
    }                                                                      \
}

#endif //__UTILS__HPP__
