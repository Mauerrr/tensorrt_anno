#ifndef __UTILS_HPP__
#define __UTILS_HPP__

#include <cuda_runtime.h>
#include <system_error>
#include <stdarg.h>

//在C/C++中，#define用于定义宏，它是预处理指令的一部分。宏本质上是一种文本替换工具，可以用来在编译之前替换代码中的文本。定义了一个名为CUDA_CHECK的宏。这个宏接收一个参数call，并在代码中扩展为调用函数__cudaCheck，同时传递三个参数：call、__FILE__和__LINE__。CUDA_CHECK是宏的名称，call是它的参数，这个参数在宏被使用时将被替换成具体的CUDA API调用。
//在C和C++中，名称前面加双下划线（__）或者以一个下划线后跟一个大写字母开始（_X）的做法通常是保留给编译器和标准库的实现的。
//避免命名冲突：使用这种命名约定可以避免与用户程序中的普通函数或变量名发生冲突。
//标明内部或低级功能：以双下划线开头的函数通常表示它们是低级或仅供内部使用的功能，普通开发者在日常编程中应避免直接调用这些函数。
//强调特殊用途：这种命名风格也用来标明某些特殊的、具有特定用途的函数或变量，如宏定义中使用的函数。
#define CUDA_CHECK(call)             __cudaCheck(call, __FILE__, __LINE__)
#define LAST_KERNEL_CHECK(call)      __kernelCheck(__FILE__, __LINE__)
#define LOG(...)                     __log_info(__VA_ARGS__)

#define BLOCKSIZE 16

static void __cudaCheck(cudaError_t err, const char* file, const int line) {
    if (err != cudaSuccess) {
        printf("ERROR: %s:%d, ", file, line);
        printf("CODE:%s, DETAIL:%s\n", cudaGetErrorName(err), cudaGetErrorString(err));
        exit(1);
    }
}

static void __kernelCheck(const char* file, const int line) {
    cudaError_t err = cudaPeekAtLastError();
    if (err != cudaSuccess) {
        printf("ERROR: %s:%d, ", file, line);
        printf("CODE:%s, DETAIL:%s\n", cudaGetErrorName(err), cudaGetErrorString(err));
        exit(1);
    }
}

// 使用变参进行LOG的打印。比较推荐的打印log的写法
static void __log_info(const char* format, ...) {
    char msg[1000];
    va_list args;
    va_start(args, format);

    vsnprintf(msg, sizeof(msg), format, args);

    fprintf(stdout, "%s\n", msg);
    va_end(args);
}

void initMatrix(float* data, int size, int low, int high, int seed);
void printMat(float* data, int size);
void compareMat(float* h_data, float* d_data, int size);

#endif //__UTILS_HPP__//
