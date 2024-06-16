#ifndef __MATMUL_HPP__
#define __MATMUL_HPP__

void MatmulOnDevice(float *M_host, float *N_host, float* P_host, int width, int blockSize);
extern void MatmulOnHost(float *M_host, float *N_host, float* P_host, int width);

#endif //__MATMUL_HPP__

//在我的main.cpp文件中我只需要include这个hpp的头文件，hpp也只是包含了一些函数的声明，我就可以在main文件中直接使用这些被声明过的函数，即使我并没有include这些函数的定义文件
//