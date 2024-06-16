//定义了几个函数的原型，在cpp中通常将函数原型（声明）和函数体（定义）分开，原型放在头文件中，函数体放在cpp文件中，这种
//做法的好处是模块化，方便管理和维护，同时也方便其他文件调用这些函数，只需要包含头文件即可，不需要关心函数的具体实现。
//__PRINT_INDEX_HPP 是一个宏，用来防止头文件print_index.hpp被多次包含。通常来说，在 C 和 C++ 的每个头文件中使用头文件保护（header guards）
//是一个很好的编程实践。这种做法可以有效防止头文件被重复包含，从而避免可能的编译错误，比如函数或变量的重复定义。因此，建议在每个头文件的开头加上类似于你给出的宏定义保护，以确保每个头文件只被编译一次，即使它在项目中的不同部分被多次引用。
#ifndef __PRINT_INDEX_HPP
#define __PRINT_INDEX_HPP

#include <cuda_runtime.h>
void print_idx_device(dim3 grid, dim3 block);
void print_dim_device(dim3 grid, dim3 block);
void print_thread_idx_per_block_device(dim3 grid, dim3 block);
void print_thread_idx_device(dim3 grid, dim3 block);

#endif //__PRINT_INDEX_HPP
