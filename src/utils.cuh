#ifndef __UTILS_CUDA__
#define __UTILS_CUDA__

using namespace std;

#define CHECK_CUDA_ERROR(val)   check((val), #val, __FILE__, __LINE__)
#define CHECK_LAST_CUDA_ERROR() checkLast(__FILE__, __LINE__)

void check(cudaError err, const char* const func, const char* const file, const int line);
void checkLast(const char* const file, const int line);

void describeDevice (int dev, cudaDeviceProp& deviceProp);
unsigned int next_pow_2(unsigned int x);

#endif

