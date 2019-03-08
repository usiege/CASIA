#ifndef __UTILS_H__
#define __UTILS_H__
#include "point.h"

#include <iostream>
#include <vector>
#include <string>

#define NUM_THREADS         5
#define checkCudaErrors(a) do { \
    if (cudaSuccess != (a)) { \
        fprintf(stderr, "Cuda runtime error in line %d of file %s \
                         : %s \n", __LINE__, __FILE__, cudaGetErrorString(cudaGetLastError())); \
        goto error; \
    } \
} while(0);

inline uint8_t ato_uint8(const char* buffer)
{
    return (uint8_t)(atoi(buffer));
}

std::vector<std::string> get_file_names(const std::string& dir);
void prepare(std::string root_folder, std::vector<std::string> files, int device_id);
void save_result(const char* fname, const double* results);

#endif
