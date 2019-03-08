#include "point.h"
#include "utils.h"

#include <iostream>
#include <string>
#include <cmath>
#include <stdexcept>

#include <pthread.h>
#include <unistd.h> 
#include <cuda_runtime.h>

#define _USE_MATH_DEFINES
#define IMAGE_SIZE          (64 * 4000)

using namespace std;

__device__ inline float get_degree(float x, float y)
{
    return atan2(y, x) / M_PI * 180 + 180;
}

__device__ bool valid(Point& point)
{
    return abs(point.x) >= 0.5 && abs(point.y) >= 0.5;
}

__device__ float distance(Point& point)
{
    return sqrt(point.x * point.x + point.y * point.y);
}

__device__ int axis_image(Point& point)
{
    float u = get_degree(point.x, point.y) / 0.09;
    float v = (196 - get_degree(distance(point), point.z)) * 2;
    if (v > 63) v = 63;

    return (int)(floor(u) + 4000 * floor(v));
}

__global__ void initialize_data(double* data)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < IMAGE_SIZE)
    {
        data[idx + 0 * IMAGE_SIZE] = 0;
        data[idx + 1 * IMAGE_SIZE] = 0;
        data[idx + 2 * IMAGE_SIZE] = 0;
        data[idx + 3 * IMAGE_SIZE] = 0;
    }
}

__global__ void prepare_data(Point* pts, uint8_t* categories, double* intensities, double* data, size_t* size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < *size)
    {
        Point& point = pts[idx];
        
        // 1. check the point is valid.
        // 2. if the axis of the image exist, check the category and then check the distance.
        if (valid(point))
        {            
            int axis = axis_image(point);
            int cd = 255.0 - distance(point);
            int cc = categories[idx];
            int od = data[axis * 4 + 1];
            int oc = data[axis * 4 + 3];

            if (od == 0 || (cc > 0 && oc == 0) ||
                (((cc == 0 && oc == 0) || (cc > 0 && oc > 0)) && cd > od))
            {
                data[axis * 4 + 0] = idx;
                data[axis * 4 + 1] = cd;
                data[axis * 4 + 2] = intensities[idx];
                data[axis * 4 + 3] = cc;
            }
        }
    }
}

template <typename T>
std::vector<T> csv_load(std::string fname, bool header, T (*convert)(const char*))
{
    char buffer[1024];
    FILE* fp = fopen(fname.c_str(), "rb");

    if(!fp) 
        throw std::runtime_error("csv_load: Unable to open file " + fname + "!");
    
    std::vector<T> arr;
    while (fgets(buffer, 1024, fp) != NULL)
    {
        if (header)
        {
            header = false;
            continue;
        }

        T cur = convert(buffer);
        arr.push_back(cur);
    }

    fclose(fp);
    return arr;
}

void prepare_each(const char* root_folder, const char* fname, int device_id)
{
    char pts_name[256], intensity_name[256], category_name[256], saved_name[256];
    snprintf(pts_name,       256, "%s/pts/%s.csv",           root_folder, fname);
    snprintf(intensity_name, 256, "%s/intensity/%s.csv",     root_folder, fname);
    snprintf(category_name,  256, "%s/category/%s.csv",      root_folder, fname);
    snprintf(saved_name,     256, "%s/binary/merged/%s.npy", root_folder, fname);

    vector<Point>   points      = csv_load<Point>(pts_name, false, atopoint);
    vector<double>  intensities = csv_load<double>(intensity_name, false, atof);
    vector<uint8_t> categories  = csv_load<uint8_t>(category_name, false, ato_uint8);
    double*         results     = (double*)malloc(IMAGE_SIZE * 4 * sizeof(double));

    size_t      count            = points.size();
    size_t*     dev_count        = 0;
    Point*      dev_points       = 0;
    double*     dev_intentsities = 0;
    uint8_t*    dev_categories   = 0;
    double*     dev_traindata    = 0;

    checkCudaErrors(cudaSetDevice(device_id));
    checkCudaErrors(cudaMalloc((void**)&dev_count,        1             * sizeof(size_t)));
    checkCudaErrors(cudaMalloc((void**)&dev_points,       points.size() * sizeof(Point)));
    checkCudaErrors(cudaMalloc((void**)&dev_intentsities, points.size() * sizeof(double)));
    checkCudaErrors(cudaMalloc((void**)&dev_categories,   points.size() * sizeof(uint8_t)));
    checkCudaErrors(cudaMalloc((void**)&dev_traindata,    IMAGE_SIZE * 4 * sizeof(double)));

    checkCudaErrors(cudaMemcpy(dev_count,        &count,          sizeof(size_t),                  cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dev_points,       &points[0],      points.size() * sizeof(Point),   cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dev_intentsities, &intensities[0], points.size() * sizeof(double),  cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dev_categories,   &categories[0],  points.size() * sizeof(uint8_t), cudaMemcpyHostToDevice));

    initialize_data<<<256, 1000>>>(dev_traindata);
    checkCudaErrors(cudaThreadSynchronize());
    prepare_data<<<256, 512>>>(dev_points, dev_categories, dev_intentsities, dev_traindata, dev_count);
    checkCudaErrors(cudaThreadSynchronize());
    checkCudaErrors(cudaMemcpy(results, dev_traindata, IMAGE_SIZE * 4 * sizeof(double), cudaMemcpyDeviceToHost));

    save_result(saved_name, results);

error:
    cudaFree(dev_points);
    cudaFree(dev_intentsities);
    cudaFree(dev_categories);
    cudaFree(dev_traindata);
    cudaFree(dev_count);
    free(results);
}

void prepare(string root_folder, vector<string> files, int device_id)
{
    int count = 0;
    char fname[100];

    const char* folder = root_folder.c_str();
    auto file = files.begin();

    while (file != files.end())
    {
        if (count % 100 == 0)
        {
            fprintf(stdout, "Now Processed %5d/%5d\n", count, files.size());
            fflush(stdout);
        }

        snprintf(fname, 100, "%s", file->substr(0, file->size() - 4).c_str());
        prepare_each(folder, fname, device_id);

        count++;
        file++;
    }
}
