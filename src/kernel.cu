#include <iostream>
#include <fstream>

#include <chrono>

#include "traders.cuh"
#include "cudamacro.h"

#define timer std::chrono::high_resolution_clock

// Default parameters
int device_id = 0;
int threads = 128;
const long long grid_height = 1024;
const long long grid_width = 1024;
const long long grid_depth = 1024;
int total_updates = 10000;
unsigned int seed = std::chrono::steady_clock::now().time_since_epoch().count();
// the rng offset can be used to return the random number generator to a specific
// state of a simulation. It is equal to the total number of random numbers
// generated. Meaning the following equation holds for this specific case:
// rng_offset = total_updates * grid_width * grid_height
long long rng_offset = 0;
float alpha = 0.0f;
float j = 1.0f;
float beta = 0.226f;

int gpuDeviceInit(int device_id) {
    int device_count;
    CHECK_CUDA(cudaGetDeviceCount(&device_count));

    if (device_count == 0) {
        fprintf(stderr,
                "gpuDeviceInit() CUDA error: "
                "no devices supporting CUDA.\n");
    exit(EXIT_FAILURE);
    }

    if (device_id < 0) {
        device_id = 0;
    }

    if (device_id > device_count - 1) {
        fprintf(stderr, "\n");
        fprintf(stderr, ">> %d CUDA capable GPU device(s) detected. <<\n",
                device_count);
        fprintf(stderr,
                ">> gpuDeviceInit (-device=%d) is not a valid"
                " GPU device. <<\n",
                device_id);
        fprintf(stderr, "\n");
        return -device_id;
    }

    int computeMode = -1, major = 0, minor = 0;
    CHECK_CUDA(cudaDeviceGetAttribute(&computeMode, cudaDevAttrComputeMode, device_id));
    CHECK_CUDA(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device_id));
    CHECK_CUDA(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device_id));
    if (computeMode == cudaComputeModeProhibited) {
        fprintf(stderr,
                "Error: device is running in <Compute Mode "
                "Prohibited>, no threads can use cudaSetDevice().\n");
        return -1;
    }

    if (major < 1) {
        fprintf(stderr, "gpuDeviceInit(): GPU device does not support CUDA.\n");
        exit(EXIT_FAILURE);
    }

    CHECK_CUDA(cudaSetDevice(device_id));
    return device_id;
}

int main(int argc, char** argv) {
    int device_id = 0;
    // finds and sets the specified cuda device
    gpuDeviceInit(device_id);

    // Finds and prints the devices name and computing power
    cudaDeviceProp deviceProp;
    deviceProp.major = 0;
    deviceProp.minor = 0;
    CHECK_CUDA(cudaGetDeviceProperties(&deviceProp, device_id));
    printf("CUDA device [%s] has %d Multi-Processors, Compute %d.%d\n",
        deviceProp.name, deviceProp.multiProcessorCount, deviceProp.major, deviceProp.minor);

    signed char *d_black_tiles, *d_white_tiles;
    float *random_values;
    curandGenerator_t rng;
    signed char *h_black_tiles, *h_white_tiles;

    // The global market represents the sum over the strategies of each
    // agent. Agents will choose a strategy contrary to the sign of the
    // global market.
    int *d_global_market;
    int *h_global_market;

    // Set up cuRAND generator
    CHECK_CURAND(curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_PHILOX4_32_10));
    CHECK_CURAND(curandSetPseudoRandomGeneratorSeed(rng, seed));
    CHECK_CURAND(curandSetGeneratorOffset(rng, rng_offset));

    // allocate memory for the arrays
    CHECK_CUDA(cudaMalloc(&d_white_tiles, grid_height * grid_width / 2 * sizeof(*d_white_tiles)));
    CHECK_CUDA(cudaMalloc(&d_black_tiles, grid_height * grid_width / 2 * sizeof(*d_black_tiles)));
    CHECK_CUDA(cudaMalloc(&random_values, grid_height * grid_width / 2 * sizeof(*random_values)));
    CHECK_CUDA(cudaMalloc(&d_global_market, sizeof(*d_global_market)));
    h_black_tiles = (signed char*)malloc(grid_height * grid_width / 2 * sizeof(*h_black_tiles));
    h_white_tiles = (signed char*)malloc(grid_height * grid_width / 2 * sizeof(*h_white_tiles));
    h_global_market = (int*)malloc(sizeof(*h_global_market));

    int blocks = (grid_height * grid_width/2 + threads - 1) / threads;
    CHECK_CURAND(curandGenerateUniform(rng, random_values, grid_height * grid_width / 2));
    init_traders<<<blocks, threads>>>(d_black_tiles, random_values, grid_height, grid_width / 2, grid_depth);
    CHECK_CURAND(curandGenerateUniform(rng, random_values, grid_height * grid_width / 2));
    init_traders<<<blocks, threads>>>(d_white_tiles, random_values, grid_height, grid_width / 2, grid_depth);

    // Synchronize operations on the GPU with CPU
    CHECK_CUDA(cudaDeviceSynchronize());

    for (int iteration = 0; iteration < 1000; iteration++) {
        update(d_black_tiles, d_white_tiles, random_values, rng, beta, grid_height, grid_width, grid_depth);
    }

    timer::time_point start = timer::now();
    for (int iteration = 0; iteration < total_updates; iteration++) {
        update(d_black_tiles, d_white_tiles, random_values, rng, beta, grid_height, grid_width, grid_depth);
    }
    timer::time_point stop = timer::now();

    double duration = (double) std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();
    double spin_updates_per_nanosecond = grid_width * grid_height / duration * 1e-3 * total_updates;
    printf("Total computing time: %f\n", duration * 1e-6);
    printf("Updates per nanosecond: %f\n", spin_updates_per_nanosecond);
    CHECK_CUDA(cudaDeviceSynchronize());
    return 0;
}
