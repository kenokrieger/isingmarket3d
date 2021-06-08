#include <iostream>
#include <fstream>
#include <ctime>
#include <iomanip>
#include <chrono>

#include "traders.cuh"
#include "cudamacro.h"
#include "cudahelper.h"

#define timer std::chrono::high_resolution_clock

// Default parameters
int device_id = 0;
int threads = 64;
const long long grid_height = 128;
const long long grid_width = 128;
const long long grid_depth = 128;
int total_updates = 10000;
unsigned int seed = 1234;//std::chrono::steady_clock::now().time_since_epoch().count();
// the rng offset can be used to return the random number generator to a specific
// state of a simulation. It is equal to the total number of random numbers
// generated. Meaning the following equation holds for this specific case:
// rng_offset = (total_updates + 1) * grid_width * grid_height
// (+ 1 because of the random numbers created for the initilisation)
long long rng_offset = 0;
float alpha = 0.0f;
float j = 1.0f;
float beta = 0.226f;


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

    // The global market represents the sum over the strategies of each
    // agent. Agents will choose a strategy contrary to the sign of the
    // global market.
    int global_market = 0;

    // Set up cuRAND generator
    CHECK_CURAND(curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_PHILOX4_32_10));
    CHECK_CURAND(curandSetPseudoRandomGeneratorSeed(rng, seed));
    CHECK_CURAND(curandSetGeneratorOffset(rng, rng_offset));

    // allocate memory for the arrays
    CHECK_CUDA(cudaMalloc(&d_white_tiles, grid_depth * grid_height * grid_width / 2 * sizeof(*d_white_tiles)));
    CHECK_CUDA(cudaMalloc(&d_black_tiles, grid_depth * grid_height * grid_width / 2 * sizeof(*d_black_tiles)));
    CHECK_CUDA(cudaMalloc(&random_values, grid_depth * grid_height * grid_width / 2 * sizeof(*random_values)));

    init_traders(d_black_tiles, d_white_tiles, rng, random_values, grid_width, grid_height, grid_depth, threads);
    // Synchronize operations on the GPU with CPU
    CHECK_CUDA(cudaDeviceSynchronize());

    std::ofstream file;
    file.open(".data/debug/magnetisation.dat");

    timer::time_point start = timer::now();
    for (int iteration = 0; iteration < total_updates; iteration++) {
        update(d_black_tiles, d_white_tiles, random_values, rng, global_market, alpha, beta, j, grid_height, grid_width, grid_depth);
        global_market = sum_array(d_black_tiles, grid_depth * grid_height * grid_width / 2);
        global_market += sum_array(d_white_tiles, grid_depth * grid_height * grid_width / 2);

        if (file.is_open()) {
            file << global_market;
            if (iteration != total_updates - 1) file << ' ';
        }
    }
    timer::time_point stop = timer::now();
    file.close();

    file.open("logs/ising.log", std::ios_base::app);
    auto t = std::time(nullptr);
    auto tm = *std::localtime(&t);
    file << std::put_time(&tm, "%d.%m.%Y %H:%M:%S") << std::endl;
    file << grid_depth << 'x' << grid_width << 'x' << grid_height << std::endl;
    file << "seed: " << seed << std::endl;
    file << "total updates: " << total_updates << std::endl;

    double duration = (double) std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();
    double spin_updates_per_nanosecond = grid_depth * grid_width * grid_height / duration * 1e-3 * total_updates;
    printf("Total computing time: %f\n", duration * 1e-6);
    file << "total computing time: " << std::to_string(duration * 1e-6) << std::endl;
    printf("Updates per nanosecond: %f\n", spin_updates_per_nanosecond);
    file << "updates per nanosecond: " << std::to_string(spin_updates_per_nanosecond) << std::endl;
    file << "-----------------------------------" << std::endl;
    file.close();
    CHECK_CUDA(cudaDeviceSynchronize());

    write_lattice(d_black_tiles, d_white_tiles, ".data/", grid_width, grid_height, grid_depth, alpha, beta, j, global_market, seed, total_updates);
    return 0;
}
