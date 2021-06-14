#include <iostream>
#include <fstream>

#include <cuda_fp16.h>
#include <curand.h>
#include <cublas_v2.h>
#include <cub/cub.cuh>

#include "cudamacro.h"

#define CUB_CHUNK_SIZE ((1ll<<31) - (1ll<<28))


__global__ void fill_array(signed char* traders,
                           const float* __restrict__ random_values,
                           const long long grid_height,
                           const long long grid_width,
                           const long long grid_depth,
                           float weight = 0.5f)
{
    const int row = blockIdx.y;
    const int col = blockDim.x * blockIdx.x + threadIdx.x;
    const int lattice_id = blockIdx.z;
    // check for out of bound access
    if (row >= grid_height || col >= grid_width || lattice_id >= grid_depth) return;
    // use random number between 0.0 and 1.0 generated beforehand
    long long index = lattice_id * grid_width * grid_height + row * grid_width + col;
    traders[index] = (random_values[index] < weight) ? -1 : 1;
}


__global__ void add_array(const signed char* black_tiles,
                          const signed char* white_tiles,
                          signed char* result,
                          const long long size)
{
    int index = static_cast<long long>(blockDim.x) * blockIdx.x + threadIdx.x;
    if (index > size) return;
    result[index] = black_tiles[index] + white_tiles[index];
}


int sum_array(const signed char* d_arr, int size)
{
    // Reduce
    int* d_sum;
    int nchunks = (size + CUB_CHUNK_SIZE - 1)/ CUB_CHUNK_SIZE;
    CHECK_CUDA(cudaMalloc(&d_sum, nchunks * sizeof(*d_sum)));
    size_t temp_storage_bytes = 0;
    // When d_temp_storage is NULL, no work is done and the required allocation
    // size is returned in temp_storage_bytes.
    void* d_temp_storage = NULL;
    // determine temporary device storage requirements
    CHECK_CUDA(cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_arr, d_sum, CUB_CHUNK_SIZE));
    CHECK_CUDA(cudaMalloc(&d_temp_storage, temp_storage_bytes));

    for (int i = 0; i < nchunks; i++) {
        CHECK_CUDA(cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, &d_arr[i * CUB_CHUNK_SIZE], d_sum + i,
                             std::min((long long) CUB_CHUNK_SIZE, size - i * CUB_CHUNK_SIZE)));
    }

    int* h_sum;
    h_sum = (int*)malloc(nchunks * sizeof(*h_sum));
    CHECK_CUDA(cudaMemcpy(h_sum, d_sum, nchunks * sizeof(*d_sum), cudaMemcpyDeviceToHost));
    int total_sum = 0;

    for (int i = 0; i < nchunks; i++) {
      total_sum += h_sum[i];
    }
    CHECK_CUDA(cudaFree(d_sum));
    CHECK_CUDA(cudaFree(d_temp_storage));
    free(h_sum);
    return total_sum;
}


void init_traders(signed char* d_black_tiles, signed char* d_white_tiles,
                  curandGenerator_t rng, float* random_values,
                  long long grid_width, long long grid_height, long long grid_depth,
                  int threads = 64)
{
    dim3 blocks((grid_width / 2 + threads - 1) / threads, grid_height, grid_depth);
    CHECK_CURAND(curandGenerateUniform(rng, random_values, grid_depth * grid_height * grid_width / 2));
    fill_array<<<blocks, threads>>>(d_black_tiles, random_values, grid_height, grid_width / 2, grid_depth);
    CHECK_CURAND(curandGenerateUniform(rng, random_values, grid_depth * grid_height * grid_width / 2));
    fill_array<<<blocks, threads>>>(d_white_tiles, random_values, grid_height, grid_width / 2, grid_depth);
}


__global__ void compute_probabilities(float* probabilities, const int market_coupling, const float reduced_j) {
    int thread_id = threadIdx.x;
    // fill the array with values -6 - market_coupling, -5 - market_coupling, ..., 5 + market_coupling, 6 + market_coupling
    double field = reduced_j * (thread_id - 6 - (thread_id % 12)) + market_coupling * ((thread_id < 14) ? -1 : 1);
    probabilities[thread_id] = 1 / (1 + exp(field));
}


template <bool is_black>
__global__ void update_strategies(signed char* traders,
                                  const signed char* __restrict__ checkerboard_agents,
                                  const float* __restrict__ random_values,
                                  const float* probabilities,
                                  const long long grid_height,
                                  const long long grid_width,
                                  const long long grid_depth)
{
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int lattice_id = blockIdx.z;

    // check for out of bound access
    if (row >= grid_height || col >= grid_width || lattice_id >= grid_depth) return;

    long long index = lattice_id * grid_width * grid_height + row * grid_width + col;
    // determine nearest neighbors on the opposite grid
    int lower_neighbor_row = (row + 1 < grid_height) ? row + 1 : 0;
    int upper_neighbor_row = (row - 1 >= 0) ? row - 1: grid_height - 1;
    int right_neighbor_col = (col + 1 < grid_width) ? col + 1 : 0;
    int left_neighbor_col = (col - 1 >= 0) ? col - 1: grid_width - 1;
    int front_neighbor_lattice = (lattice_id - 1 >= 0) ? lattice_id - 1: grid_depth - 1;
    int back_neighbor_lattice = (lattice_id + 1 <= grid_depth - 1) ? lattice_id + 1: 0;

    // Select off-column index based on color and row index parity:
    // One of the neighbors will always have the exact same index
    // as the traders where as the remaining one will either have an
    // index differing by +1 or -1 depending on the position of the
    // agent on the grid
    int horizontal_neighbor_col;
    if (is_black) {
        horizontal_neighbor_col = (row % 2) ? left_neighbor_col : right_neighbor_col;
    } else {
        horizontal_neighbor_col = (row % 2) ? right_neighbor_col : left_neighbor_col;
    }
    // Compute sum of nearest neighbor spins
    signed char neighbor_sum =
            checkerboard_agents[lattice_id * grid_height * grid_width + upper_neighbor_row * grid_width + col]
          + checkerboard_agents[lattice_id * grid_height * grid_width + lower_neighbor_row * grid_width + col]
          + checkerboard_agents[index]
          + checkerboard_agents[lattice_id * grid_height * grid_width + row * grid_width + horizontal_neighbor_col]
          + checkerboard_agents[front_neighbor_lattice * grid_height * grid_width + row * grid_width + col]
          + checkerboard_agents[back_neighbor_lattice * grid_height * grid_width + row * grid_width + col];

    // use one of the 26 precomputed values for p = 1 / (1 + exp(-2 * beta ...))
    float probability = probabilities[13 * ((traders[index] < 0) ? 0 : 1) + neighbor_sum + 6];
    signed char new_strategy = random_values[index] < probability ? 1 : -1;
    traders[index] = new_strategy;
}


int update(signed char *d_black_tiles,
           signed char *d_white_tiles,
           signed char *d_black_plus_white,
           float* random_values,
           float* d_probabilities,
           curandGenerator_t rng,
           const float reduced_alpha,
           const float reduced_j,
           const long long grid_height, const long long grid_width, const long long grid_depth,
           int threads = 16)
{
    dim3 blocks(grid_width / threads, grid_width / threads, grid_depth);
    dim3 threads_per_block(threads / 2, threads);

    add_array<<<(grid_depth * grid_width / 2 * grid_height + 127) / 128, 128>>>(d_black_tiles, d_white_tiles, d_black_plus_white, grid_width / 2 * grid_height * grid_depth);
    int global_market = sum_array(d_black_plus_white, grid_depth * grid_height * grid_width / 2);
    int reduced_global_market = abs(global_market / (grid_width * grid_height * grid_depth));
    int market_coupling = -reduced_alpha * reduced_global_market;

    // precompute possible exponentials
    compute_probabilities<<<1, 26>>>(d_probabilities, market_coupling, reduced_j);
    CHECK_CURAND(curandGenerateUniform(rng, random_values, grid_depth * grid_height * grid_width / 2));
    update_strategies<true><<<blocks, threads_per_block>>>(d_black_tiles, d_white_tiles, random_values, d_probabilities, grid_height, grid_width / 2, grid_depth);

    CHECK_CURAND(curandGenerateUniform(rng, random_values, grid_depth * grid_height * grid_width / 2));
    update_strategies<false><<<blocks, threads_per_block>>>(d_white_tiles, d_black_tiles, random_values, d_probabilities, grid_height, grid_width / 2, grid_depth);

    return global_market;
}


void write_lattice(signed char *d_black_tiles,
                   signed char *d_white_tiles,
                   std::string fileprefix,
                   long long grid_width, long long grid_height, long long grid_depth,
                   float alpha, float beta, float j,
                   int global_market,
                   unsigned int seed,
                   int number_of_updates)
{
    signed char *h_black_tiles, *h_white_tiles;
    bool use_black;

    h_black_tiles = (signed char*)malloc(grid_depth * grid_height * grid_width / 2 * sizeof(*h_black_tiles));
    h_white_tiles = (signed char*)malloc(grid_depth * grid_height * grid_width / 2 * sizeof(*h_white_tiles));

    CHECK_CUDA(cudaMemcpy(h_white_tiles, d_white_tiles, grid_depth * grid_height * grid_width / 2 * sizeof(*d_white_tiles), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_black_tiles, d_black_tiles, grid_depth * grid_height * grid_width / 2 * sizeof(*d_black_tiles), cudaMemcpyDeviceToHost));

    for (int lattice_id = 0; lattice_id < grid_depth; lattice_id++) {
        std::string file_id = fileprefix + "lid=" + std::to_string(lattice_id) + ".dat";
        std::ofstream file;
        file.open(file_id);

        if (!file.is_open()) {
            printf("Could not write to file\n");
            return;
        }

        file << '#' << "lid = " << lattice_id << std::endl;
        file << '#' << "grid = " << grid_width << 'x' << grid_height << 'x' << grid_depth << std::endl;
        file << '#' << "beta = " << beta << std::endl;
        file << '#' << "alpha = " << alpha << std::endl;
        file << '#' << "j = " << j << std::endl;
        file << '#' << "market = " << global_market << std::endl;
        file << '#' << "seed = " << seed << std::endl;
        file << '#' << "total updates = " << number_of_updates << std::endl;

        for (int row = 0; row < grid_width; row++) {
            for (int col = 0; col < grid_height; col++) {
                    use_black = row % 2 == col % 2;
                    if (lattice_id % 2) use_black = !use_black;

                    if (use_black) {
                        file << (int)h_black_tiles[lattice_id * grid_height * grid_width / 2 + row * grid_width / 2 + col / 2] << " ";
                    } else {
                        file << (int)h_white_tiles[lattice_id * grid_height * grid_width / 2 + row * grid_width / 2 + col / 2] << " ";
                    }
            }
            file << std::endl;
        }
        file.close();
    }

    free(h_black_tiles);
    free(h_white_tiles);
}


void read_from_file(std::string fileprefix, signed char* d_black_tiles, signed char* d_white_tiles,
                    const long long grid_height, const long long grid_width, const long long grid_depth)
{
    std::ifstream file;
    std::string filename;
    signed char* h_black_tiles;
    signed char* h_white_tiles;

    h_black_tiles = (signed char*) malloc(grid_depth * grid_height * grid_width / 2 * sizeof(*h_black_tiles));
    h_white_tiles = (signed char*) malloc(grid_depth * grid_height * grid_width / 2 * sizeof(*h_white_tiles));

    std::string line = "";
    std::string tmp = "";
    bool use_black;

    for (int lattice_id = 0; lattice_id < grid_depth; lattice_id++) {
        filename = fileprefix + std::to_string(lattice_id) + ".dat";
        file.open(filename);
        if (!file.is_open()) {
            printf("Input file could not be read");
            return;
        }
        line = "";
        tmp = "";
        int row = 0;
        int col = 0;

        while (getline(file, line)) {

            if (line[0] == '#') continue;
            // add seperator to the end of the line
            col = 0;
            for (int idx = 0; idx < line.length(); idx++) {
                // checks for seperator character
                if (line[idx] != ' ' and line[idx] != '\n') {
                    tmp += line[idx];
                }
                else {
                    use_black = row % 2 == col % 2;
                    if (lattice_id % 2) use_black = !use_black;

                    if (use_black) {
                        h_black_tiles[lattice_id * grid_height * grid_width / 2 + row * grid_width / 2 + col / 2] = std::stoi(tmp);
                    } else {
                        h_white_tiles[lattice_id * grid_height * grid_width / 2 + row * grid_width / 2 + col / 2] = std::stoi(tmp);
                    }
                    tmp = "";
                    col++;
                }
            }
            row++;
        }
        file.close();
    }

    CHECK_CUDA(cudaMemcpy(d_black_tiles, h_black_tiles, grid_depth * grid_height * grid_width / 2 * sizeof(*h_black_tiles), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_white_tiles, h_white_tiles, grid_depth * grid_height * grid_width / 2 * sizeof(*h_white_tiles), cudaMemcpyHostToDevice));

    free(h_black_tiles);
    free(h_white_tiles);
}
