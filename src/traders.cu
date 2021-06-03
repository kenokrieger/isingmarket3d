#include <iostream>
#include <fstream>

#include <cuda_fp16.h>
#include <curand.h>
#include <cublas_v2.h>

#include "cudamacro.h"


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


template <bool is_black>
__global__ void update_strategies(signed char* traders,
                                  const signed char* __restrict__ checkerboard_agents,
                                  const float* __restrict__ random_values,
                                  int *d_global_market,
                                  const float alpha,
                                  const float beta,
                                  const float j,
                                  const long long grid_height,
                                  const long long grid_width,
                                  const long long grid_depth)
{
    const int row = blockIdx.y;
    const int col = blockDim.x * blockIdx.x + threadIdx.x;
    const int lattice_id = blockDim.z;

    // check for out of bound access
    if (row >= grid_height || col >= grid_width || lattice_id >= grid_depth) return;

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
    // Compute sum of nearest neighbor spins:
    // Multiply the row with the grid-width to receive
    // the actual index in the array
    signed char neighbor_coupling = j * (
            checkerboard_agents[lattice_id * grid_height * grid_width + upper_neighbor_row * grid_width + col]
          + checkerboard_agents[lattice_id * grid_height * grid_width + lower_neighbor_row * grid_width + col]
          + checkerboard_agents[lattice_id * grid_height * grid_width + row * grid_width + col]
          + checkerboard_agents[lattice_id * grid_height * grid_width + row * grid_width + horizontal_neighbor_col]
          + checkerboard_agents[front_neighbor_lattice * grid_height * grid_width + row * grid_width + col]
          + checkerboard_agents[back_neighbor_lattice * grid_height * grid_width + row * grid_width + col]
          );

    signed char old_strategy = traders[row * grid_width + col];
    double market_coupling = -alpha / (grid_width * grid_height) * abs(d_global_market[0]);
    double field = neighbor_coupling + market_coupling * old_strategy;
    // Determine whether to flip spin
    float probability = 1 / (1 + exp(-2.0f * beta * field));
    long long index = lattice_id * grid_width * grid_height + row * grid_width + col;
    signed char new_strategy = random_values[index] < probability ? 1 : -1;
    traders[index] = new_strategy;
    __syncthreads();
    // If the strategy was changed remove the old value from the sum and add the new value.
    if (new_strategy != old_strategy)
        d_global_market[0] -= 2 * old_strategy;
}


void update(signed char *d_black_tiles,
            signed char *d_white_tiles,
            float* random_values,
            curandGenerator_t rng,
            int *d_global_market,
            float alpha, float beta, float j,
            long long grid_height, long long grid_width, long long grid_depth,
            int threads = 64)
{
    dim3 blocks((grid_width / 2 + threads - 1) / threads, grid_height, grid_depth);

    CHECK_CURAND(curandGenerateUniform(rng, random_values, grid_depth * grid_height * grid_width / 2));
    update_strategies<true><<<blocks, threads>>>(d_black_tiles, d_white_tiles, random_values, d_global_market, alpha, beta, j, grid_height, grid_width / 2, grid_depth);

    CHECK_CURAND(curandGenerateUniform(rng, random_values, grid_depth * grid_height * grid_width / 2));
    update_strategies<false><<<blocks, threads>>>(d_white_tiles, d_black_tiles, random_values, d_global_market, alpha, beta, j, grid_height, grid_width / 2, grid_depth);
}


void write_lattice(signed char *d_black_tiles,
                   signed char *d_white_tiles,
                   std::string fileprefix,
                   long long grid_width, long long grid_height, long long grid_depth,
                   float alpha, float beta, float j,
                   int *d_global_market,
                   unsigned int seed,
                   int number_of_updates)
{
    signed char *h_black_tiles, *h_white_tiles;
    int *h_global_market;
    bool use_black;

    h_black_tiles = (signed char*)malloc(grid_depth * grid_height * grid_width / 2 * sizeof(*h_black_tiles));
    h_white_tiles = (signed char*)malloc(grid_depth * grid_height * grid_width / 2 * sizeof(*h_white_tiles));
    h_global_market = (int*)malloc(sizeof(*h_global_market));
    CHECK_CUDA(cudaMemcpy(h_global_market, d_global_market, sizeof(*d_global_market), cudaMemcpyDeviceToHost));
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
        file << '#' << "market = " << h_global_market[0] << std::endl;
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

    free(h_global_market);
    free(h_black_tiles);
    free(h_white_tiles);
}
