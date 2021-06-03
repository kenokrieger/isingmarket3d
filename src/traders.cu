#include <iostream>

#include <cuda_fp16.h>
#include <curand.h>
#include <cublas_v2.h>

#include "cudamacro.h"


__global__ void init_traders(signed char* traders,
                             const float* __restrict__ random_values,
                             const long long grid_height,
                             const long long grid_width,
                             const long long grid_depth,
                             float weight)
{
    const int row = static_cast<long long>(blockDim.x) * blockIdx.x + threadIdx.x;
    const int col = static_cast<long long>(blockDim.y) * blockIdx.y + threadIdx.y;
    const int lattice_id = static_cast<long long>(blockDim.z) * blockIdx.z + threadIdx.z;
    // check for out of bound access
    if (row >= grid_height || col >= grid_width || lattice_id >= grid_depth) return;
    // use random number between 0.0 and 1.0 generated beforehand
    long long index = lattice_id * grid_width * grid_height + row * grid_width + col;
    traders[index] = (random_values[index] < weight) ? -1 : 1;
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
    const int row = static_cast<long long>(blockDim.x) * blockIdx.x + threadIdx.x;
    const int col = static_cast<long long>(blockDim.y) * blockIdx.y + threadIdx.y;
    const int lattice_id = static_cast<long long>(blockDim.z) * blockIdx.z + threadIdx.z;

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
        horizontal_neighbor_col = (row % 2) ? right_neighbor_col : left_neighbor_col;
    } else {
        horizontal_neighbor_col = (row % 2) ? left_neighbor_col : right_neighbor_col;
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
            int threads)
{
    dim3 threads_per_block(threads, threads, threads);
    dim3 number_of_blocks((grid_height + threads_per_block.x -1) / threads_per_block.x,
                          (grid_width / 2 + threads_per_block.y - 1) / threads_per_block.y,
                          (grid_depth + threads_per_block.z - 1 / threads_per_block.z));

    CHECK_CURAND(curandGenerateUniform(rng, random_values, grid_depth * grid_height * grid_width / 2));
    update_strategies<true><<<number_of_blocks, threads_per_block>>>(d_black_tiles, d_white_tiles, random_values, d_global_market, alpha, beta, j, grid_height, grid_width / 2, grid_depth);

    CHECK_CURAND(curandGenerateUniform(rng, random_values, grid_depth * grid_height * grid_width / 2));
    update_strategies<false><<<number_of_blocks, threads_per_block>>>(d_white_tiles, d_black_tiles, random_values, d_global_market, alpha, beta, j, grid_height, grid_width / 2, grid_depth);
}
