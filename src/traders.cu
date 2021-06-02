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
    // iterate over all traders in parallel and assign each of them
    // a strategy of either +1 or -1
    const long long  thread_id = static_cast<long long>(blockDim.x) * blockIdx.x + threadIdx.x;
    // check for out of bound access
    if (thread_id >= grid_width * grid_height) return;
    // use random number between 0.0 and 1.0 generated beforehand
    traders[thread_id] = (random_values[thread_id] < weight) ? -1 : 1;
}


template <bool is_black>
__global__ void update_strategies(signed char* traders,
                                  const signed char* __restrict__ checkerboard_agents,
                                  const float* __restrict__ random_values,
                                  //int *d_global_market,
                                  //const float alpha,
                                  const float beta,
                                  //const float j,
                                  const long long grid_height,
                                  const long long grid_width,
                                  const long long grid_depth)
{
    const long long thread_id = static_cast<long long>(blockDim.x) * blockIdx.x + threadIdx.x;
    const int row = thread_id / grid_width;
    const int col = thread_id % grid_width;

    // check for out of bound access
    if (row >= grid_height || col >= grid_width) return;

    // determine nearest neighbors on the opposite grid
    int lower_neighbor_row = (row + 1 < grid_height) ? row + 1 : 0;
    int upper_neighbor_row = (row - 1 >= 0) ? row - 1: grid_height - 1;
    int right_neighbor_col = (col + 1 < grid_width) ? col + 1 : 0;
    int left_neighbor_col = (col - 1 >= 0) ? col - 1: grid_width - 1;

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
    signed char neighbor_coupling = (
            checkerboard_agents[upper_neighbor_row * grid_width + col]
          + checkerboard_agents[lower_neighbor_row * grid_width + col]
          + checkerboard_agents[row * grid_width + col]
          + checkerboard_agents[row * grid_width + horizontal_neighbor_col]
          );

    //signed char old_strategy = traders[row * grid_width + col];
    //double market_coupling = -alpha / (grid_width * grid_height) * abs(d_global_market[0]);
    //double field = neighbor_coupling + market_coupling * old_strategy;
    // Determine whether to flip spin
    float probability = 1 / (1 + exp(-2.0f * beta * neighbor_coupling));
    signed char new_strategy = random_values[row * grid_width + col] < probability ? 1 : -1;
    traders[row * grid_width + col] = new_strategy;
    //__syncthreads();
    // If the strategy was changed remove the old value from the sum and add the new value.
    //if (new_strategy != old_strategy)
        //d_global_market[0] -= 2 * old_strategy;
}


void update(signed char *d_black_tiles,
            signed char *d_white_tiles,
            float* random_values,
            curandGenerator_t rng,
            //int *d_global_market,
            float beta,
            //float alpha, float beta, float j,
            long long grid_height, long long grid_width, long long grid_depth,
            int threads)
{
    int blocks = (grid_height * grid_width / 2 + threads - 1) / threads;

    CHECK_CURAND(curandGenerateUniform(rng, random_values, grid_height * grid_width / 2));
    update_strategies<true><<<blocks, threads>>>(d_black_tiles, d_white_tiles, random_values, beta, grid_height, grid_width / 2, grid_depth);

    CHECK_CURAND(curandGenerateUniform(rng, random_values, grid_height * grid_width / 2));
    update_strategies<false><<<blocks, threads>>>(d_white_tiles, d_black_tiles, random_values, beta, grid_height, grid_width / 2, grid_depth);
}
