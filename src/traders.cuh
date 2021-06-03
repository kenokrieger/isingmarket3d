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
                             float weight = 0.5f);
    /*
    Initialise a given array of traders to contain values of either -1 or 1.

    Args:
        traders: A pointer to the device array to fill.
        random_values: A device array of random float values between 0 and 1 which are
                       used to populate the traders array. Must have the same
                       dimensions as the traders array.
        grid_height: The height of the grid.
        grid_width: The width of the grid.
        grid_depth: The depth of the grid.
        weight: A float value between 0 and 1 that determines the relative
                amount of -1 spins.
    */


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
                                  const long long grid_depth);
    /*
    Update the strategy of each trader. The update utilises the metropolis
    algorithm where traders and their respective neighbors are updated
    seperately.

    template:
        is_black: Specifies which tile color on the checkerboard gets updated.

    Args:
        traders: A pointer to the device array of traders.
        checkerboard_agents: The device array containing the neighbors of the
                             traders.
        random_values: A device array containing random float values between 0
                       and 1. Must have the same dimensions as the traders array.
        d_global_market: A pointer to the device integer containing the value of
                         the sum over all traders.
        alpha: A parameter controlling the strength of the market-coupling.
        beta: A parameter controlling the randomness. The greater beta the
              smaller the randomness.
        j: A parameter controlling the strength of the neighbor-coupling.
        grid_height: The height of the grid.
        grid_width: The width of the grid.
        grid_depth: The depth of the grid.
    */


void update(signed char *d_black_tiles,
            signed char *d_white_tiles,
            float* random_values,
            curandGenerator_t rng,
            int *d_global_market,
            float alpha, float beta, float j,
            long long grid_height, long long grid_width, long long grid_depth,
            int threads = 128);

    /*
    Update all of the traders by updating the white and black tiles in succesion.

    Args:
        d_black_tiles: A pointer to the device array containg the black tiles.
        d_white_tiles: A pointer to the device array containing the white tiles.
        random_values: A device array containing/to be filled with random values.
        rng: The generator for the random numbers.
        d_global_market: A pointer to the device integer containing the value of
                         the sum over all traders.
        alpha: A parameter controlling the strength of the market-coupling.
        beta: A parameter controlling the randomness. The greater beta the
              smaller the randomness.
        j: A parameter controlling the strength of the neighbor-coupling.
        grid_height: The height of the grid.
        grid_width: The width of the grid.
        grid_depth: The depth of the grid.
    */
