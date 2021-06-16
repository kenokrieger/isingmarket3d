#include <iostream>
#include <fstream>

#include <cuda_fp16.h>
#include <curand.h>
#include <cublas_v2.h>
#include <cub/cub.cuh>

#include "cudamacro.h"


__global__ void fill_array(signed char* traders,
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


__global__ void add_array(const signed char* black_tiles, const signed char* white_tiles, signed char* result, const long long size);
    /*
    Add two arrays in parallel.

    Args:
        black_tiles: A pointer to one of the summands.
        white_tiles: A pointer to another summand.
        result: A pointer to the resulting array.
        size: The length of the array.
    */


__global__ void compute_probabilities(float* probabilities, const float market_coupling, const float reduced_j);
    /*
    Precompute possible probabilities for a spin flip and store them in the array
    "probabilities".

    Args:
        probabilities: A pointer to the array to fill.
        market_coupling: The relative magnetisation multiplied by -2 beta.
        reduced_j: The spin coupling multiplied by -2 beta.
    */


void init_traders(signed char* d_black_tiles, signed char* d_white_tiles,
                  curandGenerator_t rng, float* random_values,
                  long long grid_width, long long grid_height, long long grid_depth,
                  int threads = 64);
    /*
    Initialise two arrays of traders with randomly distributed values of +1 or -1.

    Args:
        d_black_tiles: One of the arrays to fill.
        d_white_tiles: The other array to fill.
        rng: The random number generator used for the distribution of values.
        random_values: Array containg/to be filled with random values.
        grid_height: The height of the grid.
        grid_width: The width of the grid.
        grid_depth: The depth of the grid.
        threads: The number of threads to invoke the kernel with. Defaults to 64.
    */


template <bool is_black>
__global__ void update_strategies(signed char* traders,
                                  const signed char* __restrict__ checkerboard_agents,
                                  const float* __restrict__ random_values,
                                  const float* probabilities,
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


int update(signed char *d_black_tiles,
           signed char *d_white_tiles,
           signed char *d_black_plus_white,
           float* random_values,
           float* d_probabilities,
           curandGenerator_t rng,
           const float reduced_alpha,
           const float reduced_j,
           const long long grid_height, const long long grid_width, const long long grid_depth,
           int threads = 16);

    /*
    Update all of the traders by updating the white and black tiles in succesion.

    Args:
        d_black_tiles: A pointer to the device array containg the black tiles.
        d_white_tiles: A pointer to the device array containing the white tiles.
        random_values: A device array containing/to be filled with random values.
        rng: The generator for the random numbers.
        global_market: The sum over all traders' strategies.
        alpha: A parameter controlling the strength of the market-coupling.
        beta: A parameter controlling the randomness. The greater beta the
              smaller the randomness.
        j: A parameter controlling the strength of the neighbor-coupling.
        grid_height: The height of the grid.
        grid_width: The width of the grid.
        grid_depth: The depth of the grid.

    Returns:
        The sum over all spins.
    */


void write_lattice(signed char *d_black_tiles,
                   signed char *d_white_tiles,
                   std::string fileprefix,
                   long long grid_width, long long grid_height, long long grid_depth,
                   float alpha, float beta, float j,
                   int global_market,
                   unsigned int seed,
                   int number_of_updates);
    /*
    Write the array to multiple files where each file contains one 2d slice of
    the whole array.

    Args:
        d_black_tiles: A pointer to the device array containg the black tiles.
        d_white_tiles: A pointer to the device array containing the white tiles.
        fileprefix: A prefix for the files to save, e.g. testrun_lattice_.
        grid_height: The height of the grid.
        grid_width: The width of the grid.
        grid_depth: The depth of the grid.
        alpha: A parameter controlling the strength of the market-coupling.
        beta: A parameter controlling the randomness. The greater beta the
              smaller the randomness.
        j: A parameter controlling the strength of the neighbor-coupling.
        global_market: The sum over all traders' strategies.
        seed: The seed of the random number generator used.
        number_of_updates: The total number of updates.
    */



void read_from_file(std::string fileprefix,
                    signed char* d_black_tiles,
                    signed char* d_white_tiles,
                    const long long grid_height, const long long grid_width, const long long grid_depth);
    /*
    Reads an existing lattice configuration from multiple files.

    Args:
        fileprefix: The prefix that all files share.
        d_black_tiles: A pointer to the device array to be filled with the values for the black tiles.
        d_white_tiles: A pointer to the device array to be filled with the values for the white tiles.
        grid_height: The height of the grid.
        grid_width: The width of the grid.
        grid_depth: The depth of the grid.
    */


int sum_array(const signed char* d_array, int size);
    /*
    Compute the total sum over a given array.

    Args:
        d_array: The pointer to the array on the device to compute the sum of.
        size: The number of items in the array.

    Returns:
        The value of all the elements summed.
    */
