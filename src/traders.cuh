/*
Initialise a given array of traders to contain values of either -1 or 1.
*/
__global__ void init_traders(signed char* traders,
                             const float* __restrict__ random_values,
                             const long long grid_height,
                             const long long grid_width);

__global__ void update_strategies(bool is_black,
                                  signed char* traders,
                                  const signed char* __restrict__ checkerboard_agents,
                                  const float* __restrict__ random_values,
                                  int *d_global_market,
                                  const float alpha,
                                  const float beta,
                                  const float j,
                                  const long long grid_height,
                                  const long long grid_width);

void update(signed char *d_black_tiles, signed char *d_white_tiles,
            float* random_values,
            curandGenerator_t rng,
            int *d_global_market,
            float alpha, float beta, float j,
            long long grid_height, long long grid_width);
