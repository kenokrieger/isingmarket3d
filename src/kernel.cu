#include <iostream>
#include <fstream>
#include <ctime>
#include <iomanip>
#include <chrono>

#include "traders.cuh"
#include "cudamacro.h"
#include "cudahelper.h"

#define timer std::chrono::high_resolution_clock

using namespace std;


map<string, string> read_config_file(string config_filename, string delimiter = "=")
{
    std::ifstream config_file;
    config_file.open(config_filename);
    map<string, string> config;

    if (!config_file.is_open()) {
        std::cout << "Could not open file '" << config_filename << "'" << std::endl;
        return config;
    } else {
        int row = 0;
        std::string line = "";
        std::string key = "";

        std::cout << "Launch configuration:" << std::endl;

        while (getline(config_file, line)) {
            if (line[0] == '#' || line == "") continue;
            int delimiter_position = line.find(delimiter);

            for (int idx = 0; idx < delimiter_position; idx++) {
                if (line[idx] != ' ') key += line[idx];
            }

            std::string value = line.substr(delimiter_position + 1, line.length() - 1);
            config[key] = value;
            std::cout << '\t' << key << ": ";
            std::cout << value << std::endl;
            row++;
            key = "";
        }
        config_file.close();
        return config;
    }
}


int main(int argc, char** argv) {

    std::ofstream file;
    signed char *d_black_tiles, *d_white_tiles, *d_black_plus_white;
    float *random_values, *d_probabilities;
    curandGenerator_t rng;
    // The global market represents the sum over the strategies of each
    // agent. Agents will choose a strategy contrary to the sign of the
    // global market.
    int device_id = 0;
    string config_filename = (argc == 1) ? "ising2d.conf" : argv[1];
    map<string, string> config = read_config_file(config_filename);

    //TODO ternary operator to replace with default arg if not passed.
    const long long grid_height = std::stoll(config["grid_height"]);
    const long long grid_width = std::stoll(config["grid_width"]);
    unsigned int total_updates = std::stoul(config["total_updates"]);
    unsigned int seed = std::stoul(config["seed"]);
    float alpha = std::stof(config["alpha"]);
    float j = std::stof(config["j"]);
    float beta = std::stof(config["beta"]);
    // the rng offset can be used to return the random number generator to a specific
    // state of a simulation. It is equal to the total number of random numbers
    // generated. Meaning the following equation holds for this specific case:
    // rng_offset = (total_updates + 1) * grid_width * grid_height
    // (+ 1 because of the random numbers created for the initilisation)
    unsigned long long rng_offset = (config["rng_offset"] != "") ? stoull(config["rng_offset"]) : 0;
    float reduced_alpha = -2.0f * beta * alpha;
    float reduced_j = -2.0f * beta * j;

    // finds and sets the specified cuda device
    gpuDeviceInit(device_id);
    // Finds and prints the devices name and computing power
    cudaDeviceProp deviceProp;
    deviceProp.major = 0;
    deviceProp.minor = 0;
    CHECK_CUDA(cudaGetDeviceProperties(&deviceProp, device_id));
    printf("CUDA device [%s] has %d Multi-Processors, Compute %d.%d\n",
        deviceProp.name, deviceProp.multiProcessorCount, deviceProp.major, deviceProp.minor);


    // Set up cuRAND generator
    CHECK_CURAND(curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_PHILOX4_32_10));
    CHECK_CURAND(curandSetPseudoRandomGeneratorSeed(rng, seed));
    CHECK_CURAND(curandSetGeneratorOffset(rng, rng_offset));

    // allocate memory for the arrays
    CHECK_CUDA(cudaMalloc(&d_white_tiles, grid_height * grid_width / 2 * sizeof(*d_white_tiles)));
    CHECK_CUDA(cudaMalloc(&d_black_tiles, grid_height * grid_width / 2 * sizeof(*d_black_tiles)));
    CHECK_CUDA(cudaMalloc(&d_black_plus_white, grid_height * grid_width / 2 * sizeof(*d_black_plus_white)));
    CHECK_CUDA(cudaMalloc(&random_values, grid_height * grid_width / 2 * sizeof(*random_values)));
    CHECK_CUDA(cudaMalloc(&d_probabilities, 10 * sizeof(*random_values)));

    init_traders(d_black_tiles, d_white_tiles, rng, random_values, grid_width, grid_height);
    // Synchronize operations on the GPU with CPU
    CHECK_CUDA(cudaDeviceSynchronize());


    file.open("magnetisation.dat");
    timer::time_point start = timer::now();
    for (int iteration = 0; iteration < total_updates; iteration++) {
        float global_market = update(d_black_tiles, d_white_tiles, d_black_plus_white, random_values,
                                     d_probabilities, rng, reduced_alpha, reduced_j, grid_height, grid_width);
        file << global_market << std::endl;
    }
    timer::time_point stop = timer::now();
    file.close();
    file.open("logs/ising.log", std::ios_base::app);
    auto t = std::time(nullptr);
    auto tm = *std::localtime(&t);
    file << std::put_time(&tm, "%d.%m.%Y %H:%M:%S") << std::endl;
    file << grid_width << 'x' << grid_height << std::endl;
    file << "seed: " << seed << std::endl;
    file << "total updates: " << total_updates << std::endl;

    double duration = (double) std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();
    double spin_updates_per_nanosecond = total_updates * grid_width * grid_height / duration * 1e-3;
    printf("Total computing time: %f\n", duration * 1e-6);
    file << "total computing time: " << std::to_string(duration * 1e-6) << std::endl;
    printf("Updates per nanosecond: %f\n", spin_updates_per_nanosecond);
    file << "updates per nanosecond: " << std::to_string(spin_updates_per_nanosecond) << std::endl;
    file << "-----------------------------------" << std::endl;
    file.close();
    file.open("log");
    file << "updates/ns: " << spin_updates_per_nanosecond << std::endl;
    file.close();
    CHECK_CUDA(cudaDeviceSynchronize());
    return 0;
}
