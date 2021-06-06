
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
