#include <string>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <tuple>
#include <iomanip>
#include <string>
#include <cfloat>
#include <cuda_runtime.h>
#include <stdexcept>
#include <algorithm>
#include <chrono>
#include "FSToolbox.h"
#include "Dataset.h"
#include "Config.h"

#define CALLOC_FUNC(a, b) calloc(a,b)
#define FREE_FUNC(a) free(a)

typedef std::vector <std::pair<int, double>> Results;
size_t MAX_SHARED_MEMORY_PER_BLOCK = 32 << 10;

void *checkedCalloc(size_t vectorLength, size_t sizeOfType) {
    void *allocated = CALLOC_FUNC(vectorLength, sizeOfType);
    if (allocated == NULL) {
        fprintf(stderr, "Error: %s\nAttempted to allocate %zu length of size %zu\n", strerror(errno), vectorLength,
                sizeOfType);
        exit(EXIT_FAILURE);
    }
    return allocated;
}

void discretizeDataset(const std::vector<std::vector<float>> &real_dat, std::vector<std::vector<unsigned int>> &disc_dat,
                       unsigned int num_feat, unsigned int num_samp, unsigned int disc_bins) {

    disc_dat.resize(num_feat, std::vector<unsigned int>(num_samp));

    for (unsigned int f_i = 0; f_i < num_feat; ++f_i) {
        const std::vector<float> &real_data = real_dat[f_i];
        std::vector<unsigned int> &disc_data = disc_dat[f_i];

        // First detect the minimum and maximum values
        float min_val = *std::min_element(real_data.begin(), real_data.end());
        float max_val = *std::max_element(real_data.begin(), real_data.end());

        double bin_size = (max_val - min_val) / (double)(disc_bins);
        for (unsigned int i = 0; i < num_samp; ++i) {
            disc_data[i] = (uint) std::floor((real_data[i] - min_val) / bin_size);
        }
    }
}

__device__ uint16_t atomicAdd_block(uint16_t *address, uint16_t val) {

    auto address_as_u = (unsigned long int) address;
    auto value = (unsigned int) val;

    // We must assert we access memory in a 32bit fashion -> We have to use &[00] for both [00] or [01]
    auto *address_u32 = (unsigned int *) ((size_t) address & (~0b0011));

    if (address_as_u & 0b0010) {
        // "Lower"/First 2 bytes
        value = value << 16;
    } else {
        // "Upper"/Second 2 bytes
        value = value & 0xFFFF;
    }

    auto res = (unsigned int) atomicAdd_block(address_u32, value);

    if (address_as_u & 0b0010) {
        // "Lower"/First 2 bytes
        res = res >> 16;
    } else {
        // "Upper"/Second 2 bytes
    }

    return res;
}

void checkCudaError(unsigned int error) {
    if (error != cudaError::cudaSuccess) {
        auto e = static_cast<cudaError_t>(error);
        throw std::runtime_error(
                std::string(cudaGetErrorName(e)) + ": " + std::string(cudaGetErrorString(e)));
    }
}

void configureCudaConstants(int device) {
    cudaSetDevice(device);

    cudaDeviceProp properties{};
    cudaGetDeviceProperties(&properties, device);

    /* Set max memory size value */
    MAX_SHARED_MEMORY_PER_BLOCK = properties.sharedMemPerBlock;
}


__device__ void memSet(void *__restrict__ s, char c, size_t n) {

    for (size_t pos = threadIdx.x; pos < n; pos += blockDim.x) {
        static_cast<char *>(s)[pos] = c;
    }
    __syncthreads();

}

template<typename T>
__device__ T *globalOrShared(T *__restrict__ global, T *__restrict__ *shmem_end, size_t size) {
    T *result;

    if (global == nullptr) {
        result = *shmem_end;
        *shmem_end = reinterpret_cast<T *>((reinterpret_cast<char *>(*shmem_end)) + size);
    } else {
        result = global;
    }

    return result;
}

template<typename D>
__device__ void dataJointHistograms(const D *__restrict__ target_vector, const D *__restrict__ data_vector,
                                    uint16_t *__restrict__ data_histogram, uint16_t *__restrict__ joint_histogram,
                                    size_t vector_length, unsigned int hist_width) {
    D target_value, data_value;

    for (unsigned int pos = threadIdx.x; pos < vector_length; pos += blockDim.x) {
        target_value = target_vector[pos];
        data_value = data_vector[pos];

        atomicAdd_block(&data_histogram[data_value], 1);
        atomicAdd_block(&joint_histogram[target_value * hist_width + data_value], 1);
    }

    __syncthreads();
}

template<typename R>
__device__ void computeMI(const unsigned *__restrict__ target_histogram, const uint16_t *__restrict__ data_histogram,
                          const uint16_t *__restrict__ joint_histogram, uint16_t hist_width, uint16_t hist_height,
                          uint16_t hist_wpad, R vector_length_inv, R *__restrict__ mutual_information) {
    R partial_mi = 0;

    for (unsigned int col = threadIdx.x; col < hist_width; col += blockDim.x) {
        if (data_histogram[col] > 0) {
            R dSP_fI_dI = data_histogram[col] * vector_length_inv;

            for (unsigned int row = 0; row < hist_height; ++row) {
                unsigned int i = row * (hist_width + hist_wpad) + col;
                R jSP_fI_i = joint_histogram[i] * vector_length_inv;

                if ((jSP_fI_i > 0) && (target_histogram[row] > 0)) {
                    partial_mi += jSP_fI_i * log2((jSP_fI_i / dSP_fI_dI) / (target_histogram[row] * vector_length_inv));
                }
            }
        }
    }

    if (threadIdx.x < hist_width) {
        atomicAdd_block(mutual_information, partial_mi);
    }
}

template<typename D, typename R>
__global__ void k_hist_and_cMI_batch(D *device_data, unsigned int target_idx, unsigned int *data_idx_v,
                                     unsigned int vector_length, uint16_t **data_histogram_v, uint16_t *data_ns_v,
                                     unsigned int *target_histogram, uint16_t target_ns, uint16_t **joint_histogram_v,
                                     R *mutual_information, R vector_length_inv) {
    extern __shared__ uint16_t data[];
    const unsigned int data_idx = data_idx_v[blockIdx.x];
    const uint16_t data_ns = data_ns_v[blockIdx.x];

    // Step 1: Choose global or shared memory
    uint16_t *shmem_end = data;
    uint16_t *data_histogram = globalOrShared(data_histogram_v[blockIdx.x], &shmem_end, data_ns * sizeof(uint16_t));
    uint16_t *joint_histogram = globalOrShared(joint_histogram_v[blockIdx.x], &shmem_end,
                                               data_ns * target_ns * sizeof(uint16_t));
    // Reset shared memory
    memSet(data, 0, (shmem_end - data) * sizeof(uint16_t));

    // Step 2: Create histograms
    dataJointHistograms(&device_data[target_idx * vector_length], &device_data[data_idx * vector_length],
                        data_histogram, joint_histogram, vector_length, data_ns);

    // Step 3: Compute MI
    computeMI(target_histogram, data_histogram, joint_histogram, data_ns, target_ns, 0, vector_length_inv,
              &mutual_information[data_idx]);
}

template<typename T>
int maxState(const T *__restrict__ vector, int length) {
    T max = 0;

    for (int i = 0; i < length; i++) {
        if (vector[i] > max) {
            max = vector[i];
        }
    }

    return max + 1;
}

template<typename T>
void analyzeStates(const T *__restrict__ dataset, unsigned int num_features, unsigned int num_samples,
                   T *__restrict__ num_states_v, T *max_state, T *min_state) {
    T max_state_res = 0;
    T min_state_res = 2 << 16;

#pragma omp parallel for default(none) shared(dataset, num_features, num_samples, num_states_v) reduction(min: min_state_res) reduction(max: max_state_res)
    for (unsigned int i = 0; i < num_features; ++i) {
        num_states_v[i] = maxState(&dataset[i * num_samples], num_samples);

        if (num_states_v[i] > max_state_res) {
            max_state_res = num_states_v[i];
        }
        if (num_states_v[i] < min_state_res) {
            min_state_res = num_states_v[i];
        }
    }

    // states of class
    num_states_v[num_features] = maxState(&dataset[num_features * num_samples], num_samples);

    // store results
    *max_state = max_state_res;
    *min_state = min_state_res;
}

template<typename D, typename R>
__host__ void calcMutualInformationAll(D *host_data, D *device_data, unsigned int target_idx,
                                       unsigned int vector_length, unsigned int num_features, double *times,
                                       R *mutual_information, Config const &config, const bool *selected_features,
                                       unsigned int *num_states, unsigned int min_state, unsigned int max_state) {

    /***
     * Calc mutual information between feature *target_idx* and all the other
     ***/

    const unsigned int blockSize = config.getBlockSize();
    dim3 blockDim(blockSize);
    const R vector_length_inv = (R) 1.0 / vector_length;
    cudaError_t error;

    //auto timepoint = std::chrono::high_resolution_clock::now();
    //auto start = std::chrono::high_resolution_clock::now();

    /* Alloc memory for results */
    R *d_mutual_information;
    cudaMalloc(&d_mutual_information, num_features * sizeof(R));
    cudaMemset(d_mutual_information, 0, num_features * sizeof(R));

    /* Alloc and compute target feature states */
    uint16_t target_ns = num_states[target_idx];
    auto target_histogram = new unsigned int[target_ns]();
    unsigned int *d_target_histogram;
    cudaMalloc(&d_target_histogram, target_ns * sizeof(unsigned int));
    D *target_vector = &host_data[target_idx * vector_length];
    for (unsigned int i = 0; i < vector_length; i++) {
        target_histogram[target_vector[i]] += 1;
    }
    cudaMemcpy(d_target_histogram, target_histogram, target_ns * sizeof(unsigned int), cudaMemcpyHostToDevice);
    //cudaMemcpyToSymbol(d_target_histogram, target_histogram, target_ns * sizeof(double));

    /* Create streams */
    bool no_async = config.getNumStreams() == 0;
    unsigned int num_streams = (no_async) ? 1 : config.getNumStreams();
    unsigned int stream_id = 0;
    cudaStream_t streams[num_streams];
    for (cudaStream_t &s: streams) {
        cudaStreamCreate(&s);
    }

    /* Calculate batch sizes */
    // max sum of numStates per batch
    unsigned long int max_ns_sum = config.getBatchFeatures() * max_state;
    // max features per batch
    unsigned int max_features = (max_ns_sum + min_state - 1) / min_state;
    if (max_features > num_features) {
        max_features = num_features;
        max_ns_sum = max_features * max_state;
    }

    /* Setup memory pools */
    uint16_t *d_data_histogram_pool, *d_joint_histogram_pool;
    error = cudaMalloc(&d_joint_histogram_pool, max_ns_sum * target_ns * num_streams * sizeof(uint16_t));
    checkCudaError(error);
    error = cudaMalloc(&d_data_histogram_pool, max_ns_sum * num_streams * sizeof(uint16_t));
    checkCudaError(error);

    unsigned int *batch_feature_idx_pool;
    uint16_t *data_ns_pool;
    // pointers to histograms in device memory
    uint16_t **dptr_data_histogram_pool, **dptr_joint_histogram_pool;
    cudaMallocHost(&dptr_data_histogram_pool, max_features * num_streams * sizeof(uint16_t *));
    cudaMallocHost(&dptr_joint_histogram_pool, max_features * num_streams * sizeof(uint16_t *));
    cudaMallocHost(&batch_feature_idx_pool, max_features * num_streams * sizeof(unsigned int));
    cudaMallocHost(&data_ns_pool, max_features * num_streams * sizeof(uint16_t));

    // starting with a while loop because block factor could become dynamically adjustable
    // lets treat each iteration as a task, where we calculate the MI for *batch_features* features
    unsigned int feature_idx = 0;
    while (feature_idx < num_features) {

        // Setup stream and pointers (to the memory zone of this stream)
        unsigned int stream_pos = stream_id % num_streams;
        cudaStream_t &stream = streams[stream_pos];
        ++stream_id;

        unsigned int *batch_feature_idx_v = &batch_feature_idx_pool[stream_pos * max_features];
        uint16_t *data_ns_v = &data_ns_pool[stream_pos * max_features];

        uint16_t **dptr_data_histogram_v = &dptr_data_histogram_pool[stream_pos * max_features];
        uint16_t **dptr_joint_histogram_v = &dptr_joint_histogram_pool[stream_pos * max_features];
        // first pointer -> start of mem zone
        uint16_t *d_data_histogram_pool_start = &d_data_histogram_pool[stream_pos * max_ns_sum];
        uint16_t *d_joint_histogram_pool_start = &d_joint_histogram_pool[stream_pos * max_ns_sum * target_ns];

        // Get the next *batch_features* features that have no MI computed

        // MANDATORY vars of previous iter could have not been copied to GPU yet
        cudaStreamSynchronize(stream);
        // number of features in this task -> *batch_features* except for last task // TODO must be less than 65535
        unsigned int f_count = 0;
        unsigned int ns_sum = 0;
        while ((ns_sum < max_ns_sum) && (feature_idx < num_features)) {
            if (!selected_features[feature_idx]) {
                unsigned int dns = num_states[feature_idx];

                // assert max_dns is not surpassed
                if (ns_sum + dns > max_ns_sum) {
                    break;
                }

                batch_feature_idx_v[f_count] = feature_idx;
                data_ns_v[f_count] = dns;
                ns_sum += dns;

                ++f_count;
            }
            ++feature_idx;
        }
        if (!f_count) {
            continue;
        }

        /* Set global histogram offsets OR null to use shared memory
         * indexes = [0, 2, 3, 4]
         * ptr_histogram = [0x0, 0x10, nullptr, 0x20]
         *  feature 0 uses global histogram [0x0, ..., 0xf]
         *  feature 2 uses global histogram [0x10, ..., 0x1f]
         *  feature 3 uses shared mem histogram
         *  feature 4 uses global histogram [0x20, ..., 0x2f]
         * */
        size_t data_histogram_offset = 0, joint_histogram_offset = 0;
        bool use_shmem = config.useSharedMem();
        size_t sharedMem = 0;
        for (unsigned int f = 0; f < f_count; ++f) {
            unsigned int ns = data_ns_v[f];

            size_t datahist_size = ns * sizeof(uint16_t);
            size_t jointhist_size = ns * target_ns * sizeof(uint16_t);

            size_t needed_shmem = 0;
            if (use_shmem && (needed_shmem + jointhist_size < MAX_SHARED_MEMORY_PER_BLOCK)) {
                dptr_joint_histogram_v[f] = nullptr;
                needed_shmem += jointhist_size;
            } else {
                dptr_joint_histogram_v[f] = d_joint_histogram_pool_start + joint_histogram_offset;
                joint_histogram_offset += ns * target_ns;
            }
            if (use_shmem && (needed_shmem + datahist_size < MAX_SHARED_MEMORY_PER_BLOCK)) {
                dptr_data_histogram_v[f] = nullptr;
                needed_shmem += datahist_size;
            } else {
                dptr_data_histogram_v[f] = d_data_histogram_pool_start + data_histogram_offset;
                data_histogram_offset += ns;
            }

            // Find max needed shared memory
            sharedMem = (needed_shmem > sharedMem) ? needed_shmem : sharedMem;
        }

        // Transfer data and reset memory
        cudaMemsetAsync(d_data_histogram_pool_start, 0, data_histogram_offset * sizeof(uint16_t));
        cudaMemsetAsync(d_joint_histogram_pool_start, 0, joint_histogram_offset * sizeof(uint16_t));

        // Launch kernel
        dim3 gridDim(f_count);
        k_hist_and_cMI_batch<D, R><<<gridDim, blockDim, sharedMem, stream>>>(device_data, target_idx,
                                                                             batch_feature_idx_v, vector_length,
                                                                             dptr_data_histogram_v, data_ns_v,
                                                                             d_target_histogram, target_ns,
                                                                             dptr_joint_histogram_v,
                                                                             d_mutual_information, vector_length_inv);

    }

    // Copy results back
    cudaDeviceSynchronize();
    checkCudaError(cudaGetLastError());
    cudaMemcpy(mutual_information, d_mutual_information, num_features * sizeof(R), cudaMemcpyDeviceToHost);

    // Free memory
    for (cudaStream_t &s: streams) {
        cudaStreamDestroy(s);
    }

    cudaFreeHost(data_ns_pool);
    cudaFreeHost(batch_feature_idx_pool);
    cudaFreeHost(dptr_data_histogram_pool);
    cudaFreeHost(dptr_joint_histogram_pool);
    cudaFree(d_data_histogram_pool);
    cudaFree(d_joint_histogram_pool);
    delete[] target_histogram;
    cudaFree(d_target_histogram);
    cudaFree(d_mutual_information);
}

double *flattenDoubleArray(double **sourceData, unsigned int num_feat, unsigned int num_samp) {
    unsigned int totalElements = num_feat * num_samp;

    double *flattenedData = new double[totalElements];

    for (unsigned int feat = 0; feat < num_feat; ++feat) {
        for (unsigned int samp = 0; samp < num_samp; ++samp) {
            flattenedData[feat * num_samp + samp] = sourceData[feat][samp];
        }
    }

    return flattenedData;
}

template<typename D, typename R>
void mrmr_base(unsigned int k, Dataset<D> const &dat, Results &results, Config &config, D *d_data) {

    /* Array for time measurement */
    double times[16] = {0};

    /* MI values */
    R *class_mi = new R[dat.num_feat]; // MI of each feature with class
    R *feature_mi_matrix = new R[k * dat.num_feat]; // MI of each feature with selected

    /* Selected features map */
    auto *selected_features = new bool[dat.num_feat]();

    /* Cache for maxState of all features and class */
    auto *max_states = new unsigned int[dat.num_feat + 1];
    unsigned int max_state = 0, min_state = 2 << 16;

    analyzeStates(dat.raw_data, dat.num_feat, dat.num_samp, max_states, &max_state, &min_state);

    /* Select first based only on relevance with class */
    calcMutualInformationAll<D, R>(dat.raw_data, d_data, dat.num_feat, dat.num_samp, dat.num_feat, times, class_mi,
                                   config, selected_features, max_states, min_state, max_state);

    selectFeature(dat.num_feat, selected_features, results, fsCriterionMIM, fs_criterion_params<R>(class_mi));

    for (unsigned int i = 1; i < k; i++) {
        // First use kernel, then sum cached values
        calcMutualInformationAll<D, R>(dat.raw_data, d_data, results[i - 1].first, dat.num_samp, dat.num_feat,
                                       times, &feature_mi_matrix[(i - 1) * dat.num_feat], config, selected_features,
                                       max_states, min_state, max_state);

        selectFeature(dat.num_feat, selected_features, results, fsCriterionMRMR,
                      fs_criterion_params<R>(class_mi, feature_mi_matrix, i));
    }

    delete[] max_states;
    delete[] selected_features;
    delete[] feature_mi_matrix;
    delete[] class_mi;
}

template<typename T>
std::tuple<float, float> mrmr(unsigned int k, Dataset<T> const &dataset, Results &results, Config &config) {

    // Set k to min(k, num_feat)
    if (k > dataset.num_feat) {
        k = dataset.num_feat;
    }

    configureCudaConstants(config.getGpuIndex());

    /* Profiling with events */
    cudaEvent_t start, stop, startMemOps, stopMemOps, startKernel, stopKernel;
    if (config.doProfile()) {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventCreate(&startMemOps);
        cudaEventCreate(&stopMemOps);
        cudaEventCreate(&startKernel);
        cudaEventCreate(&stopKernel);
        checkCudaError(cudaGetLastError());
    }

    float memOpsTime = 0.0f, kernelTime = 0.0f;

    if (config.doProfile()) {
        cudaEventRecord(startMemOps);
    }

    /* Alloc memory */
    T *d_data;
    cudaMalloc(&d_data, dataset.numBytes());

    /* Copy data host -> device */
    cudaMemcpy(d_data, dataset.raw_data, dataset.numBytes(), cudaMemcpyHostToDevice);

    if (config.doProfile()) {
        cudaEventRecord(stopMemOps);
        cudaEventSynchronize(stopMemOps);
        cudaEventElapsedTime(&memOpsTime, startMemOps, stopMemOps);
        checkCudaError(cudaGetLastError());
    }

    /* Launch kernel */
    if (config.doProfile()) {
        cudaEventRecord(startKernel);
    }

    // reset last error if any
    cudaGetLastError();
    if (config.getPrecision() == Config::DOUBLE) {
        mrmr_base<T, double>(k, dataset, results, config, d_data);
    } else if (config.getPrecision() == Config::SINGLE) {
        mrmr_base<T, float>(k, dataset, results, config, d_data);
    }

    if (config.doProfile()) {
        cudaEventRecord(stopKernel);
        cudaEventSynchronize(stopKernel);
        cudaEventElapsedTime(&kernelTime, startKernel, stopKernel);
        checkCudaError(cudaGetLastError());
    }

    /* Free memory */
    if (config.doProfile()) {
        cudaEventRecord(startMemOps);
    }

    cudaFree(d_data);

    if (config.doProfile()) {
        cudaEventRecord(stopMemOps);
        cudaEventSynchronize(stopMemOps);
        float memOpsTimeFree;
        cudaEventElapsedTime(&memOpsTimeFree, startMemOps, stopMemOps);
        memOpsTime += memOpsTimeFree;
        checkCudaError(cudaGetLastError());
    }

    /* Free events */
    if (config.doProfile()) {
        cudaEventDestroy(stop);
        cudaEventDestroy(start);
        cudaEventDestroy(startMemOps);
        cudaEventDestroy(stopMemOps);
        cudaEventDestroy(startKernel);
        cudaEventDestroy(stopKernel);
        checkCudaError(cudaGetLastError());
    }

    return std::make_tuple(memOpsTime, kernelTime);
}

void read_txt(const std::string &file_path, int &noOfFeatures, int &noOfSamples,
              std::vector<std::vector<float>> &featureMatrix) {
    std::ifstream file(file_path);
    if (!file.is_open()) {
        std::cerr << "Error al abrir el archivo" << std::endl;
        exit(EXIT_FAILURE);
    }

    std::string line;

    if (!std::getline(file, line)) {
        std::cerr << "Error al leer el encabezado del archivo" << std::endl;
        exit(EXIT_FAILURE);
    }
    std::istringstream issHeader(line);

    noOfSamples = std::count_if(line.begin(), line.end(), [](unsigned char c) { return std::isspace(c); }) + 1;

    std::vector<std::vector<float>> tempFeatureVectors;
    noOfFeatures = 0; // This will be determined as we read

    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string sampleID;
        std::string value_str;
        std::vector<float> values;

        iss >> sampleID;

        bool hasNull = false;

        while (iss >> value_str) {
            if (value_str == "null") {
                hasNull = true;
                break;
            }
            values.push_back(std::stof(value_str));
        }

        if (!hasNull) {
            tempFeatureVectors.push_back(values);
            noOfFeatures++;
        }
    }

    featureMatrix.resize(noOfFeatures, std::vector<float>(noOfSamples));
    for (int i = 0; i < noOfFeatures; ++i) {
        for (int j = 0; j < noOfSamples; ++j) {
            featureMatrix[i][j] = tempFeatureVectors[i][j];
        }
    }
}

int main(int argc, char *argv[]) {

    if (argc < 5) {
        std::cerr << "Usage: " << argv[0] << " -i <input_file_path> -o <output_file_name> [-t]\n";
        return EXIT_FAILURE;
    }

    std::string inputPath, outputFileName;
    Config config;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-i") {
            if (i + 1 < argc) {
                inputPath = argv[++i];
            } else {
                std::cerr << "-i option requires one argument." << std::endl;
                return EXIT_FAILURE;
            }
        } else if (arg == "-o") {
            if (i + 1 < argc) {
                outputFileName = argv[++i];
            } else {
                std::cerr << "-o option requires one argument." << std::endl;
                return EXIT_FAILURE;
            }
        } else if (arg == "-t") {
            config.setProfile(true);
        } else {
            std::cerr << "Unknown option: " << arg << "\n";
            return EXIT_FAILURE;
        }
    }

    if (inputPath.empty() || outputFileName.empty()) {
        std::cerr << "Both input file path (-i) and output file name (-o) must be specified.\n";
        return EXIT_FAILURE;
    }

    std::ofstream outputFile(outputFileName);
    if (!outputFile.is_open()) {
        std::cerr << "Error: Could not open output file\n";
        return EXIT_FAILURE;
    }

    std::chrono::high_resolution_clock::time_point start_program, end_program;
    if (config.doProfile()) start_program = std::chrono::high_resolution_clock::now();

    int noOfSamples, noOfFeatures;
    std::vector <std::vector<float>> oriFeatureMatrix;

    std::chrono::high_resolution_clock::time_point start_read, end_read;
    if (config.doProfile()) start_read = std::chrono::high_resolution_clock::now();
    read_txt(inputPath, noOfFeatures, noOfSamples, oriFeatureMatrix);
    if (config.doProfile()) {
        end_read = std::chrono::high_resolution_clock::now();
        std::cout << "File Reading Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_read - start_read).count() << " ms\n";
    }

    std::vector<std::vector<unsigned int>> discFeatureMatrix;
    discretizeDataset(oriFeatureMatrix, discFeatureMatrix, noOfFeatures, noOfSamples, 128);

    auto **featureMatrix = new unsigned int *[noOfFeatures];
    for (int i = 0; i < noOfFeatures; ++i) {
        featureMatrix[i] = new unsigned int[noOfSamples];
    }

    for (int i = 0; i < noOfFeatures; ++i) {
        for (int j = 0; j < noOfSamples; ++j) {
            featureMatrix[i][j] = discFeatureMatrix[i][j];
        }
    }

    std::chrono::high_resolution_clock::time_point start_mrmr, end_mrmr, start_sort, end_sort;
    double total_mrmr_time = 0.0, total_sort_time = 0.0;
    float totalMemOpsTime = 0.0f, totalKernelTime = 0.0f;

    for (int featureIndex = noOfFeatures; featureIndex > 0; --featureIndex) {
        Results results;
        std::vector <unsigned int> actualFeatureColumn(noOfSamples);
        std::vector <unsigned int> originalLastColumn(noOfSamples);
        std::vector <unsigned int> flatData;
        Dataset<unsigned int> discDataset;

        if (featureIndex != noOfFeatures) {
            for (int i = 0; i < noOfSamples; ++i) {
                actualFeatureColumn[i] = featureMatrix[featureIndex - 1][i];
                originalLastColumn[i] = featureMatrix[noOfFeatures - 1][i];
            }
            for (int i = 0; i < noOfSamples; ++i) {
                featureMatrix[featureIndex - 1][i] = originalLastColumn[i];
                featureMatrix[noOfFeatures - 1][i] = actualFeatureColumn[i];
            }
        }

        for (int i = 0; i < noOfFeatures; ++i) {
            for (int j = 0; j < noOfSamples; ++j) {
                flatData.push_back(featureMatrix[i][j]);
            }
        }

        discDataset.copy(flatData.data(), noOfFeatures - 1, noOfSamples);

        float memOpsTime = 0.0f, kernelTime = 0.0f;

        if (config.doProfile()) start_mrmr = std::chrono::high_resolution_clock::now();

        std::tie(memOpsTime, kernelTime) = mrmr<unsigned int>(noOfFeatures - 1, discDataset, results, config);

        totalMemOpsTime += memOpsTime;
        totalKernelTime += kernelTime;

        if (config.doProfile()) {
            end_mrmr = std::chrono::high_resolution_clock::now();
            total_mrmr_time += std::chrono::duration_cast<std::chrono::milliseconds>(end_mrmr - start_mrmr).count();
            start_sort = std::chrono::high_resolution_clock::now();
        }

        std::sort(results.begin(), results.end(), [](const auto &a, const auto &b) {
            return a.first < b.first;
        });

        if (config.doProfile()) {
            end_sort = std::chrono::high_resolution_clock::now();
            total_sort_time += std::chrono::duration_cast<std::chrono::milliseconds>(end_sort - start_sort).count();
        }

        for (auto &scorePair: results) {
            if (featureIndex == 1 && scorePair.first == 0) {
                outputFile << "NA ";
            }
            outputFile << std::setprecision(15) << scorePair.second << " ";
            if (scorePair.first == featureIndex - 2) {
                outputFile << "NA ";
            }
        }
        outputFile << std::endl;

    }

    if (config.doProfile()) {
        std::cout << "Total mRMR Calculation Time: " << total_mrmr_time << " ms\n";
        std::cout << "Memory Operations Time (Allocation, Transfer, Deallocation): " << totalMemOpsTime << " ms\n";
        std::cout << "Kernel Execution Time: " << totalKernelTime << " ms\n";
        std::cout << "sort Time: " << total_sort_time << " ms\n";
        end_program = std::chrono::high_resolution_clock::now();
        std::cout << "Total Execution Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_program - start_program).count() << " ms\n";
    }

    for (int i = 0; i < noOfFeatures; ++i) {
        delete[] featureMatrix[i];
    }
    delete[] featureMatrix;

    return EXIT_SUCCESS;
}