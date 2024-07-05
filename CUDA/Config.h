#ifndef CUFEAST_CONFIG_H
#define CUFEAST_CONFIG_H

class Config {

public:
    enum Precision {
        DOUBLE, SINGLE, FIXED
    };

    explicit Config(unsigned int blockSize = 128, unsigned int numStreams = 1, unsigned int batchFeatures = 4096,
                    Precision precision = DOUBLE, bool sharedMemory = true, int gpuIndex = 0,
                    bool profile = false)
            : precision(precision),
              block_size(blockSize),
              num_streams(numStreams),
              batch_features(batchFeatures),
              shared_memory(sharedMemory),
              gpu_index(gpuIndex),
              profile(profile) {
    }

    unsigned int getBlockSize() const {
        return block_size;
    }

    unsigned int getNumStreams() const {
        return num_streams;
    }

    unsigned int getBatchFeatures() const {
        return batch_features;
    }

    Precision getPrecision() const {
        return precision;
    }

    bool useSharedMem() const {
        return shared_memory;
    }

    bool doProfile() const {
        return profile;
    }

    int getGpuIndex() const {
        return gpu_index;
    }

    std::vector<std::pair<float, std::string>> getProfileData() const {
        return this->profile_data;
    }

    void setBlockSize(unsigned int block_size) {
        this->block_size = block_size;
    }

    void setNumStreams(unsigned int num_streams) {
        this->num_streams = num_streams;
    }

    void setBatchFeatures(unsigned int batch_features) {
        this->batch_features = batch_features;
    }

    void setPrecision(Precision precision) {
        this->precision = precision;
    }

    void setSharedMem(bool shared_memory) {
        this->shared_memory = shared_memory;
    }

    void setProfile(bool profile) {
        this->profile = profile;
    }

    void setGpuIndex(int gpu_index) {
        this->gpu_index = gpu_index;
    }

    void appendProfileEntry(double ms, std::string what) {
        this->profile_data.emplace_back(ms, what);
    }

    void clearProfileEntries() {
        this->profile_data.clear();
    }

private:
    Precision precision = DOUBLE;

    unsigned int block_size = 128;
    unsigned int num_streams = 1;
    unsigned int batch_features = 4096; // magic number -> works well in gtx1650, T4
    bool shared_memory = true;

    int gpu_index;
    bool profile = false;
    std::vector<std::pair<float, std::string>> profile_data;
};

#endif //CUFEAST_CONFIG_H