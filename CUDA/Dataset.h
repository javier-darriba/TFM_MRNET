#ifndef CUFEAST_DATASET_H
#define CUFEAST_DATASET_H

#include <cmath>
#include <stdexcept>
#include <iostream>

template<typename T>
class Dataset {
public:
    /* Dataset operations */
    template<typename R, typename D>
    static void discretizeDataset(Dataset<R> &real_dat, Dataset<D> &disc_dat, unsigned int disc_bins = 0);

    Dataset() = default;

    inline void copy(T *raw_data, unsigned int num_feat, unsigned int num_samp, unsigned int num_clas = 1) {
        initAllocate(num_feat, num_samp, num_clas);
        for (unsigned int i = 0; i < numElements(); ++i) {
            this->raw_data[i] = raw_data[i];
        }
    }

    inline void copy(const Dataset<T> &dataset) {
        copy(dataset.raw_data, dataset.num_feat, dataset.num_samp, dataset.num_clas);
    }

    virtual ~Dataset() {
        if (raw_data != nullptr) {
            delete[] raw_data;
        }
    }

    inline unsigned int numElements() const { return (num_feat + num_clas) * num_samp; }

    inline std::size_t numBytes() const { return numElements() * sizeof(T); }

    virtual inline void allocate() {
        if (raw_data != nullptr) {
            throw std::runtime_error("Error: Dataset: Memory already allocated");
        }
        raw_data = new T[numElements()];
    }

    inline T *feature(unsigned int feature_id) const {
        if (feature_id >= num_feat) {
            throw std::invalid_argument("Error: Dataset: Feature '" + std::to_string(feature_id) + "' not in dataset");
        }
        return &raw_data[feature_id * num_samp];
    }

    inline void initAllocate(unsigned int num_feat, unsigned int num_samp, unsigned int num_clas) {
        this->num_feat = num_feat;
        this->num_samp = num_samp;
        this->num_clas = num_clas;
        allocate();
        this->classes = &raw_data[num_feat * num_samp];
    }

    inline void print() {
        for (int i = 0; i < num_feat + 1; ++i) {
            for (int j = 0; j < num_samp; ++j) {
                std::cout << raw_data[i * num_samp + j] << " ";
            }
            std::cout << std::endl;
        }
    }

public:
    unsigned int num_feat = 0;
    unsigned int num_samp = 0;
    unsigned int num_clas = 0;

    T *raw_data = nullptr;
    T *classes = nullptr;
};

template<typename T>                    //  T -> float
template<typename R, typename D>        //  R -> float   D -> uint
void Dataset<T>::discretizeDataset(Dataset<R> &real_dat, Dataset<D> &disc_dat, unsigned int disc_bins) {

    for (unsigned int f_i = 0; f_i < real_dat.num_feat; ++f_i) {

        R *real_data = &real_dat.raw_data[f_i * real_dat.num_samp];
        D *disc_data = &disc_dat.raw_data[f_i * real_dat.num_samp];

        // First detect the minimum and maximum values
        R min_val = real_data[0];
        R max_val = real_data[0];
        for (unsigned int i = 1; i < real_dat.num_samp; i++) {
            if (real_data[i] < min_val) {
                min_val = real_data[i];
            }
            if (real_data[i] > max_val) {
                max_val = real_data[i];
            }
        }

        double bin_size = (max_val - min_val) / ((double) disc_bins);
        for (unsigned int i = 0; i < real_dat.num_samp; i++) {
            disc_data[i] = (D) std::floor((real_data[i] - min_val) / bin_size);
        }
    }
}

#endif //CUFEAST_DATASET_H