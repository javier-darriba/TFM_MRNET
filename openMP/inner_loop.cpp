#include <cstdlib>
#include <cstring>
#include <cmath>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <iomanip>
#include <string>
#include <cfloat>
#include <algorithm>
#include <chrono>

#define BASE_TWO 2.0
#define LOG_BASE BASE_TWO
#define CALLOC_FUNC(a, b) calloc(a,b)
#define FREE_FUNC(a) free(a)

typedef unsigned int uint;

typedef struct {
    double *jointProbabilityVector;
    int numJointStates;
    double *firstProbabilityVector;
    int numFirstStates;
    double *secondProbabilityVector;
    int numSecondStates;
} JointProbabilityState;

void *checkedCalloc(size_t vectorLength, size_t sizeOfType) {
    void *allocated = CALLOC_FUNC(vectorLength, sizeOfType);
    if (allocated == NULL) {
        fprintf(stderr, "Error: %s\nAttempted to allocate %zu length of size %zu\n", strerror(errno), vectorLength,
                sizeOfType);
        exit(EXIT_FAILURE);
    }
    return allocated;
}

void freeJointProbabilityState(JointProbabilityState state) {
    FREE_FUNC(state.firstProbabilityVector);
    state.firstProbabilityVector = nullptr;
    FREE_FUNC(state.secondProbabilityVector);
    state.secondProbabilityVector = nullptr;
    FREE_FUNC(state.jointProbabilityVector);
    state.jointProbabilityVector = nullptr;
}

int maxState(uint *vector, int vectorLength) {
    int i, max;
    max = 0;
    for (i = 0; i < vectorLength; i++) {
        if (vector[i] > max) {
            max = vector[i];
        }
    }
    return max + 1;
}

JointProbabilityState calculateJointProbability(uint *firstVector, uint *secondVector, int vectorLength) {
    int *firstStateCounts;
    int *secondStateCounts;
    int *jointStateCounts;
    double *firstStateProbs;
    double *secondStateProbs;
    double *jointStateProbs;
    int firstNumStates;
    int secondNumStates;
    int jointNumStates;
    int i;
    double length = vectorLength;
    JointProbabilityState state;

    firstNumStates = maxState(firstVector, vectorLength);
    secondNumStates = maxState(secondVector, vectorLength);
    jointNumStates = firstNumStates * secondNumStates;

    firstStateCounts = (int *) checkedCalloc(firstNumStates, sizeof(int));
    secondStateCounts = (int *) checkedCalloc(secondNumStates, sizeof(int));
    jointStateCounts = (int *) checkedCalloc(jointNumStates, sizeof(int));

    firstStateProbs = (double *) checkedCalloc(firstNumStates, sizeof(double));
    secondStateProbs = (double *) checkedCalloc(secondNumStates, sizeof(double));
    jointStateProbs = (double *) checkedCalloc(jointNumStates, sizeof(double));

    /* Optimised for number of FP operations now O(states) instead of O(vectorLength) */
    for (i = 0; i < vectorLength; i++) {
        firstStateCounts[firstVector[i]] += 1;
        secondStateCounts[secondVector[i]] += 1;
        jointStateCounts[secondVector[i] * firstNumStates + firstVector[i]] += 1;
    }

    for (i = 0; i < firstNumStates; i++) {
        firstStateProbs[i] = firstStateCounts[i] / length;
    }

    for (i = 0; i < secondNumStates; i++) {
        secondStateProbs[i] = secondStateCounts[i] / length;
    }

    for (i = 0; i < jointNumStates; i++) {
        jointStateProbs[i] = jointStateCounts[i] / length;
    }

    FREE_FUNC(firstStateCounts);
    FREE_FUNC(secondStateCounts);
    FREE_FUNC(jointStateCounts);

    firstStateCounts = nullptr;
    secondStateCounts = nullptr;
    jointStateCounts = nullptr;

    state.jointProbabilityVector = jointStateProbs;
    state.numJointStates = jointNumStates;
    state.firstProbabilityVector = firstStateProbs;
    state.numFirstStates = firstNumStates;
    state.secondProbabilityVector = secondStateProbs;
    state.numSecondStates = secondNumStates;

    return state;
}/*calcJointProbability(uint *,uint *, int)*/

double mi(JointProbabilityState state) {
    double mutualInformation = 0.0;
    int firstIndex, secondIndex;
    int i;

    /*
    ** I(X;Y) = \sum_x \sum_y p(x,y) * \log (p(x,y)/p(x)p(y))
    */
    for (i = 0; i < state.numJointStates; i++) {
        firstIndex = i % state.numFirstStates;
        secondIndex = i / state.numFirstStates;

        if ((state.jointProbabilityVector[i] > 0) && (state.firstProbabilityVector[firstIndex] > 0) &&
            (state.secondProbabilityVector[secondIndex] > 0)) {
            /*double division is probably more stable than multiplying two small numbers together
            ** mutualInformation += state.jointProbabilityVector[i] * log(state.jointProbabilityVector[i] / (state.firstProbabilityVector[firstIndex] * state.secondProbabilityVector[secondIndex]));
            */
            mutualInformation += state.jointProbabilityVector[i] *
                                 log(state.jointProbabilityVector[i] / state.firstProbabilityVector[firstIndex] /
                                     state.secondProbabilityVector[secondIndex]);
        }
    }

    mutualInformation /= log(LOG_BASE);

    return mutualInformation;
}/*mi(JointProbabilityState)*/

double calcMutualInformation(uint *dataVector, uint *targetVector, int vectorLength) {
    JointProbabilityState state = calculateJointProbability(dataVector, targetVector, vectorLength);

    double mutualInformation = mi(state);

    freeJointProbabilityState(state);

    return mutualInformation;
}/*calculateMutualInformation(uint *,uint *,int)*/

void discretizeDataset(const std::vector<std::vector<float>> &real_dat, std::vector<std::vector<unsigned int>> &disc_dat,
                       unsigned int num_feat, unsigned int num_samp, unsigned int disc_bins) {

    disc_dat.resize(num_feat, std::vector<unsigned int>(num_samp));

    for (unsigned int f_i = 0; f_i < num_feat; ++f_i) {
        const std::vector<float> &real_data = real_dat[f_i];
        std::vector<unsigned int> &disc_data = disc_dat[f_i];

        // First detect the minimum and maximum values
        float min_val = *std::min_element(real_data.begin(), real_data.end());
        float max_val = *std::max_element(real_data.begin(), real_data.end());

        float bin_size = (max_val - min_val) / static_cast<float>(disc_bins);
        for (unsigned int i = 0; i < num_samp; ++i) {
            disc_data[i] = static_cast<unsigned int>(std::floor((real_data[i] - min_val) / bin_size));
        }
    }
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
    // Initialize noOfFeatures by counting the number of elements in the header
    noOfSamples = std::count_if(line.begin(), line.end(), [](unsigned char c) { return std::isspace(c); }) + 1;

    // Assuming the file has a consistent format where the number of samples equals the number of lines minus one (for the header)
    // Since we don't know the number of samples upfront, we'll temporarily store each feature's values in a vector of vectors
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

    // Now, transpose the data to fit the expected output format (features as columns, samples as rows)
    featureMatrix.resize(noOfFeatures, std::vector<float>(noOfSamples));
    for (int i = 0; i < noOfFeatures; ++i) {
        for (int j = 0; j < noOfSamples; ++j) {
            featureMatrix[i][j] = tempFeatureVectors[i][j];
        }
    }
}

uint *mRMR_D(uint k, uint noOfSamples, uint noOfFeatures, uint **featureMatrix, uint *classColumn, uint *outputFeatures,
             double *featureScores, int currentFeature) {
    /*holds the class MI values*/
    double *classMI = (double *) checkedCalloc(noOfFeatures, sizeof(double));
    char *selectedFeatures = (char *) checkedCalloc(noOfFeatures, sizeof(char));
    /*holds the intra feature MI values*/
    int sizeOfMatrix = k * noOfFeatures;
    double *featureMIMatrix = (double *) checkedCalloc(sizeOfMatrix, sizeof(double));

    /*Changed to ensure it always picks a feature*/
    double maxMI = -1.0;
    int maxMICounter = -1;

    /*init variables*/

    double score, currentScore, totalFeatureMI;
    int currentHighestFeature;

    int arrayPosition, i, j, x;

    for (i = 0; i < sizeOfMatrix; i++) {
        featureMIMatrix[i] = -1;
    }/*for featureMIMatrix - blank to -1*/

    for (i = 0; i < noOfFeatures; i++) {
        if (i != currentFeature) {
            classMI[i] = calcMutualInformation(featureMatrix[i], classColumn, noOfSamples);
            if (classMI[i] > maxMI) {
                maxMI = classMI[i];
                maxMICounter = i;
            }/*if bigger than current maximum*/
        }
    }/*for noOfFeatures - filling classMI*/

    selectedFeatures[maxMICounter] = 1;
    outputFeatures[0] = maxMICounter;
    featureScores[maxMICounter] = maxMI;

    /*************
     ** Now we have populated the classMI array, and selected the highest
     ** MI feature as the first output feature
     ** Now we move into the mRMR-D algorithm
     *************/

    for (i = 1; i < k; i++) {
        /****************************************************
        ** to ensure it selects some features
        **if this is zero then it will not pick features where the redundancy is greater than the
        **relevance
        ****************************************************/
        score = -DBL_MAX;
        currentHighestFeature = 0;
        currentScore = 0.0;
        totalFeatureMI = 0.0;

#pragma omp parallel for ordered private(x, currentScore, totalFeatureMI, arrayPosition)
        for (j = 0; j < noOfFeatures; j++) {
            /*if we haven't selected j*/
            if (j != currentFeature && selectedFeatures[j] == 0) {
                currentScore = classMI[j];
                totalFeatureMI = 0.0;

                for (x = 0; x < i; x++) {
                    arrayPosition = x * noOfFeatures + j;
                    if (featureMIMatrix[arrayPosition] == -1) {
                        /*work out intra MI*/
                        /*double calcMutualInformation(uint *firstVector, uint *secondVector, int vectorLength);*/
                        featureMIMatrix[arrayPosition] = calcMutualInformation(featureMatrix[outputFeatures[x]],
                                                                               featureMatrix[j], noOfSamples);
                    }

                    totalFeatureMI += featureMIMatrix[arrayPosition];
                }/*for the number of already selected features*/

                currentScore -= (totalFeatureMI / i);
#pragma omp critical
                {
                    if (currentScore > score) {
                        score = currentScore;
                        currentHighestFeature = j;
                    }
                }
            }/*if j is unselected*/
        }/*for number of features*/

        selectedFeatures[currentHighestFeature] = 1;
        outputFeatures[i] = currentHighestFeature;
        featureScores[currentHighestFeature] = score;

    }/*for the number of features to select*/

    FREE_FUNC(classMI);
    FREE_FUNC(featureMIMatrix);
    FREE_FUNC(selectedFeatures);

    classMI = NULL;
    featureMIMatrix = NULL;
    selectedFeatures = NULL;

    return outputFeatures;
}

int main(int argc, char *argv[]) {

    if (argc < 5) {
        std::cerr << "Usage: " << argv[0] << " -i <input_file_path> -o <output_file_name> [-t]\n";
        return EXIT_FAILURE;
    }

    std::string inputPath, outputFileName;
    bool timingEnabled = false;

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
            timingEnabled = true;
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

    long long total_mrmr_time = 0;
    std::chrono::high_resolution_clock::time_point start_program, end_program, start_read, end_read,
    start_disc, end_disc;
    if (timingEnabled) start_program = std::chrono::high_resolution_clock::now();

    int noOfSamples, noOfFeatures;
    std::vector<std::vector<float>> oriFeatureMatrix;

    if (timingEnabled) start_read = std::chrono::high_resolution_clock::now();
    read_txt(inputPath, noOfFeatures, noOfSamples, oriFeatureMatrix);
    if (timingEnabled) {
        end_read = std::chrono::high_resolution_clock::now();
        std::cout << "File Reading Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_read - start_read).count() << " ms\n";
        start_disc = std::chrono::high_resolution_clock::now();
    }

    std::vector<std::vector<unsigned int>> discFeatureMatrix;
    discretizeDataset(oriFeatureMatrix, discFeatureMatrix, noOfFeatures, noOfSamples, 128);

    auto *classColumn = new unsigned int[noOfSamples], **featureMatrix = new unsigned int *[noOfFeatures];
    auto *outputFeatures = new double[noOfFeatures], *featureScores = new double[noOfFeatures];

    for (int i = 0; i < noOfFeatures; ++i) {
        featureMatrix[i] = new uint[noOfSamples];
        for (int j = 0; j < noOfSamples; ++j) {
            featureMatrix[i][j] = discFeatureMatrix[i][j];
        }
    }

    if (timingEnabled) {
        end_disc = std::chrono::high_resolution_clock::now();
        std::cout << "Discretization Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_disc - start_disc).count() << " ms\n";
    }

    for (size_t i = 0; i < noOfFeatures; ++i) {
        uint *intOutputs = (uint *) checkedCalloc(noOfFeatures - 1, sizeof(uint));

        for (size_t j = 0; j < noOfSamples; ++j) {
            classColumn[j] = discFeatureMatrix[i][j];
        }

        std::chrono::high_resolution_clock::time_point start_mrmr, end_mrmr;
        if(timingEnabled) {
            start_mrmr = std::chrono::high_resolution_clock::now();
        }
        mRMR_D(noOfFeatures - 1, noOfSamples, noOfFeatures, featureMatrix, classColumn, intOutputs, featureScores, i);
        if(timingEnabled) {
            end_mrmr = std::chrono::high_resolution_clock::now();
            total_mrmr_time += std::chrono::duration_cast<std::chrono::milliseconds>(end_mrmr - start_mrmr).count();
        }

        for (int k = 0; k < noOfFeatures; ++k) {
            if (k != i) {
                outputFile << std::setprecision(15) << featureScores[k] << " ";
            } else {
                outputFile << "NA" << " ";
            }
        }
        outputFile << std::endl;

        FREE_FUNC(intOutputs);
    }

    for (int i = 0; i < noOfFeatures; ++i) {
        delete[] featureMatrix[i];
    }

    if (timingEnabled) {
        end_program = std::chrono::high_resolution_clock::now();
        std::cout << "Total mRMR_D Calculation Time: " << total_mrmr_time << " ms\n";
        std::cout << "Total Execution Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_program - start_program).count() << " ms\n";
    }

    delete[] classColumn;
    delete[] featureMatrix;
    delete[] outputFeatures;
    delete[] featureScores;

    return EXIT_SUCCESS;
}