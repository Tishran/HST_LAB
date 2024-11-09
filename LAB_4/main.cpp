#include <vector>
#include <chrono>
#include <iostream>
#include "HDFUtils.h"
#include "calculations.cuh"

void readVectorsFromH5File(H5FileReader &h5FileReader, std::vector<int> &data, int &n, int &d) {
    auto vectorDataset = h5FileReader.getDataset(VECTORS_DATASET_NAME);

    n = H5FileReader::getIntAttr(vectorDataset, N_ATTR_NAME);
    d = H5FileReader::getIntAttr(vectorDataset, D_ATTR_NAME);

    data.resize(n * d);

    H5FileReader::read(vectorDataset, data);
    vectorDataset.close();
}

void writeResIntoH5File(H5FileReader &h5FileReader, std::vector<double> &lengths, double durationMcs) {
    DataSet lengthsDataset = h5FileReader.createDataset(LENGTHS_DATASET_NAME, lengths.size());
    H5FileReader::write(lengthsDataset, lengths);
    H5FileReader::setAttr(h5FileReader.getFile(), EXEC_TIME_NAME, durationMcs);

    lengthsDataset.close();
}

size_t getDeviceInfo(bool verbose = true) {
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, 0);

    std::cout << "Device " << 0 << ": " << prop.name << std::endl;
    std::cout << "Max threads per block: " << prop.maxThreadsPerBlock << std::endl;
    std::cout << "Max block dimensions: ("
              << prop.maxThreadsDim[0] << ", "
              << prop.maxThreadsDim[1] << ", "
              << prop.maxThreadsDim[2] << ")" << std::endl;
    std::cout << "Max grid size: ("
              << prop.maxGridSize[0] << ", "
              << prop.maxGridSize[1] << ", "
              << prop.maxGridSize[2] << ")" << std::endl;

    size_t maxBlocks1D = prop.maxGridSize[0];
    std::cout << "Max block count (1D): " << maxBlocks1D << std::endl;
    std::cout << std::endl;

    return maxBlocks1D;
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        throw std::runtime_error("You MUST specify data path and cuda block size!");
    }

    auto maxBlockCount = getDeviceInfo();

    std::vector<int> data;
    int numVectors = 0;
    int dimVectors = 0;
    std::string data_path = argv[1];
    int cudaBlockSize = std::stoi(argv[2]);

    std::cout << "Reading data..." << std::endl;
    H5FileReader h5FileReader = H5FileReader(data_path);
    readVectorsFromH5File(h5FileReader, data, numVectors, dimVectors);

    std::cout << "Start calculation" << std::endl;

    std::vector<double> lengths(numVectors);

    auto startTime = std::chrono::steady_clock::now();
    calculate_lengths(numVectors, dimVectors, data.data(), lengths.data(), cudaBlockSize, maxBlockCount);
    auto endTime = std::chrono::steady_clock::now();

    auto diffTime = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);
    std::cout << "Calculation time: " << diffTime << std::endl;

    writeResIntoH5File(h5FileReader, lengths, diffTime.count());
    std::cout << "Result has been saved." << std::endl;

    return 0;
}
