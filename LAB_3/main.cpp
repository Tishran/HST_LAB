#include <vector>
#include <chrono>
#include <iostream>
#include "HDFUtils.h"
#include "calculations.h"
#include <mpi.h>

const int ROOT = 0;

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

int main(int argc, char *argv[]) {
    if (argc == 1) {
        throw std::runtime_error("You MUST specify data path!");
    }

    MPI_Init(&argc, &argv);

    int worldSize;
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    std::vector<int> data;
    int numVectors = 0;
    int dimVectors = 0;
    H5FileReader h5FileReader;
    if (rank == 0) {
        std::string data_path = argv[1];

        h5FileReader = H5FileReader(data_path);
        readVectorsFromH5File(h5FileReader, data, numVectors, dimVectors);
    }

    std::chrono::steady_clock::time_point startTime = std::chrono::steady_clock::now();

    MPI_Bcast(&numVectors, 1, MPI_INT, ROOT, MPI_COMM_WORLD);
    MPI_Bcast(&dimVectors, 1, MPI_INT, ROOT, MPI_COMM_WORLD);

    int remainingVectors = numVectors % worldSize;
    int vectorsPerProcess = numVectors / worldSize;

    int numLocalVectors = vectorsPerProcess + (rank < remainingVectors && rank != 0 ? 1 : 0);
    std::vector<int> localVectors(numLocalVectors * dimVectors);

    std::vector<int> sendCounts(worldSize, vectorsPerProcess * dimVectors);
    std::vector<int> displacements(worldSize, 0);

    for (int i = 0; i < remainingVectors; ++i) {
        sendCounts[i] += dimVectors;
    }

    for (int i = 1; i < worldSize; ++i) {
        displacements[i] = displacements[i - 1] + sendCounts[i - 1];
    }

    if (rank == 0) {
        std::cout << "Start calculation..." << std::endl;

        MPI_Scatterv(data.data(), sendCounts.data(), displacements.data(), MPI_INT,
                     localVectors.data(), localVectors.size(), MPI_INT,
                     ROOT, MPI_COMM_WORLD);
    } else {
        MPI_Scatterv(nullptr, nullptr, nullptr, MPI_DATATYPE_NULL,
                     localVectors.data(), localVectors.size(), MPI_INT,
                     ROOT, MPI_COMM_WORLD);
    }

    data.clear();

    std::vector<double> localLengths(numLocalVectors);
    calculate_lengths(numLocalVectors, dimVectors, localVectors.data(), localLengths.data());

    std::vector<double> lengths;
    if (rank == 0) {
        lengths.resize(numVectors);
    }

    sendCounts.clear();
    displacements.clear();

    sendCounts.resize(worldSize, vectorsPerProcess);
    displacements.resize(worldSize, 0);

    for (int i = 0; i < remainingVectors; ++i) {
        sendCounts[i] += 1;
    }

    for (int i = 1; i < worldSize; ++i) {
        displacements[i] = displacements[i - 1] + sendCounts[i - 1];
    }

    MPI_Gatherv(localLengths.data(), numLocalVectors, MPI_DOUBLE,
                lengths.data(), sendCounts.data(), displacements.data(), MPI_DOUBLE,
                ROOT, MPI_COMM_WORLD);

    localLengths.clear();

    if (rank == 0) {
        auto endTime = std::chrono::steady_clock::now();
        auto diffTime = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);

        std::cout << "Calculation time: " << diffTime << std::endl;
        std::cout << "Result has been saved." << std::endl;

        writeResIntoH5File(h5FileReader, lengths, diffTime.count());
    }

    MPI_Finalize();

    return 0;
}
