#include <mpi.h>

#include <chrono>
#include <iostream>
#include <vector>

#include "HDFUtils.h"
#include "calculations_cpu.h"
#include "calculations_gpu.cuh"

const int ROOT = 0;

void readVectorsFromH5File(H5FileReader &h5FileReader, std::vector<int> &data,
                           int &n, int &d) {
	auto vectorDataset = h5FileReader.getDataset(VECTORS_DATASET_NAME);

	n = H5FileReader::getIntAttr(vectorDataset, N_ATTR_NAME);
	d = H5FileReader::getIntAttr(vectorDataset, D_ATTR_NAME);

	data.resize(n * d);

	H5FileReader::read(vectorDataset, data);
	vectorDataset.close();
}

void writeResIntoH5File(H5FileReader &h5FileReader,
                        std::vector<double> &lengths, double durationMcs) {
	DataSet lengthsDataset =
	    h5FileReader.createDataset(LENGTHS_DATASET_NAME, lengths.size());
	H5FileReader::write(lengthsDataset, lengths);
	H5FileReader::setAttr(h5FileReader.getFile(), EXEC_TIME_NAME, durationMcs);

	lengthsDataset.close();
}

size_t getDeviceInfo(bool verbose = true) {
	cudaDeviceProp prop{};
	cudaGetDeviceProperties(&prop, 0);

	size_t maxBlocks1D = prop.maxGridSize[0];

	if (verbose) {
		std::cout << "Device " << 0 << ": " << prop.name << std::endl;
		std::cout << "Max threads per block: " << prop.maxThreadsPerBlock
		          << std::endl;
		std::cout << "Max block dimensions: (" << prop.maxThreadsDim[0] << ", "
		          << prop.maxThreadsDim[1] << ", " << prop.maxThreadsDim[2]
		          << ")" << std::endl;
		std::cout << "Max grid size: (" << prop.maxGridSize[0] << ", "
		          << prop.maxGridSize[1] << ", " << prop.maxGridSize[2] << ")"
		          << std::endl;

		std::cout << "Max block count (1D): " << maxBlocks1D << std::endl;
		std::cout << std::endl;
	}

	return maxBlocks1D;
}

int main(int argc, char *argv[]) {
	if (argc != 4) {
		throw std::runtime_error(
		    "You MUST specify data path, cuda block size and gpu data "
		    "precentage!");
	}

	MPI_Init(&argc, &argv);

	int worldSize;
	MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	std::vector<int> data;
	int numVectors = 0;
	int dimVectors = 0;
	int gpuVectorsCount = 0;
	int cpuVectorsCount = 0;

	H5FileReader h5FileReader;
	if (rank == ROOT) {
		std::cout << "Reading data..." << std::endl;
		std::string data_path = argv[1];

		h5FileReader = H5FileReader(data_path);
		readVectorsFromH5File(h5FileReader, data, numVectors, dimVectors);

		std::cout << "number of vectors: " << numVectors << std::endl;
		std::cout << "dim of vectors: " << dimVectors << std::endl;

		int gpuDataPercentage = std::stoi(argv[3]);

		if (worldSize > 1) {
			gpuVectorsCount = static_cast<int>(
			    std::ceil((gpuDataPercentage / 100.0) * numVectors));
			cpuVectorsCount = numVectors - gpuVectorsCount;
		} else {
			gpuVectorsCount = numVectors;
			cpuVectorsCount = 0;
		}

		std::cout << "gpu vectors: " << gpuVectorsCount << std::endl;
		std::cout << "cpu vectors: " << cpuVectorsCount << std::endl;
	}

	std::chrono::steady_clock::time_point startTime =
	    std::chrono::steady_clock::now();

	MPI_Bcast(&numVectors, 1, MPI_INT, ROOT, MPI_COMM_WORLD);
	MPI_Bcast(&dimVectors, 1, MPI_INT, ROOT, MPI_COMM_WORLD);
	MPI_Bcast(&gpuVectorsCount, 1, MPI_INT, ROOT, MPI_COMM_WORLD);
	MPI_Bcast(&cpuVectorsCount, 1, MPI_INT, ROOT, MPI_COMM_WORLD);

	int cpuRemainingVectors = cpuVectorsCount % (worldSize - 1);
	int cpuVectorsPerProcess = cpuVectorsCount / (worldSize - 1);

	int numLocalVectors = cpuVectorsPerProcess +
	                      (rank <= cpuRemainingVectors && rank != 0 ? 1 : 0);

	if (rank == ROOT) {
		numLocalVectors = gpuVectorsCount;
	}

	std::vector<int> localVectors(numLocalVectors * dimVectors);

	std::vector<int> sendCounts(worldSize, cpuVectorsPerProcess * dimVectors);
	sendCounts[ROOT] = gpuVectorsCount * dimVectors;

	std::vector<int> displacements(worldSize, 0);

	for (int i = 1; i <= cpuRemainingVectors; ++i) {
		sendCounts[i] += dimVectors;
	}

	for (int i = 1; i < worldSize; ++i) {
		displacements[i] = displacements[i - 1] + sendCounts[i - 1];
	}

	if (rank == 0) {
		std::cout << "Start calculation..." << std::endl;

		MPI_Scatterv(data.data(), sendCounts.data(), displacements.data(),
		             MPI_INT, localVectors.data(), localVectors.size(), MPI_INT,
		             ROOT, MPI_COMM_WORLD);
	} else {
		std::cout << "non root proccesses: " << numLocalVectors << std::endl;
		MPI_Scatterv(nullptr, nullptr, nullptr, MPI_DATATYPE_NULL,
		             localVectors.data(), localVectors.size(), MPI_INT, ROOT,
		             MPI_COMM_WORLD);
	}

	data.clear();

	std::vector<double> localLengths(numLocalVectors);
	if (rank == ROOT) {
		std::cout << "gpu rank: " << rank << std::endl;
		std::cout << numLocalVectors << " " << dimVectors << std::endl;
		auto maxBlockCount = getDeviceInfo(false);
		int cudaBlockSize = std::stoi(argv[2]);
		calculate_lengths_gpu(numLocalVectors, dimVectors, localVectors.data(),
		                      localLengths.data(), cudaBlockSize,
		                      maxBlockCount);
	} else {
		calculate_lengths_cpu(numLocalVectors, dimVectors, localVectors.data(),
		                      localLengths.data());
	}

	std::vector<double> lengths;
	if (rank == ROOT) {
		lengths.resize(numVectors);
	}

	sendCounts.clear();
	displacements.clear();

	sendCounts.resize(worldSize, cpuVectorsPerProcess);
	sendCounts[ROOT] = gpuVectorsCount;
	displacements.resize(worldSize, 0);

	for (int i = 1; i <= cpuRemainingVectors; ++i) {
		sendCounts[i] += 1;
	}

	for (int i = 1; i < worldSize; ++i) {
		displacements[i] = displacements[i - 1] + sendCounts[i - 1];
	}

	MPI_Gatherv(localLengths.data(), numLocalVectors, MPI_DOUBLE,
	            lengths.data(), sendCounts.data(), displacements.data(),
	            MPI_DOUBLE, ROOT, MPI_COMM_WORLD);

	localLengths.clear();

	if (rank == ROOT) {
		auto endTime = std::chrono::steady_clock::now();
		auto diffTime = std::chrono::duration_cast<std::chrono::microseconds>(
			endTime - startTime);

		std::cout << "Calculation time: " << diffTime << std::endl;
		std::cout << "Result has been saved." << std::endl;

		writeResIntoH5File(h5FileReader, lengths, diffTime.count());
	}

	MPI_Finalize();

	return 0;
	}
