#include "HDFUtils.h"

H5FileReader::H5FileReader() = default;

H5FileReader::H5FileReader(const std::string &dataPath) : file(dataPath, H5F_ACC_RDWR) {}

H5FileReader::~H5FileReader() {
    file.close();
}

const H5File &H5FileReader::getFile() const {
    return file;
}

DataSet H5FileReader::getDataset(const std::string &datasetName) const {
    return file.openDataSet(datasetName);
}

int H5FileReader::getIntAttr(const H5Object &object, const std::string &attrName) {
    Attribute attr = object.openAttribute(attrName);

    int attrVal = 0;
    attr.read(attr.getDataType(), &attrVal);

    return attrVal;
}

DataSet H5FileReader::createDataset(const std::string &datasetName, hsize_t dataSize) {
    hsize_t dims[1] = {dataSize};
    DataSpace dataspace(1, dims);

    return file.createDataSet(datasetName, PredType::NATIVE_DOUBLE, dataspace);
}

void H5FileReader::read(const DataSet &dataset, std::vector<int> &data) {
    DataType datatype = dataset.getDataType();
    dataset.read(static_cast<void *>(data.data()), datatype);
}

void H5FileReader::write(const DataSet &dataset, const std::vector<double> &data) {
    dataset.write(data.data(), PredType::NATIVE_DOUBLE);
}

void H5FileReader::setAttr(const H5Object &h5object, const std::string &attrName, double data) {
    DataSpace attrDataSpace(H5S_SCALAR);

    Attribute attr = h5object.createAttribute(attrName, PredType::NATIVE_DOUBLE, attrDataSpace);
    attr.write(PredType::NATIVE_DOUBLE, &data);
}
