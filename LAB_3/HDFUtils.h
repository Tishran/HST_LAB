#include <H5Cpp.h>
#include <vector>

const std::string VECTORS_DATASET_NAME = "vectors";
const std::string LENGTHS_DATASET_NAME = "lengths";
const std::string N_ATTR_NAME = "num_vectors";
const std::string D_ATTR_NAME = "dim_vectors";
const std::string EXEC_TIME_NAME = "execution_time";

using namespace H5;

class H5FileReader {
public:
    H5FileReader();

    explicit H5FileReader(const std::string &dataPath);

    ~H5FileReader();

    [[nodiscard]] const H5File &getFile() const;

    [[nodiscard]] DataSet getDataset(const std::string &datasetName) const;

    DataSet createDataset(const std::string &datasetName, hsize_t dataSize);

    static void read(const DataSet &dataset, std::vector<int> &data);

    static void write(const DataSet &dataset, const std::vector<double> &data);

    [[nodiscard]] static int getIntAttr(const H5Object &object, const std::string &attrName);

    static void setAttr(const H5Object &h5object, const std::string &attrName, double data);

private:
    H5File file;
};