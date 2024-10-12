#include <memory>
#include <boost/asio/ip/tcp.hpp>
#include <boost/asio/streambuf.hpp>
#include <boost/asio/read_until.hpp>
#include <iostream>
#include <future>
#include "calculations.h"

using boost::asio::ip::tcp;

const char DELIMITER = '\n';
const int RESULT_REQUEST = 0;
const int INPUT_REQUEST = 1;

class Session : public std::enable_shared_from_this<Session> {
public:
    explicit Session(tcp::socket socket);

    void run();

private:
    void read_first();

    void read_shape();

    void read_vectors();

    inline int bytes_to_int(auto bytes) {
        return (bytes[0] & 0xFF)
               | (bytes[1] << 8) & 0xFF00
               | (bytes[2] << 16) & 0xFF0000
               | (bytes[3] << 24);
    };

    tcp::socket socket_;
    boost::asio::streambuf buffer_;
    std::future<std::vector<double>> future_;
    std::vector<int> vectors_;
    int num_vec_;
    int dim_vec_;
};
