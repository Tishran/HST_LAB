#include <iostream>
#include "Session.h"
#include <boost/asio.hpp>
#include <shared_mutex>

using namespace boost::asio;
using ip::tcp;
using std::string;
using std::cout;
using std::endl;

class TcpServer {
public:
    TcpServer(boost::asio::io_context &io_context, short port);

private:
    void do_accept();

    tcp::acceptor acceptor_;
//    std::unordered_map<int, std::future<std::vector<double>>> future_results_;
//    mutable std::shared_mutex mutex_;
};
