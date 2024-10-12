#include <iostream>
#include <boost/asio/io_context.hpp>
#include "TcpServer.h"

int main() {
    boost::asio::io_context io_context;
    TcpServer server(io_context, 12345);
    io_context.run();

    return 0;
}
