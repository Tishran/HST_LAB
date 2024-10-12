#include "TcpServer.h"

TcpServer::TcpServer(io_context &io_context, short port) : acceptor_(io_context, tcp::endpoint(tcp::v4(), port)) {
    do_accept();
}

void TcpServer::do_accept() {
    acceptor_.async_accept([this](boost::system::error_code ec, tcp::socket socket) {
        if (!ec) {
            std::make_shared<Session>(std::move(socket))->run();
        } else {
            std::cout << "error: " << ec.message() << std::endl;
        }
        do_accept();
    });
}


