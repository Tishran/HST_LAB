#include <boost/asio/write.hpp>
#include "Session.h"

// TODO: refactor, add some logging


Session::Session(tcp::socket socket) : socket_(std::move(socket)),
                                       num_vec_(0),
                                       dim_vec_(0),
                                       buffer_(8192) {}

void Session::run() {
    read_first();
}

void Session::read_first() {
    auto self(shared_from_this());

    auto callback = [this, self](const boost::system::error_code &error, std::size_t bytes_transferred) {
        if (!error) {
            int request_type = bytes_to_int(boost::asio::buffers_begin(buffer_.data()));
            buffer_.consume(sizeof(int) + 1);

            if (request_type == INPUT_REQUEST) {
                read_shape();
            } else if (request_type == RESULT_REQUEST) {
                if (future_.valid()) {
                    auto res = future_.get();
                    boost::asio::write(socket_, boost::asio::buffer(res.data(),
                                                                    res.size() *
                                                                    sizeof(double)));

                    socket_.close();
                } else {
                    read_first();
                }
            }
        } else {
            std::cerr << "Error reading first: " << error.message() << std::endl;
        }
    };

    boost::asio::async_read_until(socket_, buffer_, DELIMITER, callback);
}

void Session::read_shape() {
    auto self(shared_from_this());

    auto callback = [this, self](const boost::system::error_code &error, std::size_t bytes_transferred) {
        if (!error) {
            num_vec_ = bytes_to_int(boost::asio::buffers_begin(buffer_.data()));
            buffer_.consume(sizeof(int));

            dim_vec_ = bytes_to_int(boost::asio::buffers_begin(buffer_.data()));
            buffer_.consume(sizeof(int) + 1);

            read_vectors();
        } else {
            std::cerr << "Error reading shape: " << error.message() << std::endl;
        }
    };

    boost::asio::async_read_until(socket_, buffer_, DELIMITER, callback);
}

void Session::read_vectors() {
    auto self(shared_from_this());

    auto callback = [this, self](const boost::system::error_code &error, std::size_t bytes_transferred) {
        if (!error) {
            while (buffer_.size() >= 4) {
                vectors_.push_back(bytes_to_int(boost::asio::buffers_begin(buffer_.data())));
                buffer_.consume(sizeof(int));

                // some logging TODO: come up with better logging strategy
                if (vectors_.size() == num_vec_ * dim_vec_) {
                    std::cout << "Received 100%" << std::endl;
                } else if (vectors_.size() == num_vec_ * dim_vec_ / 4 * 3) {
                    std::cout << "Received 75%" << std::endl;
                } else if (vectors_.size() == num_vec_ * dim_vec_ / 4 * 2) {
                    std::cout << "Received 50%" << std::endl;
                } else if (vectors_.size() == num_vec_ * dim_vec_ / 4) {
                    std::cout << "Received 25%" << std::endl;
                    std::cout << vectors_[0] << std::endl;
                }
            }

            if (vectors_.size() == num_vec_ * dim_vec_) {
                // TODO: чекнуть костыль ли
                if (buffer_.size() != 1) {
                    read_vectors();
                    return;
                }

                std::cout << "Starting computation" << std::endl;
                buffer_.consume(1);

                // sending accept here
                boost::asio::write(socket_, boost::asio::buffer({2}));

                std::promise<std::vector<double>> promise;
                future_ = promise.get_future();

                read_first();

                std::vector<double> lengths(num_vec_);

                auto start = std::chrono::steady_clock::now();
                calculate_lengths(num_vec_, dim_vec_, vectors_.data(), lengths.data());
                auto end = std::chrono::steady_clock::now();

                auto diff = end - start;

                lengths.push_back(std::chrono::duration_cast<std::chrono::microseconds>(
                        diff).count());

                promise.set_value(lengths);
            } else {
                read_vectors();
            }
        } else {
            std::cerr << "Error reading vectors: " << error.message() << std::endl;
        }
    };

    boost::asio::async_read_until(socket_, buffer_, DELIMITER, callback);
}
