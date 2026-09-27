#include "tensor.h"
#include <stdexcept>  // for std::out_of_range
#include <iostream>
#include <random>

// constructor 
Tensor::Tensor(std::vector<size_t> shape)
    : shape_(std::move(shape))
{
    // stride and shape share size
    strides_.resize(shape_.size());
    size_t current_stride = 1;
    size_t total_elements = 1;
    
    for (int i = shape_.size(); i >= 1; i--){
        strides_[i-1] = current_stride;
        current_stride *= shape_[i-1];
        total_elements *= shape_[i-1];
    }

    data_.resize(total_elements, 0.0f);
}

size_t Tensor::size() const{
    return data_.size();
}

size_t Tensor::ndim() const{
    return shape_.size();
}

const std::vector<size_t>& Tensor::shape() const{
    return shape_;
}

float& Tensor::operator()(size_t i, size_t j){
    if (i < shape_[0] && j < shape_[1]){
        return data_[i * strides_[0] + j * strides_[1]];
    }
    else{
        throw std::out_of_range("Index out of bounds");
    }
}

float Tensor::operator()(size_t i, size_t j) const{
    if (i < shape_[0] && j < shape_[1]){
        return data_[i * strides_[0] + j * strides_[1]];
    }
    else{
        throw std::out_of_range("Index out of bounds");
    }
}

void Tensor::print() const{

    size_t dims = ndim();
    std::cout << "Tensor with shape : " << dims << "\n";

    // pretty print for 2-D tensors
    if (dims == 2){
        for (size_t i = 0; i < shape_[0]; i++){
            for (size_t j = 0; j < shape_[1]; j++){
                std::cout << (*this)(i, j) << " ";
            }
            std::cout << "\n";
        }
    }
    else{
        for (size_t i = 0; i < data_.size(); i++){
            std::cout << data_[i] << "\n";
        }
    }
}


Tensor Tensor::zeros(std::vector<size_t> shape){
    Tensor t(shape);
    return t;
}

Tensor Tensor::ones(std::vector<size_t> shape){
    Tensor t(shape);
    for (auto& v : t.data_) v = 1.0f;
    return t;
}

Tensor Tensor::random(std::vector<size_t> shape){
    Tensor t(shape);
    std::mt19937 generator;
    std::uniform_real_distribution<double> uni_dist(0.0, 1.0);

    for (auto& v : t.data_){
        v = uni_dist(generator);
    }

    return t;

}

Tensor operator+(const Tensor& a, const Tensor& b){

    if (a.shape() != b.shape()){
        throw std::out_of_range("Mismatch tensor shapes!");
    }

    Tensor c(a.shape());

    for (size_t i = 0; i < a.size(); i++) c.data_[i] = a.data_[i] + b.data_[i];
    
    return c;

}

Tensor operator-(const Tensor& a, const Tensor& b){

    if (a.shape() != b.shape()){
        throw std::out_of_range("Mismatch tensor shapes!");
    }

    Tensor c(a.shape());

    for (size_t i = 0; i < a.size(); i++) c.data_[i] = a.data_[i] - b.data_[i];
    
    return c;

}

Tensor operator*(const Tensor& a, const Tensor& b){

    if (a.shape() != b.shape()){
        throw std::out_of_range("Mismatch tensor shapes!");
    }

    Tensor c(a.shape());

    for (size_t i = 0; i < a.size(); i++) c.data_[i] = a.data_[i] * b.data_[i];
    
    return c;

}

Tensor Tensor::matmul(const Tensor& a, const Tensor& b){

    if (a.ndim() != 2 || b.ndim() != 2){
        throw std::out_of_range("Only 2D matmul is implemented!");
    }

    if (a.shape()[1] != b.shape()[0]){
        throw std::out_of_range("Columns of first tensor must match rows of second tensor!");
    }

    Tensor c({a.shape()[0], b.shape()[1]});

    float acum = 0.0;
    for (size_t i = 0; i < a.shape()[0]; i++){
        for (size_t j = 0; j < b.shape()[1]; j++){
            acum = 0.0;
            for (size_t k = 0; k < b.shape()[0]; k++){
                acum += a(i, k) * b(k, j);
            }
            c(i, j) = acum;
        }
    }

    return c;
}