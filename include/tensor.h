#pragma once
#include <vector>
#include <cstddef>  // size_t

class Tensor {
    public:
        explicit Tensor(std::vector<size_t> shape);

        // size () const means it doesn't modify the object being called
        size_t size() const;  // total number of elements
        size_t ndim() const;  // number of dimensions
        const std::vector<size_t>& shape() const;  // shape of the tensor

        // operators
        float& operator()(size_t i, size_t j);  // modifiable tensors
        float operator()(size_t i, size_t j) const;  // read-only tensors

        // print method
        void print() const;

        // static functions: zeros, ones, random, matmul
        static Tensor zeros(std::vector<size_t> shape);
        static Tensor ones(std::vector<size_t> shape);
        static Tensor random(std::vector<size_t> shape);
        static Tensor matmul(const Tensor& a, const Tensor& b);

        //aritmetic operators
        friend Tensor operator+(const Tensor& a, const Tensor& b);
        friend Tensor operator-(const Tensor& a, const Tensor& b);
        friend Tensor operator*(const Tensor& a, const Tensor& b);


    private:
        std::vector<float> data_;
        std::vector<size_t> shape_;
        std::vector<size_t> strides_;
};