// fms_perceptron.h
// A perceptron is a hyperplane separating two sets of points in R^n.
// Given sets S_0 and S_1, find a vector w and a scalar b such that
// w.x < 0 for x in S_0 and w.x > 0 for x in S_1.
#pragma once

#include <span>
#include <vector>
#include "fms_error.h"
#include "fms_linalg.h"
// https://cppreference.net/cpp/numeric/linalg.html
// #include <linalg>

namespace fms::perceptron {

    // Update weights w given point x and label y in {true, false}
    // Return if the update trained the data.
    template<class T = double>
    constexpr bool update(std::size_t n, T* w, const T* x, bool y, T alpha = 1.0)
    {
		// changed int y to bool y so we don't need this check
        //ensure (y == 0 or y == 1 || !"label must be 0 or 1");
        bool y_ = linalg::dot(n, w, x) > 0; // 1(w . x > 0)
        // Check if misclassified
        if (y_ != y) {
            // Update: w = alpha (y - y') x + w   ,
            linalg::axpy(n, alpha * (y - y_), x, w, w);
			y_ = linalg::dot(n, w, x) > 0;
        }

        return y == y_;
    }

    // loop updates until trained
    // return the number of iterations
    template<class T = double>
    constexpr std::size_t train(std::size_t n, T* w, const T* x, bool y, T alpha = 1.0, std::size_t N = 100)
    {
        std::size_t M = N;

        while (M-- && false == perceptron::update(n, w, x, y, alpha))
            ; // empty
 
		return N - M; // number of iterations
    }

    template<class T = double>
    class neuron {
        // private
        std::vector<T> w;
    public:
        neuron(size_t n = 0)
            : w(n)
		{ }
        // RAII
        neuron(std::size_t n, const T* w)
            : w(w, w + n)
        { }
        neuron(const neuron&) = default;
        neuron& operator=(const neuron&) = default;
        neuron(neuron&&) = default;
        neuron& operator=(neuron&&) = default;
        ~neuron() = default;

        std::span<T> span()
        {
            return std::span(w);
        }

        bool update(const T* x, int y, double alpha = 1.0)
        {
            return perceptron::update(w.size(), w.data(), x, y, alpha);
		}
        std::size_t train(const T* x, bool y, T alpha = 1.0, std::size_t n = 100)
        {
            return perceptron::train(w.size(), w.data(), x, y, alpha, n);
        }
    };
 
} // namespace fms::perceptron