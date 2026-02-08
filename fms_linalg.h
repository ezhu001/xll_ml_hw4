
// fms_linalg.h - Generic linear algebra utilities
#pragma once
#include <numeric> // inner_product
#include "fms_error.h"

namespace fms::linalg {

	// Compute the dot product of two vectors
	// https://en.cppreference.com/w/cpp/algorithm/inner_product.html
	// https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2025-2/cblas-dot.html
	template<class T = double>
	constexpr T dot(std::size_t n, const T* x, const T* y)
	{
		return std::inner_product(x, x + n, y, T(0));
	}
	namespace { // anonymous
		constexpr double x[] = { 1.0, 2.0, 3.0 };
		constexpr double y[] = { 3.0, 4.0, 5.0 };
		static_assert(dot(3, x, y) == 3 + 8 + 15);
	}

	// z = a * x + y
	// BLAS level 1 axpy
	// https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2025-2/cblas-axpy.html
	template<class T>
	constexpr void axpy(std::size_t n, T a,  const T* x, const T* y, T* z)
	{
		for (size_t i = 0; i < n; ++i) {
			z[i] = a * x[i] + y[i];
		}
	}
	namespace {
		constexpr bool test_axpy() {
			constexpr double x[] = { 1.0, 2.0, 3.0 };
			constexpr double y[] = { 3.0, 4.0, 5.0 };
			constexpr double a = 2.0;
			double z[3] = {};  // Local array (can be modified in constexpr)

			axpy(3, a, x, y, z);

			// Expected: z = 2*x + y = {2+3, 4+4, 6+5} = {5, 8, 11}
			return z[0] == 5.0 && z[1] == 8.0 && z[2] == 11.0;
		}
		static_assert(test_axpy(), "axpy test failed");
	}

} // namespace fms::linalg	
