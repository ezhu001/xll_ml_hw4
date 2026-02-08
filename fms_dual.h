// fms_dual.h - Automatic differentiation the easy way.
// If e != 0 and e^2 == 0, then f(e + h) = f(e) + f'(e)h.
// More generaly, e^{n - 1} != 0 and e^n == 0 then
// f(e + h) = f(e) + f'(e)h + f''(e)h^2/2! + ... + f^{n - 1}(e)h^{n - 1}/(n - 1)!
// This allows derivatives to be calculated to machine precision using only algebra.
//
// The can be generalized to f:R^n -> R using e_1, ..., e_n where e_i e_j = e_j e_i
// and the generalized Taylor expansion.
// f(x + h) = sum_n sum_{|a| = n} f^{(a)}(x) h^a / a!
// where a = (a_1, ..., a_n) is a multi-index, |a| = a_1 + ... + a_n,
// h^a = h_1^{a_1} ... h_n^{a_n}, a! = a_1! ... a_n!	
// and f^{(a)} = d^{|a|}f / dx_1^{a_1} ... dx_n^{a_n}.		
#pragma once

#include <valarray>

namespace fms {

	// First order dual number
	// x + x_1 e_1 + ... + x_n e_n, e_j^2 = 0
	template<std::size_t N, class X = double>
	class dual {
		X x;
		std::valarray<X,N> dx;
	public:
		constexpr dual(X x, std::initializer_list<X> dx)
			: x(x), dx(dx)
		{ }
		constexpr X operator X() const
		{
			return x;
		}
		// derivative of i-th component
		constexpr D(unsigned i) const
		{
			return dx[i];
		}

		constexpr dual& operator+=(const dual& y)
		{
			x += y.x;
			dx += y.dx;

			return *this;
		}
		constexpr dual& operator-=(const dual& y)
		{
			x -= y.x;
			dx -= y.dx;

			return *this;
		}
		// (x + x_1 e_1 + ... + x_n e_n) * (y + y_1 e_1 + ... + y_n e_n)
		// = xy + (x y_1 + y x_1) e_1 + ... + (x y_n + y x_n) e_n
		constexpr dual& operator*=(const dual& y)
		{
			x *= y.x;
			dx = x * y.dx + y.x * dx;

			return *this;
		}
		// z = (x + x_1 e_1 + ... + x_n e_n) / (y + y_1 e_1 + ... + y_n e_n)
		// iff z y = x.
		// z y + (z y_1 + y z_1) e_1 + ... + (z y_n + y z_n) e_n
		// = x + x_1 e_1 + ... + x_n e_n
		constexpr dual& operator/=(const dual& y)
		{
			x /= y.x;
			// dx = (dx - x * y.dx) / y.x;

			return *this;
		}
	};

} // namespace fms	
