// fms_option.h - Generalized option pricing model.
// F = f exp(s X - kappa(s)), kappa(s) = log E[exp(s X)]
// E[F] = f, Var(log(F)) = s^2 if E[X] = 0 and Var(X) = 1
// E[(k - F)^+] = E[(k - F) 1(k - F > 0)]
// 			    = E[(k - F) 1(F < k)]
//              = k P(F < k) - E[F 1(F < k)]
//              = k P(F < k) - f P_s(F < k)
// where dP_s/dP = exp(s X - kappa(s))

#pragma once
#include <cmath>

namespace fms::option {

	// Interface for option pricing models. 
	template<class F = double, class S = double>
	struct model {
		using T = std::common_type_t<F, S>;

		// Cumulative share distribution function
		// P_s(X < x) = E[1(X < x) exp(s X - kappa(s))]
		T cdf(F x, S s) const
		{
			return _cdf(x, s);
		}
		// Cumulant generating function
		// kappa(s) = log E[exp(s X)]
		S cgf(S s) const
		{
			return _cgf(s);
		}
	private:
		virtual T _cdf(F x, S s) const = 0;
		virtual S _cgf(S s) const = 0;
	};
	
	namespace black {

		// F < k iff X < (log(k/f) + kappa(s))/s
		template<class F = double, class S = double, class K = double>
		auto moneyness(F f, S s, K k, const model<F, S>& m)
		{
			return (std::log(k / f) + m.cgf(s)) / s;
		}

		template<class F = double, class S = double, class K = double>
		auto put(F f, S s, K k, const model<F, S>& m)
		{
			auto x = moneyness(f, s, k, m);

			return k * m.cdf(x, 0) - f * m.cdf(x, s);
		}
	}

} // namespace fms