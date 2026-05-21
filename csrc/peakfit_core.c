#include "peakfit_core.h"

#include <ctype.h>
#include <math.h>
#include <stddef.h>

#if defined(_MSC_VER)
#include <float.h>
#define pf_isfinite _finite
#else
#define pf_isfinite isfinite
#endif

static const double PF_NUMERIC_EPS = 1e-24;
static const double PF_IK_SMALL = 1e-12;
static const double PF_PI = 3.14159265358979323846264338327950288;

static int pf_ascii_ieq(const char *lhs, const char *rhs) {
    if (lhs == NULL || rhs == NULL) {
        return 0;
    }
    while (*lhs != '\0' && *rhs != '\0') {
        unsigned char a = (unsigned char)(*lhs);
        unsigned char b = (unsigned char)(*rhs);
        if ((unsigned char)tolower(a) != (unsigned char)tolower(b)) {
            return 0;
        }
        ++lhs;
        ++rhs;
    }
    return (*lhs == '\0' && *rhs == '\0') ? 1 : 0;
}

int pf_kernel_code_from_name(const char *kernel_name) {
    if (kernel_name == NULL) {
        return PF_KERNEL_BESSEL;
    }
    if (pf_ascii_ieq(kernel_name, "pcm_fano_gauss") ||
        pf_ascii_ieq(kernel_name, "exp") ||
        pf_ascii_ieq(kernel_name, "gauss") ||
        pf_ascii_ieq(kernel_name, "gaussian_exp")) {
        return PF_KERNEL_GAUSS;
    }
    return PF_KERNEL_BESSEL;
}

static double pf_safe_nonzero(double value) {
    if (fabs(value) > PF_NUMERIC_EPS) {
        return value;
    }
    return (value < 0.0) ? -PF_NUMERIC_EPS : PF_NUMERIC_EPS;
}

static double pf_integrand_scalar(
    double ik,
    double wi,
    int kernel_code,
    double C,
    double D,
    double b,
    double g0,
    double q,
    double a,
    double L,
    double alpha
) {
    double sqrt_arg = C + D * cos(ik * PF_PI / 2.0);
    double w0;
    double eps;
    double num;

    if (sqrt_arg < 0.0) {
        sqrt_arg = 0.0;
    }
    w0 = sqrt(sqrt_arg) - b;

    eps = 2.0 * (wi - w0) / pf_safe_nonzero(g0);
    num = ((eps + q) * (eps + q)) / (1.0 + (eps * eps));

    if (kernel_code == PF_KERNEL_GAUSS) {
        double a_safe = pf_safe_nonzero(a);
        double alpha_safe = pf_safe_nonzero(alpha);
        double exponent = (-2.0 * (PF_PI * PF_PI) * (ik * ik) * (L * L)) / (alpha_safe * (a_safe * a_safe));
        double den = (ik * ik) * exp(exponent);
        return den * num;
    }

    {
        double a_safe = pf_safe_nonzero(a);
        double x = (ik * PF_PI / a_safe) * L;
        double numerator = sin(x) - x * cos(x);
        double den;
        if (fabs(ik) <= PF_IK_SMALL) {
            double c = (L * PF_PI / a_safe);
            den = (c * c * c * c * c * c) * (ik * ik) / 9.0;
        } else {
            double ik2 = ik * ik;
            den = (numerator * numerator) / (ik2 * ik2);
        }
        return den * num;
    }
}

int pf_grid_integral(
    const double *iw_arr,
    int n_w,
    double ik_min,
    double ik_max,
    int grid_size,
    int kernel_code,
    double C,
    double D,
    double b,
    double g0,
    double q,
    double a,
    double L,
    double alpha,
    double *out_integral
) {
    int j;
    int i;
    int n_grid;
    double step;

    if (iw_arr == NULL || out_integral == NULL) {
        return -1;
    }
    if (n_w <= 0) {
        return -2;
    }
    if (!pf_isfinite(ik_min) || !pf_isfinite(ik_max)) {
        return -3;
    }

    n_grid = (grid_size > 1) ? grid_size : 2;

    if (ik_max == ik_min) {
        for (j = 0; j < n_w; ++j) {
            out_integral[j] = 0.0;
        }
        return 0;
    }

    step = (ik_max - ik_min) / (double)(n_grid - 1);
    if (!pf_isfinite(step)) {
        return -4;
    }

    for (j = 0; j < n_w; ++j) {
        double wi = iw_arr[j];
        double ik_prev = ik_min;
        double prev = pf_integrand_scalar(ik_prev, wi, kernel_code, C, D, b, g0, q, a, L, alpha);
        double acc = 0.0;

        for (i = 1; i < n_grid; ++i) {
            double ik_cur = ik_min + (double)i * step;
            double cur = pf_integrand_scalar(ik_cur, wi, kernel_code, C, D, b, g0, q, a, L, alpha);
            acc += 0.5 * (prev + cur) * step;
            prev = cur;
        }

        out_integral[j] = acc;
    }

    return 0;
}

const char *pf_core_version(void) {
    return "0.1.0";
}
