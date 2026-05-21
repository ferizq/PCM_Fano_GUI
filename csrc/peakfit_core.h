#ifndef PEAKFIT_CORE_H
#define PEAKFIT_CORE_H

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32) || defined(__CYGWIN__)
  #ifdef PF_CORE_BUILD
    #define PF_EXPORT __declspec(dllexport)
  #else
    #define PF_EXPORT __declspec(dllimport)
  #endif
#else
  #define PF_EXPORT
#endif

enum {
    PF_KERNEL_BESSEL = 0,
    PF_KERNEL_GAUSS = 1
};

PF_EXPORT int pf_kernel_code_from_name(const char *kernel_name);

PF_EXPORT int pf_grid_integral(
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
);

PF_EXPORT const char *pf_core_version(void);

#ifdef __cplusplus
}
#endif

#endif
