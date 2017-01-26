// Copyright (c) 2017 Evan S Weinberg
// Include file for GCR

#ifndef ESW_INVERTER_GCR
#define ESW_INVERTER_GCR

#include <string>
#include <complex>
using std::complex;

#include "inverter_struct.h"
#include "verbosity.h"

// Solves lhs = A^(-1) rhs using generalized conjugate residual
// Makes no assumptions about the matrix. 
inversion_info minv_vector_gcr(double  *phi, double  *phi0, int size, int max_iter, double res, void (*matrix_vector)(double*,double*,void*), void* extra_info, inversion_verbose_struct* verbosity = 0);
inversion_info minv_vector_gcr(complex<double>  *phi, complex<double>  *phi0, int size, int max_iter, double res, void (*matrix_vector)(complex<double>*,complex<double>*,void*), void* extra_info, inversion_verbose_struct* verbosity = 0);

// Solves lhs = A^(-1) with GCR(m), where m is restart_freq. 
inversion_info minv_vector_gcr_restart(double  *phi, double  *phi0, int size, int max_iter, double res, int restart_freq, void (*matrix_vector)(double*,double*,void*), void* extra_info, inversion_verbose_struct* verbosity = 0);
inversion_info minv_vector_gcr_restart(complex<double>  *phi, complex<double>  *phi0, int size, int max_iter, double res, int restart_freq, void (*matrix_vector)(complex<double>*,complex<double>*,void*), void* extra_info, inversion_verbose_struct* verbosity = 0);

#endif