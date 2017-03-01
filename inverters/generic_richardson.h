// Copyright (c) 2017 Evan S Weinberg
// Richardson inverter.
// Solves lhs = A^(-1) rhs with Richardson iterations.
// For an operator satisfying the positive half-plane condition,
// the Richardson iteration converges if the relaxation
// parameter, alpha, satisfies
// alpha < 2 Re(lambda)/[ Re(lambda)^2 + Im(lambda)^2]
// for the eigenvalue lambda with the largest magnitude.

#ifndef QLINALG_INVERTER_RICHARDSON
#define QLINALG_INVERTER_RICHARDSON

#include <string>
#include <sstream>
#include <complex>

using std::complex;
using std::stringstream;

#include "inverter_struct.h"
#include "../verbosity/verbosity.h"
#include "../blas/generic_vector.h"

// Solve Ax = b with Richardson.
inversion_info minv_vector_richardson(double  *phi, double  *phi0, int size, int max_iter, double res, double omega, int check_freq, void (*matrix_vector)(double*,double*,void*), void* extra_info, inversion_verbose_struct* verbosity = 0);
inversion_info minv_vector_richardson(complex<double>  *phi, complex<double>  *phi0, int size, int max_iter, double res, complex<double> omega, int check_freq, void (*matrix_vector)(complex<double>*,complex<double>*,void*), void* extra_info, inversion_verbose_struct* verbosity = 0);

// Solve Ax = b with Richardson, assuming the relaxation parameter = 1.0 and check_freq = 1
inversion_info minv_vector_richardson(double  *phi, double  *phi0, int size, int max_iter, double res, void (*matrix_vector)(double*,double*,void*), void* extra_info, inversion_verbose_struct* verbosity = 0);
inversion_info minv_vector_richardson(complex<double>  *phi, complex<double>  *phi0, int size, int max_iter, double res, void (*matrix_vector)(complex<double>*,complex<double>*,void*), void* extra_info, inversion_verbose_struct* verbosity = 0);


inversion_info minv_vector_richardson(double  *phi, double  *phi0, int size, int max_iter, double res, double omega, int check_freq, void (*matrix_vector)(double*,double*,void*), void* extra_info, inversion_verbose_struct* verbosity)
{
  // Iterators.
  int k;

  // Initialize vectors.
  double *Aphi;
  double truersq, rsq, bsqrt;
  inversion_info invif;

  stringstream ss;
  ss << "Richardson_" << omega;

  // Allocate memory.
  Aphi = allocate_vector<double>(size);
  
  // Find norm of rhs.
  bsqrt = sqrt(norm2sq(phi0, size));

  // iterate till convergence: x_{n+1} = x_n + omega(b - Ax_n)
  for(k = 0; k < max_iter; k++)
  {
    // Apply A to phi.
    zero_vector(Aphi, size);
    (*matrix_vector)(Aphi, phi, extra_info); invif.ops_count++;

    // Update x = x + omega(b - Ax)
    caxpbypz(omega, phi0, -omega, Aphi, phi, size);

    // Compute norm if it's time. Technically this is the norm of
    // the previous iteration, but eh.
    if (k % check_freq == 0)
    {
      rsq = diffnorm2sq(Aphi, phi0, size);
      print_verbosity_resid(verbosity, ss.str(), k+1, invif.ops_count, sqrt(rsq)/bsqrt); 

      // Check convergence. 
      if (sqrt(rsq) < res*bsqrt || k == max_iter-1) {
        break;
      }
    }
  } 
    
  if(k == max_iter-1) {
    invif.success = false;
  }
  else
  {
     invif.success = true;
  }
  k++;
  
  // Check true residual.
  zero_vector(Aphi, size);
  (*matrix_vector)(Aphi, phi, extra_info); invif.ops_count++;
  truersq = diffnorm2sq(Aphi, phi0, size);
  
  // Free all the things!
  deallocate_vector(&Aphi);

  print_verbosity_summary(verbosity, ss.str(), invif.success, k, invif.ops_count, sqrt(truersq)/bsqrt);
  
  invif.resSq = truersq;
  invif.iter = k;
  invif.name = ss.str();

  return invif; // Convergence 
} 

// Version without relaxation parameter and and check_freq = 1.
inversion_info minv_vector_richardson(double  *phi, double  *phi0, int size, int max_iter, double res, void (*matrix_vector)(double*,double*,void*), void* extra_info, inversion_verbose_struct* verbosity)
{
  return minv_vector_richardson(phi, phi0, size, max_iter, res, 1.0, 1, matrix_vector, extra_info, verbosity);
}

// Complex version of Richardson iterations
inversion_info minv_vector_richardson(complex<double>  *phi, complex<double>  *phi0, int size, int max_iter, double res, complex<double> omega, int check_freq, void (*matrix_vector)(complex<double>*,complex<double>*,void*), void* extra_info, inversion_verbose_struct* verbosity)
{
  // Iterators.
  int k;

  // Initialize vectors.
  complex<double> *Aphi;
  double truersq, rsq, bsqrt;
  inversion_info invif;

  stringstream ss;
  ss << "Richardson_" << omega;

  // Allocate memory.
  Aphi = allocate_vector<complex<double>>(size);
  
  // Find norm of rhs.
  bsqrt = sqrt(norm2sq(phi0, size));

  // iterate till convergence: x_{n+1} = x_n + omega(b - Ax_n)
  for(k = 0; k < max_iter; k++)
  {
    // Apply A to phi.
    zero_vector(Aphi, size);
    (*matrix_vector)(Aphi, phi, extra_info); invif.ops_count++;

    // Update x = x + omega(b - Ax)
    caxpbypz(omega, phi0, -omega, Aphi, phi, size);

    // Compute norm if it's time. Technically this is the norm of
    // the previous iteration, but eh.
    if (k % check_freq == 0)
    {
      rsq = diffnorm2sq(Aphi, phi0, size);
      print_verbosity_resid(verbosity, ss.str(), k+1, invif.ops_count, sqrt(rsq)/bsqrt); 

      // Check convergence. 
      if (sqrt(rsq) < res*bsqrt || k == max_iter-1) {
        break;
      }
    }
  } 
    
  if(k == max_iter-1) {
    invif.success = false;
  }
  else
  {
     invif.success = true;
  }
  k++;
  
  // Check true residual.
  zero_vector(Aphi, size);
  (*matrix_vector)(Aphi, phi, extra_info); invif.ops_count++;
  truersq = diffnorm2sq(Aphi, phi0, size);
  
  // Free all the things!
  deallocate_vector(&Aphi);

  print_verbosity_summary(verbosity, ss.str(), invif.success, k, invif.ops_count, sqrt(truersq)/bsqrt);
  
  invif.resSq = truersq;
  invif.iter = k;
  invif.name = ss.str();

  return invif; // Convergence 
} 

// Version without relaxation parameter.
inversion_info minv_vector_richardson(complex<double>  *phi, complex<double>  *phi0, int size, int max_iter, double res, void (*matrix_vector)(complex<double>*,complex<double>*,void*), void* extra_info, inversion_verbose_struct* verbosity)
{
  return minv_vector_richardson(phi, phi0, size, max_iter, res, 1.0, 1, matrix_vector, extra_info, verbosity);
}

#endif