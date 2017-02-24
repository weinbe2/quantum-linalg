// Copyright (c) 2017 Evan S Weinberg
// MinRes inverter.
// Solves lhs = A^(-1) rhs with CG.
// Assumes the matrix has a non-zero Hermitian component,
// though I haven't quite figured out if relaxation could
// change that.

#ifndef QLINALG_INVERTER_MINRES
#define QLINALG_INVERTER_MINRES

#include <string>
#include <sstream>
#include <complex>

using std::complex;
using std::stringstream;

#include "inverter_struct.h"
#include "../verbosity/verbosity.h"
#include "../blas/generic_vector.h"

// Solve Ax = b with Minres.
inversion_info minv_vector_minres(double  *phi, double  *phi0, int size, int max_iter, double res, double omega, void (*matrix_vector)(double*,double*,void*), void* extra_info, inversion_verbose_struct* verbosity = 0);
inversion_info minv_vector_minres(complex<double>  *phi, complex<double>  *phi0, int size, int max_iter, double res, complex<double> omega, void (*matrix_vector)(complex<double>*,complex<double>*,void*), void* extra_info, inversion_verbose_struct* verbosity = 0);

// Solve Ax = b with Minres, assuming the relaxation parameter = 1.0.
inversion_info minv_vector_minres(double  *phi, double  *phi0, int size, int max_iter, double res, void (*matrix_vector)(double*,double*,void*), void* extra_info, inversion_verbose_struct* verbosity = 0);
inversion_info minv_vector_minres(complex<double>  *phi, complex<double>  *phi0, int size, int max_iter, double res, void (*matrix_vector)(complex<double>*,complex<double>*,void*), void* extra_info, inversion_verbose_struct* verbosity = 0);


inversion_info minv_vector_minres(double  *phi, double  *phi0, int size, int max_iter, double res, double omega, void (*matrix_vector)(double*,double*,void*), void* extra_info, inversion_verbose_struct* verbosity)
{
  // Iterators.
  int k;

  // Initialize vectors.
  double *r, *p;
  double alpha, rsq, truersq, bsqrt;
  inversion_info invif;

  stringstream ss;
  ss << "MR_" << omega;

  // Allocate memory.
  r = allocate_vector<double>(size);
  p = allocate_vector<double>(size);

  // Zero vectors;
  zero_vector(r, size); 
  zero_vector(p, size);
  
  // Find norm of rhs.
  bsqrt = sqrt(norm2sq(phi0, size));
  
  // 1. Compute r = b - Ax using p as a temporary vector. 
  (*matrix_vector)(p, phi, extra_info); invif.ops_count++;
  cxpayz(phi0, -1.0, p, r, size); // r = b - Ax

  // Initialize values.
  alpha = 0.0; rsq = 0.0; truersq = 0.0;

  // iterate till convergence
  for(k = 0; k < max_iter; k++)
  {
    // 2. p = Ar.
    zero_vector(p, size);
    (*matrix_vector)(p, r, extra_info); invif.ops_count++; // p = Ar

    // 3. alpha = omega*<p,r>/<p,p>
    alpha = omega*dot(p, r, size)/norm2sq(p, size);

    // 4. x = x + alpha r
    // 5. r = r - alpha p
    caxpyBzpx(alpha, r, phi, -alpha, p, size); 

    // 6. Compute norm.
    rsq = norm2sq(r, size);
    
    print_verbosity_resid(verbosity, ss.str(), k+1, invif.ops_count, sqrt(rsq)/bsqrt); 
    
    // Check convergence. 
    if (sqrt(rsq) < res*bsqrt || k == max_iter-1) {
      break;
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
  zero_vector(p, size);
  (*matrix_vector)(p, phi, extra_info); invif.ops_count++;
  truersq = diffnorm2sq(p, phi0, size);
  
  // Free all the things!
  deallocate_vector(&r);
  deallocate_vector(&p);

  print_verbosity_summary(verbosity, ss.str(), invif.success, k, invif.ops_count, sqrt(truersq)/bsqrt);
  
  invif.resSq = truersq;
  invif.iter = k;
  invif.name = ss.str();

  return invif; // Convergence 
} 

// Version without relaxation parameter.
inversion_info minv_vector_minres(double  *phi, double  *phi0, int size, int max_iter, double res, void (*matrix_vector)(double*,double*,void*), void* extra_info, inversion_verbose_struct* verbosity)
{
  return minv_vector_minres(phi, phi0, size, max_iter, res, 1.0, matrix_vector, extra_info, verbosity);
}

// Complex version of Minres
inversion_info minv_vector_minres(complex<double>  *phi, complex<double>  *phi0, int size, int max_iter, double res, complex<double> omega, void (*matrix_vector)(complex<double>*,complex<double>*,void*), void* extra_info, inversion_verbose_struct* verbosity)
{
  // Iterators.
  int k;

  // Initialize vectors.
  complex<double> *r, *p;
  complex<double> alpha;
  double rsq, truersq, bsqrt;
  inversion_info invif;

  stringstream ss;
  ss << "MR_" << omega;

  // Allocate memory.
  r = allocate_vector<complex<double>>(size);
  p = allocate_vector<complex<double>>(size);

  // Zero vectors;
  zero_vector(r, size); 
  zero_vector(p, size);
  
  // Find norm of rhs.
  bsqrt = sqrt(norm2sq(phi0, size));
  
  // 1. Compute r = b - Ax using p as a temporary vector. 
  (*matrix_vector)(p, phi, extra_info); invif.ops_count++;
  cxpayz(phi0, -1.0, p, r, size); // r = b - Ax

  // Initialize values.
  alpha = 0.0; rsq = 0.0; truersq = 0.0;

  // iterate till convergence
  for(k = 0; k < max_iter; k++)
  {
    // 2. p = Ar.
    zero_vector(p, size);
    (*matrix_vector)(p, r, extra_info); invif.ops_count++; // p = Ar

    // 3. alpha = omega*<p,r>/<p,p>
    alpha = omega*dot(p, r, size)/norm2sq(p, size);

    // 4. x = x + alpha r
    // 5. r = r - alpha p
    caxpyBzpx(alpha, r, phi, -alpha, p, size); 

    // 6. Compute norm.
    rsq = norm2sq(r, size);
    
    print_verbosity_resid(verbosity, ss.str(), k+1, invif.ops_count, sqrt(rsq)/bsqrt); 
    
    // Check convergence. 
    if (sqrt(rsq) < res*bsqrt || k == max_iter-1) {
      break;
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
  zero_vector(p, size);
  (*matrix_vector)(p, phi, extra_info); invif.ops_count++;
  truersq = diffnorm2sq(p, phi0, size);
  
  // Free all the things!
  deallocate_vector(&r);
  deallocate_vector(&p);

  print_verbosity_summary(verbosity, ss.str(), invif.success, k, invif.ops_count, sqrt(truersq)/bsqrt);
  
  invif.resSq = truersq;
  invif.iter = k;
  invif.name = ss.str();

  return invif; // Convergence 
} 

// Version without relaxation parameter.
inversion_info minv_vector_minres(complex<double>  *phi, complex<double>  *phi0, int size, int max_iter, double res, void (*matrix_vector)(complex<double>*,complex<double>*,void*), void* extra_info, inversion_verbose_struct* verbosity)
{
  return minv_vector_minres(phi, phi0, size, max_iter, res, 1.0, matrix_vector, extra_info, verbosity);
}

#endif