// Copyright (c) 2017 Evan S Weinberg
// CG inverter.
// Solves lhs = A^(-1) rhs with CG.
// Assumes the matrix is Hermitian (symmetric) positive definite.



#ifndef QLINALG_INVERTER_CG
#define QLINALG_INVERTER_CG

#include <string>
#include <sstream>
#include <complex>

using std::complex;
using std::stringstream;


#ifndef QLINALG_FCN_POINTER
#define QLINALG_FCN_POINTER
typedef void (*matrix_op_real)(double*,double*,void*);
typedef void (*matrix_op_cplx)(complex<double>*,complex<double>*,void*);
#endif

#include "inverter_struct.h"
#include "../verbosity/verbosity.h"

inversion_info minv_vector_cg(double  *phi, double  *phi0, int size, int max_iter, double res, void (*matrix_vector)(double*,double*,void*), void* extra_info, inversion_verbose_struct* verbosity = 0);
inversion_info minv_vector_cg(complex<double>  *phi, complex<double>  *phi0, int size, int max_iter, double res, void (*matrix_vector)(complex<double>*,complex<double>*,void*), void* extra_info, inversion_verbose_struct* verbosity = 0);

// Solves lhs = A^(-1) with CG(m), where m is restart_freq. 
inversion_info minv_vector_cg_restart(double  *phi, double  *phi0, int size, int max_iter, double res, int restart_freq, void (*matrix_vector)(double*,double*,void*), void* extra_info, inversion_verbose_struct* verbosity = 0);
inversion_info minv_vector_cg_restart(complex<double>  *phi, complex<double>  *phi0, int size, int max_iter, double res, int restart_freq, void (*matrix_vector)(complex<double>*,complex<double>*,void*), void* extra_info, inversion_verbose_struct* verbosity = 0);

inversion_info minv_vector_cg(double  *phi, double  *phi0, int size, int max_iter, double eps, void (*matrix_vector)(double*,double*,void*), void* extra_info, inversion_verbose_struct* verb)
{
// CG solutions to Mphi = b 
//  see http://en.wikipedia.org/wiki/Conjugate_gradient_method
  
  int k;
  // Initialize vectors.
  double *r, *p, *Ap;
  double alpha, beta, rsq, rsqNew, bsqrt, truersq;
  inversion_info invif;

  // Allocate memory.
  r = allocate_vector<double>(size);
  p = allocate_vector<double>(size);
  Ap = allocate_vector<double>(size);

  // Initialize values.
  rsq = 0.0; rsqNew = 0.0; bsqrt = 0.0; truersq = 0.0; k=0;

  // Zero vectors;
  zero_vector(r, size); 
  zero_vector(p, size); zero_vector(Ap, size);
  
  // Find norm of rhs.
  bsqrt = sqrt(norm2sq(phi0, size));
  
  // 1. Compute r = b - Ax using p as a temporary vector. 
  (*matrix_vector)(p, phi, extra_info); invif.ops_count++;
  cxpayz(phi0, -1.0, p, r, size);
  
  // 2. p_0 = r_0.
  copy_vector(p, r, size);
  
  // Compute Ap.
  zero_vector(Ap, size);
  (*matrix_vector)(Ap, p, extra_info); invif.ops_count++;
  
  // Compute rsq.
  rsq = norm2sq(r, size);

  // iterate till convergence
  for(k = 0; k< max_iter; k++) {
    
    // alpha = <r, r>/<p, Ap>
    alpha = rsq/re_dot(p, Ap, size);

    // phi += alpha*p
    caxpy(alpha, p, phi, size);
    
    // r -= alpha*Ap
    caxpy(-alpha, Ap, r, size);
    
    // Exit if new residual is small enough
    rsqNew = norm2sq(r, size);
    
    print_verbosity_resid(verb, "CG", k+1, invif.ops_count, sqrt(rsqNew)/bsqrt); 

    if (sqrt(rsqNew) < eps*bsqrt || k == max_iter-1) {
      //        printf("Final rsq = %g\n", rsqNew);
      break;
    }
  
    // Update vec using new residual
    beta = rsqNew / rsq;
    rsq = rsqNew; 
    
    // p = r + beta*p
    cxpay(r, beta, p, size);
    
    // Compute the new Ap.
    zero_vector(Ap, size);
    (*matrix_vector)(Ap, p, extra_info); invif.ops_count++;
  } 
    
  if(k == max_iter-1) {
    invif.success = false;
  }
  else
  {
     invif.success = true;
  }
  k++;
  
  (*matrix_vector)(Ap,phi,extra_info); invif.ops_count++;
  truersq = diffnorm2sq(Ap, phi0, size);
  
  // Free all the things!
  deallocate_vector(&r);
  deallocate_vector(&p);
  deallocate_vector(&Ap);

  print_verbosity_summary(verb, "CG", invif.success, k, invif.ops_count, sqrt(truersq)/bsqrt);
  
  invif.resSq = truersq;
  invif.iter = k;
  invif.name = "CG";
  return invif; // Convergence 
} 

// Performs CG(restart_freq) with restarts when restart_freq is hit.
// This may be sloppy, but it works.
inversion_info minv_vector_cg_restart(double  *phi, double  *phi0, int size, int max_iter, double res, int restart_freq, void (*matrix_vector)(double*,double*,void*), void* extra_info, inversion_verbose_struct* verb)
{
  int iter; // counts total number of iterations.
  int ops_count; 
  inversion_info invif;
  double bsqrt = sqrt(norm2sq(phi0, size));
  
  inversion_verbose_struct verb_rest;
  shuffle_verbosity_restart(&verb_rest, verb);
  
  stringstream ss;
  ss << "CG(" << restart_freq << ")";
  
  iter = 0; ops_count = 0; 
  do
  {
    invif = minv_vector_cg(phi, phi0, size, min(max_iter, restart_freq), res, matrix_vector, extra_info, &verb_rest);
    iter += invif.iter;
    ops_count += invif.ops_count; 
    
    print_verbosity_restart(verb, ss.str(), iter, ops_count, sqrt(invif.resSq)/bsqrt);
  }
  while (iter < max_iter && invif.success == false && sqrt(invif.resSq)/bsqrt > res);
  
  invif.iter = iter; invif.ops_count = ops_count; 
  
  print_verbosity_summary(verb, ss.str(), invif.success, iter, invif.ops_count, sqrt(invif.resSq)/bsqrt);
  
  invif.name = ss.str();
  // invif.resSq is good.
  if (sqrt(invif.resSq)/bsqrt > res)
  {
    invif.success = false;
  }
  else
  {
    invif.success = true;
  }
  
  return invif;
}


inversion_info minv_vector_cg(complex<double>  *phi, complex<double>  *phi0, int size, int max_iter, double eps, void (*matrix_vector)(complex<double>*,complex<double>*,void*), void* extra_info, inversion_verbose_struct* verb)
{
// CG solutions to Mphi = b 
//  see http://en.wikipedia.org/wiki/Conjugate_gradient_method

  int k;
  // Initialize vectors.
  complex<double> *r, *p, *Ap;
  double alpha, beta;
  complex<double> denom;
  double rsq, rsqNew, bsqrt, truersq;
  inversion_info invif;

  // Allocate memory.
  r = allocate_vector<complex<double>>(size);
  p = allocate_vector<complex<double>>(size);
  Ap = allocate_vector<complex<double>>(size);

  // Initialize values.
  rsq = 0.0; rsqNew = 0.0; bsqrt = 0.0; truersq = 0.0; k=0;

  // Zero vectors;
  zero_vector(r, size); 
  zero_vector(p, size); zero_vector(Ap, size);
  
  // Find norm of rhs.
  bsqrt = sqrt(norm2sq(phi0, size));
  
  // 1. Compute r = b - Ax
  (*matrix_vector)(p, phi, extra_info); invif.ops_count++;
  cxpayz(phi0, -1.0, p, r, size);
  
  // 2. p_0 = r_0.
  copy_vector(p, r, size);
  
  // Compute Ap.
  zero_vector(Ap, size);
  (*matrix_vector)(Ap, p, extra_info); invif.ops_count++;
  
  // Compute rsq.
  rsq = norm2sq(r, size);

  // iterate till convergence
  for(k = 0; k< max_iter; k++) {
    
    // alpha = <r, r>/<p, Ap>
    alpha = rsq/re_dot(p, Ap, size);

    // phi += alpha*p
    caxpy(alpha, p, phi, size);
    
    // r -= alpha*Ap
    caxpy(-alpha, Ap, r, size);
    
    // Exit if new residual is small enough
    rsqNew = norm2sq(r, size);
      
    print_verbosity_resid(verb, "CG", k+1, invif.ops_count, sqrt(rsqNew)/bsqrt);

    if (sqrt(rsqNew) < eps*bsqrt || k == max_iter - 1) {
      //        printf("Final rsq = %g\n", rsqNew);
      break;
    }
  
    // Update vec using new residual
    beta = rsqNew / rsq;
    rsq = rsqNew; 
    
    // p = r + beta*p
    cxpay(r, beta, p, size);
    
    // Compute the new Ap.
    (*matrix_vector)(Ap, p, extra_info); invif.ops_count++;
  } 
    
  if(k == max_iter-1) {
    invif.success = false;
  }
  else
  {
     invif.success = true;
  }
	
  k++; 
  
  (*matrix_vector)(Ap,phi,extra_info); invif.ops_count++;
  truersq = diffnorm2sq(Ap, phi0, size);
  
  // Free all the things!
  deallocate_vector(&r);
  deallocate_vector(&p);
  deallocate_vector(&Ap);

  
  print_verbosity_summary(verb, "CG", invif.success, k, invif.ops_count, sqrt(truersq)/bsqrt);
  
  
  invif.resSq = truersq;
  invif.iter = k;
  invif.name = "CG";
  return invif; // Convergence 
} 


// Performs CG(restart_freq) with restarts when restart_freq is hit.
// This may be sloppy, but it works.
inversion_info minv_vector_cg_restart(complex<double>  *phi, complex<double>  *phi0, int size, int max_iter, double res, int restart_freq, void (*matrix_vector)(complex<double>*,complex<double>*,void*), void* extra_info, inversion_verbose_struct* verb)
{
  int iter; // counts total number of iterations.
  int ops_count; 
  inversion_info invif;
  double bsqrt = sqrt(norm2sq(phi0, size));
  
  inversion_verbose_struct verb_rest;
  shuffle_verbosity_restart(&verb_rest, verb);
  
  stringstream ss;
  ss << "CG(" << restart_freq << ")";
  
  iter = 0; ops_count = 0; 
  do
  {
    invif = minv_vector_cg(phi, phi0, size, min(max_iter, restart_freq), res, matrix_vector, extra_info, &verb_rest);
    iter += invif.iter;
    ops_count += invif.ops_count; 
    
    print_verbosity_restart(verb, ss.str(), iter, ops_count, sqrt(invif.resSq)/bsqrt);
  }
  while (iter < max_iter && invif.success == false && sqrt(invif.resSq)/bsqrt > res);
  
  invif.iter = iter; invif.ops_count = ops_count; 
  
  print_verbosity_summary(verb, ss.str(), invif.success, iter, invif.ops_count, sqrt(invif.resSq)/bsqrt);
  
  invif.name = ss.str();
  // invif.resSq is good.
  if (sqrt(invif.resSq)/bsqrt > res)
  {
    invif.success = false;
  }
  else
  {
    invif.success = true;
  }
  
  return invif;
}



#endif