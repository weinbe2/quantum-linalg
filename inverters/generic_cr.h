// Copyright (c) 2017 Evan S Weinberg
// CG inverter.
// Solves lhs = A^(-1) rhs with CR.
// Assumes the matrix is Hermitian, not necessarily definite.
// Taken from section 6.8 of Saad, 2nd Edition.
// Should implement the modified version:
// The Modified Conjugate Residual Method for Partial Differential Equations, R Chandra, 1977. 



#ifndef QLINALG_INVERTER_CR
#define QLINALG_INVERTER_CR

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

inversion_info minv_vector_cr(double  *phi, double  *phi0, int size, int max_iter, double res, void (*matrix_vector)(double*,double*,void*), void* extra_info, inversion_verbose_struct* verbosity = 0);
inversion_info minv_vector_cr(complex<double>  *phi, complex<double>  *phi0, int size, int max_iter, double res, void (*matrix_vector)(complex<double>*,complex<double>*,void*), void* extra_info, inversion_verbose_struct* verbosity = 0);

// Solves lhs = A^(-1) with CR(m), where m is restart_freq. 
inversion_info minv_vector_cr_restart(double  *phi, double  *phi0, int size, int max_iter, double res, int restart_freq, void (*matrix_vector)(double*,double*,void*), void* extra_info, inversion_verbose_struct* verbosity = 0);
inversion_info minv_vector_cr_restart(complex<double>  *phi, complex<double>  *phi0, int size, int max_iter, double res, int restart_freq, void (*matrix_vector)(complex<double>*,complex<double>*,void*), void* extra_info, inversion_verbose_struct* verbosity = 0);

inversion_info minv_vector_cr(double  *phi, double  *phi0, int size, int max_iter, double eps, void (*matrix_vector)(double*,double*,void*), void* extra_info, inversion_verbose_struct* verb)
{
// CG solutions to Mphi = b 
//  see http://en.wikipedia.org/wiki/Conjugate_gradient_method
  
  int k;
  // Initialize vectors.
  double *r, *Ar, *p, *Ap;
  double alpha, beta, rsq, rsqNew, bsqrt, truersq, Apsq;
  inversion_info invif;

  // Allocate memory.
  r = allocate_vector<double>(size);
  Ar = allocate_vector<double>(size);
  p = allocate_vector<double>(size);
  Ap = allocate_vector<double>(size);

  // Initialize values.
  rsq = 0.0; rsqNew = 0.0; bsqrt = 0.0; truersq = 0.0; k=0;

  // Zero vectors;
  zero_vector(r, size); zero_vector(Ar, size);
  zero_vector(p, size); zero_vector(Ap, size);
  
  // Find norm of rhs.
  bsqrt = sqrt(norm2sq(phi0, size));
  
  // 1. Compute r = b - Ax using p as a temporary vector. 
  (*matrix_vector)(p, phi, extra_info); invif.ops_count++;
  cxpayz(phi0, -1.0, p, r, size);
  
  // 2. p_0 = r_0.
  copy_vector(p, r, size);
  
  // 3. Compute A p_0 = A r_0, presave beta.
  (*matrix_vector)(Ap, p, extra_info); invif.ops_count++;
  copy_vector(Ar, Ap, size);
  Apsq = norm2sq(Ap, size);

  // iterate till convergence
  for(k = 0; k< max_iter; k++) {
    
    // 4. alpha = <Ap_k, r>/<Ap_k, Ap_k>
    alpha = re_dot(Ap, r, size)/Apsq;

    // 5. x = x + alpha p_k
    // 6. r = r - alpha Ap_k
    caxpy(alpha, p, phi, size);
    caxpy(-alpha, Ap, r, size);
    
    // Exit if new residual is small enough
    rsq = norm2sq(r, size);
    
    print_verbosity_resid(verb, "CR", k+1, invif.ops_count, sqrt(rsqNew)/bsqrt); 

    if (sqrt(rsq) < eps*bsqrt || k == max_iter-1) {
      //        printf("Final rsq = %g\n", rsqNew);
      break;
    }

    // 7. Compute Ar.
    zero_vector(Ar, size);
    (*matrix_vector)(Ar, r, extra_info); invif.ops_count++;

    // 8. b_j = <Ap_{j}, Ar_{j+1}>/<Ap_j, Ap_j> (Update beta)
    beta = -dot<double>(Ap, Ar, size)/Apsq;

    // 9. p_{j+1} = r_{j+1} + b_j p_j
    // 10. Ap_{j+1} = Ar_{j+1} + b_j Ap_j
    cxpay(r, beta, p, size);
    cxpay(Ar, beta, Ap, size);
    
    // p = r + beta*p
    cxpay(r, beta, p, size);
  } 
    
  if(k == max_iter-1) {
    invif.success = false;
  }
  else
  {
     invif.success = true;
  }
  k++;
  
  zero_vector(Ap, size);
  (*matrix_vector)(Ap,phi,extra_info); invif.ops_count++;
  truersq = diffnorm2sq(Ap, phi0, size);
  
  // Free all the things!
  deallocate_vector(&r);
  deallocate_vector(&Ar);
  deallocate_vector(&p);
  deallocate_vector(&Ap);

  print_verbosity_summary(verb, "CR", invif.success, k, invif.ops_count, sqrt(truersq)/bsqrt);
  
  invif.resSq = truersq;
  invif.iter = k;
  invif.name = "CR";
  return invif; // Convergence 
} 

// Performs CG(restart_freq) with restarts when restart_freq is hit.
// This may be sloppy, but it works.
inversion_info minv_vector_cr_restart(double  *phi, double  *phi0, int size, int max_iter, double res, int restart_freq, void (*matrix_vector)(double*,double*,void*), void* extra_info, inversion_verbose_struct* verb)
{
  int iter; // counts total number of iterations.
  int ops_count; 
  inversion_info invif;
  double bsqrt = sqrt(norm2sq(phi0, size));
  
  inversion_verbose_struct verb_rest;
  shuffle_verbosity_restart(&verb_rest, verb);
  
  stringstream ss;
  ss << "CR(" << restart_freq << ")";
  
  iter = 0; ops_count = 0; 
  do
  {
    invif = minv_vector_cr(phi, phi0, size, min(max_iter, restart_freq), res, matrix_vector, extra_info, &verb_rest);
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


inversion_info minv_vector_cr(complex<double>  *phi, complex<double>  *phi0, int size, int max_iter, double eps, void (*matrix_vector)(complex<double>*,complex<double>*,void*), void* extra_info, inversion_verbose_struct* verb)
{
  // Initialize vectors.
  complex<double> *r, *Ar, *p, *Ap;
  double rsq, bsqrt, truersq, Apsq; 
  complex<double> alpha, beta;
  int k;
  inversion_info invif;

  // Allocate memory.
  r = allocate_vector<complex<double>>(size);
  Ar = allocate_vector<complex<double>>(size);
  p = allocate_vector<complex<double>>(size);
  Ap = allocate_vector<complex<double>>(size);
  
  // Zero vectors. 
  zero_vector(p, size); zero_vector(r, size);
  zero_vector(Ap, size); zero_vector(Ar, size);

  // Initialize values.
  rsq = 0.0; bsqrt = 0.0; truersq = 0.0;
  
  // Find norm of rhs.
  bsqrt = sqrt(norm2sq(phi0, size));
  
  // 1. r_0 = b - Ax_0.
  (*matrix_vector)(p, phi, extra_info); invif.ops_count++; // Put Ax_0 into p, temp.
  cxpayz(phi0, -1.0, p, r, size);
  
  // 2. p_0 = r_0.
  copy_vector(p, r, size);
  
  // 3. Compute A p_0 = A r_0, presave beta.
  (*matrix_vector)(Ap, p, extra_info); invif.ops_count++;
  
  copy_vector(Ar, Ap, size);
  Apsq = norm2sq(Ap, size); 
  

  // iterate until convergence
  for(k = 0; k< max_iter; k++) {
    
    // 4. alpha = <Ap_k, r>/<Ap_k, Ap_k>
    alpha = dot(Ap, r, size)/Apsq; 
      
    // 5. x = x + alpha p_k
    // 6. r = r - alpha Ap_k
    caxpy(alpha, p, phi, size);
    caxpy(-alpha, Ap, r, size);
    
    // Compute norm.
    rsq = norm2sq(r, size);
    
    print_verbosity_resid(verb, "CR", k+1, invif.ops_count, sqrt(rsq)/bsqrt); 
    
    // Check convergence. 
    if (sqrt(rsq) < eps*bsqrt || k == max_iter-1) {
      //        printf("Final rsq = %g\n", rsqNew);
      break;
    }
    
    // 7. Compute Ar.
    zero_vector(Ar, size);
    (*matrix_vector)(Ar, r, extra_info); invif.ops_count++;
    
    // 8. b_j = -<Ap_{j}, Ar_{j+1}>/<Ap_j, Ap_j> (Update beta)
    beta = -dot(Ap, Ar, size)/Apsq; // Might be unstable, there's a way to correct this.
    
    // 9. p_{j+1} = r_{j+1} + b_j p_j
    // 10. Ap_{j+1} = Ar_{j+1} + b_j Ap_j
    cxpay(r, beta, p, size);
    cxpay(Ar, beta, Ap, size);
    
    Apsq = norm2sq(Ap, size); 
  } 
    
  if(k == max_iter) {
    //printf("CG: Failed to converge iter = %d, rsq = %e\n", k,rsq);
    invif.success = false;
    //return 0;// Failed convergence 
  }
  else
  {
     invif.success = true;
     //printf("CG: Converged in %d iterations.\n", k);
  }
  k++; 
  
  // Check true residual.
  zero_vector(Ap, size);
  (*matrix_vector)(Ap,phi,extra_info); invif.ops_count++;
  truersq = diffnorm2sq(Ap, phi0, size);
  
  // Free all the things!
  deallocate_vector(&r);
  deallocate_vector(&Ar);
  deallocate_vector(&p);
  deallocate_vector(&Ap);

  print_verbosity_summary(verb, "CR", invif.success, k, invif.ops_count, sqrt(truersq)/bsqrt);
  
  invif.resSq = truersq;
  invif.iter = k;
  invif.name = "CR";
  return invif; // Convergence 
  
} 


// Performs CG(restart_freq) with restarts when restart_freq is hit.
// This may be sloppy, but it works.
inversion_info minv_vector_cr_restart(complex<double>  *phi, complex<double>  *phi0, int size, int max_iter, double res, int restart_freq, void (*matrix_vector)(complex<double>*,complex<double>*,void*), void* extra_info, inversion_verbose_struct* verb)
{
  int iter; // counts total number of iterations.
  int ops_count; 
  inversion_info invif;
  double bsqrt = sqrt(norm2sq(phi0, size));
  
  inversion_verbose_struct verb_rest;
  shuffle_verbosity_restart(&verb_rest, verb);
  
  stringstream ss;
  ss << "CR(" << restart_freq << ")";
  
  iter = 0; ops_count = 0; 
  do
  {
    invif = minv_vector_cr(phi, phi0, size, min(max_iter, restart_freq), res, matrix_vector, extra_info, &verb_rest);
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