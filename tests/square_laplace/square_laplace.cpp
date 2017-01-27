// Copyright (c) 2017 Evan S Weinberg
// Test code for a real operator.

#include <iostream>
#include <cmath>
#include <string>
#include <sstream>
#include <complex>

#include "generic_vector.h"

#include "generic_cg.h"
#include "generic_bicgstab.h"
#include "generic_gcr.h"


using namespace std; 

// Square laplacian function.
void square_laplacian(double* lhs, double* rhs, void* extra_data);

struct laplace_struct
{
  int length;
  double m_sq;
};

// Prepare vectors for various tests.
void reset_vectors(double* rhs, double* lhs, double* check, int length)
{
  zero<double>(lhs, length*length);
  zero<double>(rhs, length*length);
  rhs[length/2+(length/2)*length] = 1.0; // set a point on rhs.
  zero<double>(check, length*length);
}


int main(int argc, char** argv)
{  
  //double *lattice; // Holds the gauge field.
  double *lhs, *rhs, *check; // For some Kinetic terms.
  double explicit_resid = 0.0;
  double bnorm = 0.0;
  inversion_info invif;

  // Basic information about the lattice.
  int length = 32;
  double m_sq = 0.01;
  
  // Structure which gets passed to the function.
  laplace_struct lapstr;
  lapstr.length = length;
  lapstr.m_sq = m_sq;
  
  // Parameters related to solve.
  double tol = 1e-8;
  int max_iter = 4000;
  int restart_freq = 64; // for restarted solves. 
  
  // Some start-up.
  int volume = length*length;
  
  //lattice = new double[2*volume];  // Gauge field eventually.
  
  // Vectors. 
  lhs = new double[volume];
  rhs = new double[volume];   
  check = new double[volume];   
  
  // Zero out vectors, set rhs point.
  // zero<double>(lattice, 2*volume);
  reset_vectors(rhs, lhs, check, length);
  
  // Get norm for rhs.
  bnorm = sqrt(norm2sq<double>(rhs, volume));

  printf("Solving A [lhs] = [rhs] for lhs, using a point source.\n");

  // lhs = A^(-1) rhs
  // Arguments:
  // 1: lhs
  // 2: rhs
  // 3: size of vector
  // 4: maximum iterations
  // 5: residual
  // 5a for gcr_restart: how often to restart.
  // 6: function pointer
  // 7: "extra data": 

  /* CG */
  invif = minv_vector_cg(lhs, rhs, volume, max_iter, tol, square_laplacian, &lapstr);
  if (invif.success == true)
  {
    printf("Algorithm %s took %d iterations to reach a residual of %.8e.\n", invif.name.c_str(), invif.iter, sqrt(invif.resSq));
  }
  else // failed, maybe.
  {
    printf("Potential error! Algorithm %s took %d iterations to reach a residual of %.8e.\n", invif.name.c_str(), invif.iter, sqrt(invif.resSq));
  }
  printf("Computing [check] = A [lhs] as a confirmation.\n");
  // Check and make sure we get the right answer.
  square_laplacian(check, lhs, &lapstr);
  explicit_resid = sqrt(diffnorm2sq<double>(rhs, check, volume))/bnorm; // sqrt(|rhs - check|^2)/bnorm
  printf("[check] should equal [rhs]. The residual is %15.20e.\n\n", explicit_resid);
  
  /* BiCGstab */
  reset_vectors(rhs, lhs, check, length);
  invif = minv_vector_bicgstab(lhs, rhs, volume, max_iter, tol, square_laplacian, &lapstr);
  if (invif.success == true)
  {
    printf("Algorithm %s took %d iterations to reach a residual of %.8e.\n", invif.name.c_str(), invif.iter, sqrt(invif.resSq));
  }
  else // failed, maybe.
  {
    printf("Potential error! Algorithm %s took %d iterations to reach a residual of %.8e.\n", invif.name.c_str(), invif.iter, sqrt(invif.resSq));
  }
  printf("Computing [check] = A [lhs] as a confirmation.\n");
  // Check and make sure we get the right answer.
  square_laplacian(check, lhs, &lapstr);
  explicit_resid = sqrt(diffnorm2sq<double>(rhs, check, volume))/bnorm; // sqrt(|rhs - check|^2)/bnorm
  printf("[check] should equal [rhs]. The residual is %15.20e.\n\n", explicit_resid);
  
  /* Restarted GCR */
  reset_vectors(rhs, lhs, check, length);
  invif = minv_vector_gcr_restart(lhs, rhs, volume, max_iter, tol, restart_freq, square_laplacian, &lapstr);
  if (invif.success == true)
  {
    printf("Algorithm %s took %d iterations to reach a residual of %.8e.\n", invif.name.c_str(), invif.iter, sqrt(invif.resSq));
  }
  else // failed, maybe.
  {
    printf("Potential error! Algorithm %s took %d iterations to reach a residual of %.8e.\n", invif.name.c_str(), invif.iter, sqrt(invif.resSq));
  }
  printf("Computing [check] = A [lhs] as a confirmation.\n");
  // Check and make sure we get the right answer.
  square_laplacian(check, lhs, &lapstr);
  explicit_resid = sqrt(diffnorm2sq<double>(rhs, check, volume))/bnorm; // sqrt(|rhs - check|^2)/bnorm
  printf("[check] should equal [rhs]. The residual is %15.20e.\n\n", explicit_resid);

  // Free the lattice.
  //delete[] lattice;
  delete[] lhs;
  delete[] rhs;
  delete[] check;
}

// Square lattice.
// Kinetic term for a 2D laplacian w/ period bc. Applies lhs = A*rhs.
// The unit vectors are e_1 = xhat, e_2 = yhat.
// The "extra_data" allows us to generalize these functions later.
// It would become an internal structure in C++ code.
void square_laplacian(double* lhs, double* rhs, void* extra_data)
{
  // Declare variables.
  int i;
  int x,y;
  laplace_struct* lapstr = (laplace_struct*)extra_data;

  int length = lapstr->length;
  int volume = lapstr->length*lapstr->length;
  
  // For a 2D square lattice, the stencil is:
  //     |  0 -1  0 |
  //     | -1 +4 -1 |
  //     |  0 -1  0 |
  //
  // e2 = yhat
  // ^
  // | 
  // |-> e1 = xhat

  // Apply the stencil.
  for (i = 0; i < volume; i++)
  {
    lhs[i] = 0.0;
    x = i%length; // integer mod.
    y = i/length; // integer divide.

    // + e1.
    lhs[i] = lhs[i]-rhs[y*length+((x+1)%length)];

    // - e1.
    lhs[i] = lhs[i]-rhs[y*length+((x+length-1)%length)];

    // + e2.
    lhs[i] = lhs[i]-rhs[((y+1)%length)*length+x];

    // - e2.
    lhs[i] = lhs[i]-rhs[((y+length-1)%length)*length+x];

    // 0
    // Added mass term here.
    lhs[i] = lhs[i]+(4+lapstr->m_sq)*rhs[i];
  }

}


