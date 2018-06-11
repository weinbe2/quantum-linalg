// Copyright (c) 2017 Evan S Weinberg
// Test code for a real operator.

#include <iostream>
#include <iomanip>
#include <cmath>
#include <string>
#include <sstream>
#include <complex>
#include <random>

// Borrow dense matrix eigenvalue routines.
#include <Eigen/Dense>

#include "blas/generic_vector.h"

#include "square_laplace.h"

using namespace std; 
using namespace Eigen;

typedef Matrix<double, Dynamic, Dynamic, ColMajor> dMatrix;
typedef Matrix<std::complex<double>, Dynamic, Dynamic, ColMajor> cMatrix;

int main(int argc, char** argv)
{  
  complex<double> *rhs_cplx;

  // Set output precision to be long.
  cout << setprecision(10);

  // RNG related things.
  std::mt19937 generator (1337u); // RNG, 1337u is the seed. 
  double inv_variance = 6.0; // inverse of variance for gaussian non-compact U(1) links.

  // Basic information about the lattice.
  int length = 8;
  double m_sq = 0.001;
  
  // Some start-up.
  int volume = length*length;
  
  // Create a random compact U(1) link.
  complex<double>* gauge_links = allocate_vector<complex<double>>(2*length*length);
  gaussian_real(gauge_links, 2*length*length, generator, 1.0/inv_variance);
  polar(gauge_links, 2*length*length);
  
  // Vectors.
  rhs_cplx = allocate_vector<complex<double>>(volume);

  // Zero out the vector.
  zero_vector(rhs_cplx, length*length);

  // Structure which gets passed to the function.
  laplace_gauged_struct lapstr_gauged;
  lapstr_gauged.length = length;
  lapstr_gauged.m_sq = m_sq;
  lapstr_gauged.gauge_links = gauge_links; 

  // Uncomment this to get the free field.
  //constant_vector(gauge_links, 1.0, 2*volume);

  std::cout << "Free case.\n\n";

  // Based on arXiv:1512.08135

  // m-step
  const int m = 20;

  // Allocate m q's
  complex<double>** Q = new complex<double>*[m];
  for (int i = 0; i < m; i++) {
    Q[i] = allocate_vector<complex<double>>(volume);
    zero_vector(Q[i], volume);
  }

  // Allocate a temporary 'w' vector
  complex<double>* w = allocate_vector<complex<double>>(volume);

  // Prepare the tridiagonal matrix T_m
  dMatrix Tm = dMatrix::Zero(m,m);

  // Prepare a random unit vector for Q[0]!
  gaussian(Q[0], volume, generator);
  normalize(Q[0], volume);

  // Let's goooo
  for (int i = 0; i < m; i++) {
    zero_vector(w, volume);
    square_laplacian_gauged(w, Q[i], &lapstr_gauged); // w = B q_i
    if (i > 0) {
      caxpy(-Tm(i-1,i), Q[i-1], w, volume); // w -= beta_i q_{i-1}
    }

    // Compute alpha_i
    Tm(i,i) = re_dot(Q[i], w, volume);

    // Break here if we're on the last step.
    if (i == (m-1)) { break; }

    // w -= alpha_i q_i
    caxpy(-Tm(i,i), Q[i], w, volume);

    // Reorthogonalize
    for (int j = 0; j < i; j++) {
      orthogonal(w, Q[j], volume);
    }

    // Compute beta_{i+1}
    Tm(i+1,i) = Tm(i,i+1) = sqrt(norm2sq(w, volume));

    // Check for breakdown
    if (fabs(Tm(i,i+1)) < 1e-10) {
      gaussian(Q[i+1], volume, generator);
      normalize(Q[i+1], volume);
    } else {
      caxy(1.0/Tm(i+1,i), w, Q[i+1], volume);
    }
  }

  // Let's check the Ritz values!
  SelfAdjointEigenSolver<dMatrix> eigsolve_Tm(m);
  eigsolve_Tm.compute(Tm);
  std::cout << "The Ritz values are:\n" << eigsolve_Tm.eigenvalues() << "\n";

  // Let's get the eigenvalues of the full operator!

  // Allocate a sufficiently gigantic matrix.
  cMatrix mat_cplx = cMatrix::Zero(volume, volume);

  // Form matrix elements. This is where it's important that
  // dMatrix and cMatrix are column major.
  // I should probably make this safer by using a "Map".
  for (int i = 0; i < volume; i++)
  {
    // Set a point on the rhs for a matrix element.
    zero_vector(rhs_cplx, volume);
    rhs_cplx[i] = 1.0;

    // Where we put the result of the matrix element.
    complex<double>* mptr = &(mat_cplx(i*volume));

    square_laplacian_gauged(mptr, rhs_cplx, &lapstr_gauged);
  }

  // Get the eigenvalues.
  SelfAdjointEigenSolver<cMatrix> eigsolve_cplx(volume);
  eigsolve_cplx.compute(mat_cplx);

  std::cout << "The eigenvalues are:\n" << eigsolve_cplx.eigenvalues() << "\n";

  //////////////
  // CLEAN UP //
  //////////////

  deallocate_vector(&w);

  for (int i = 0; i < m; i++) {
    deallocate_vector(&Q[i]);
  }
  delete[] Q;

  deallocate_vector(&rhs_cplx);
  deallocate_vector(&gauge_links);
  return 0;
}


