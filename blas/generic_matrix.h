// Copyright (c) 2017 Evan S Weinberg
// Header file for templated vector operations.

#include <complex>
#include <random>

using std::complex; 
using std::polar;

#include "generic_local_matrix.h"

#ifndef QLINALG_MATRIX
#define QLINALG_MATRIX

#ifndef PI
#define PI 3.14159265358979323846
#endif

// Apply the global mat-vec operation in row-major.
template<typename T> inline void cMATxpy(T* mat, T* x, T* y, int nelem, int nrow, int ncol)
{
  int mat_vol = nrow*ncol;
  for (int i = 0; i < nelem; i++)
  {
    cMATxpy_local(mat+i*mat_vol, x+i*ncol, y+i*nrow, nrow, ncol);
  }
}

// Transpose the global mat.
template<typename T> inline void cMATtranspose_square(T* mat, int nelem, int ndim)
{
  int mat_vol = ndim*ndim;
  for (int i = 0; i < nelem; i++)
  {
    cMATtranspose_square_local(mat+i*mat_vol, ndim);
  }
}

// Copy + Transpose the global mat.
template<typename T> inline void cMATcopy_transpose_square(T* mat, T* mat_dest, int nelem, int ndim)
{
  int mat_vol = ndim*ndim;
  for (int i = 0; i < nelem; i++)
  {
    cMATcopy_transpose_square_local(mat+i*mat_vol, mat_dest+i*mat_vol, ndim);
  }
}

// Conjugate transpose the global mat.
template<typename T> inline void cMATconjtrans_square(complex<T>* mat, int nelem, int ndim)
{
  int mat_vol = ndim*ndim;
  for (int i = 0; i < nelem; i++)
  {
    cMATconjtrans_square_local(mat+i*mat_vol, ndim);
  }
}

// Conjugate transpose the global mat.
template<typename T> inline void cMATcopy_conjtrans_square(complex<T>* mat, complex<T>* mat_dest, int nelem, int ndim)
{
  int mat_vol = ndim*ndim;
  for (int i = 0; i < nelem; i++)
  {
    cMATcopy_conjtrans_square_local(mat+i*mat_vol, mat_dest+i*mat_vol, ndim);
  }
}


#endif // QLINALG_MATRIX

