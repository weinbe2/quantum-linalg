// Copyright (c) 2017 Evan S Weinberg
// Header file for templated vector operations.

#include <complex>
#include <random>

using std::complex; 
using std::polar;

#include "generic_local_matrix.h"

namespace ESW_QR
{
  #include "qr.h"
}

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

// Do a mat-mat operation in row-major.
// Z = X*Y, square matrix. 
template<typename T> inline void cMATxtMATyMATz_square(T* xmat, T* ymat, T* zmat, int nelem, int ndim)
{
  const int mat_vol = ndim*ndim;
  for (int i = 0; i < nelem; i++)
  {
    cMATxtMATyMATz_square_local(xmat+i*mat_vol, ymat+i*mat_vol, zmat+i*mat_vol, ndim);
  }
}

// Perform a QR decomposition in row-major.
// Complex only.
// xmat -> qmat * rmat
template<typename T> inline void cMATx_do_qr_square(complex<T>* xmat, complex<T>* qmat, complex<T>* rmat, int nelem, int ndim)
{
  const int mat_vol = ndim*ndim;
  for (int i = 0; i < nelem; i++)
  {
    ESW_QR::qr_decomposition(qmat+i*mat_vol, rmat+i*mat_vol, xmat+i*mat_vol, ndim);
  }
}

// Form the inverse of a matrix from a QR decomposition.
// Complex only.
template<typename T> inline void cMATqr_do_xinv_square(complex<T>* qmat, complex<T>* rmat, complex<T>* xinvmat, int nelem, int ndim)
{
  const int mat_vol = ndim*ndim;
  for (int i = 0; i < nelem; i++)
  {
    ESW_QR::matrix_invert_qr(xinvmat+i*mat_vol, qmat+i*mat_vol, rmat+i*mat_vol, ndim);
  }
}

#endif // QLINALG_MATRIX

