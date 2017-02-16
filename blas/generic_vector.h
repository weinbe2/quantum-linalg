// Copyright (c) 2017 Evan S Weinberg
// Header file for templated vector operations.

#include <complex>
#include <random>

using std::complex; 
using std::polar;

#ifndef GENERIC_VECTOR
#define GENERIC_VECTOR

#ifndef PI
#define PI 3.14159265358979323846
#endif

// Create and destroy a vector.
template<typename T> inline T* allocate_vector(int size)
{
  return new T[size];
}

template<typename T> inline void deallocate_vector(T** x)
{
  delete[] *x;
  *x = 0;
}

// Zeroes a vector.
template<typename T> inline void zero_vector(T* x, int size)
{
   for (int i = 0; i < size; i++)
   {
    x[i] = static_cast<T>(0.0);
   }
}


// Assign a vector to a constant everywhere.
template <typename T, typename U = T> inline void constant_vector(T* x, U val, int size)
{
  for (int i = 0; i < size; i++)
  {
  x[i] = static_cast<T>(val);
  }
}


// Random gaussian vector.
template<typename T> inline void gaussian(T* x, int size, std::mt19937 &generator, double deviation = 1.0)
{
  // Generate a normal distribution.
  std::normal_distribution<> dist(0.0, deviation);
  for (int i = 0; i < size; i++)
  {
  x[i] = static_cast<T>(dist(generator));
  }
}

// Random gaussian vector, random in real and imag.
template <typename T> inline void gaussian(complex<T>* x, int size, std::mt19937 &generator, double deviation = 1.0)
{
  // Generate a normal distribution.
  std::normal_distribution<> dist(0.0, deviation);
  for (int i = 0; i < size; i++)
  {
  x[i] = std::complex<T>(static_cast<T>(dist(generator)), static_cast<T>(dist(generator)));
  }
}

// random vector on a uniform interval.
template <typename T> inline void random_uniform(T* x, int size, std::mt19937 &generator, T min, T max)
{
  std::uniform_real_distribution<> dist(min, max);
  for (int i = 0; i < size; i++)
  {
  x[i] = static_cast<T>(dist(generator));
  }
}

template <typename T> inline void random_uniform(complex<T>* x, int size, std::mt19937 &generator, T min, T max)
{
  std::uniform_real_distribution<> dist(min, max);
  for (int i = 0; i < size; i++)
  {
  x[i] = static_cast<complex<T>>(dist(generator));
  }
}

// vectorized polar. applies x = polar(1.0, real(x));. Ignores imag(x).
template <typename T> inline void polar(complex<T>* x, int size)
{
  for (int i = 0; i < size; i++)
  {
  x[i] = static_cast<complex<T>>(polar(1.0, real(x[i])));
  }
}

// vectorized arg.
template <typename T> inline void arg_vector(complex<T>* x, int size)
{
  for (int i = 0; i < size; i++)
  {
  x[i] = static_cast<complex<T>>(arg(x[i]));
  }
}

// Copy v2 into v1.
template<typename T> inline void copy_vector(T* v1, T* v2, int size)
{
  for (int i = 0; i < size; i++)
  {
    v1[i] = v2[i];
  }
}

// Implement cxpy, y += x.
template<typename T> inline void cxpy(T* x, T* y, int size)
{
  for (int i = 0; i < size; i++)
  {
    y[i] += x[i];
  }
}

// Implement cxty, y *= x.
template<typename T> inline void cxty(T* x, T* y, int size)
{
  for (int i = 0; i < size; i++)
  {
    y[i] *= x[i];
  }
}

// Implement caxpy, y += a*x
template<typename T, typename U = T> inline void caxpy(U a, T* x, T* y, int size)
{
  for (int i = 0; i < size; i++)
  {
    y[i] += a*x[i];
  }
}

// Special strided caxpy.
template<typename T, typename U = T> inline void caxpy_stride(U a, T* x, T* y, int size, int start, int stride)
{
  for (int i = start; i < size; i += stride)
    y[i] += a*x[i];
}

// Implement cxpay, y = x + a*y
template<typename T, typename U = T> inline void cxpay(T* x, U a, T* y, int size)
{
  for (int i = 0; i < size; i++)
  {
  y[i] = x[i] + a*y[i];
  }
}

// Implement caxpby, y = a*x + b*y
template<typename T, typename U = T> inline void caxpby(U a, T* x, U b, T* y, int size)
{
  for (int i = 0; i < size; i++)
  {
  y[i] = a*x[i] + b*y[i];
  }
}

// Implement cxpayz, z = x + a*y
template<typename T, typename U = T> inline void cxpayz(T* x, U a, T* y, T* z, int size)
{
  for (int i = 0; i < size; i++)
  {
  z[i] = x[i] + a*y[i];
  }
}

// Implement caxpbypz, z += a*x + b*y
template<typename T, typename U = T> inline void caxpbypz(U a, T* x, U b, T* y, T* z, int size)
{
  for (int i = 0; i < size; i++)
  {
  z[i] += a*x[i] + b*y[i];
  }
}

// Implement caxpbypcz, z = a*x + b*y + cz
template<typename T, typename U = T> inline void caxpbypcz(U a, T* x, U b, T* y, U c, T* z, int size)
{
  for (int i = 0; i < size; i++)
  {
  z[i] = a*x[i] + b*y[i] + c*z[i];
  }
}


// Implement caxpyBzpx, y += a*x THEN x += b*z. 
template<typename T, typename U = T> inline void caxpyBzpx(U a, T* x, T* y, U b, T* z, int size)
{
  for (int i = 0; i < size; i++)
  {
  y[i] += a*x[i];
  x[i] += b*z[i];
  }
}

// Implement caxpyBxpz, y += a*x, z += b*x
template<typename T, typename U = T> inline void caxpyBxpz(U a, T* x, T* y, U b, T* z, int size)
{
  for (int i = 0; i < size; i++)
  {
  y[i] += a*x[i];
  z[i] += b*x[i];
  }
}

// Computes v1 dot v2.
template<typename T> inline T dot(T* v1, T* v2, int size)
{
  T res = static_cast<T>(0.0);
  for (int i = 0; i < size; i++)
  {
  res = res + v1[i]*v2[i];
  }
  return res;
}

// computes conj(v1) dot v2.
template <typename T> inline complex<T> dot(complex<T>* v1, complex<T>* v2, int size)
{
  complex<T> res = static_cast<T>(0.0);
  for (int i = 0; i < size; i++)
  {
  res = res + conj(v1[i])*v2[i];
  }
  return res;
}

// Computes re(conj(v1) dot v2)
// trivial
template <typename T> inline T re_dot(T* v1, T* v2, int size)
{
  T res = static_cast<T>(0.0);
  for (int i = 0; i < size; i++)
  {
    res = res + v1[i]*v2[i];
  }
  return res;
}

template <typename T> inline T re_dot(complex<T>* v1, complex<T>* v2, int size)
{
  complex<T> res = static_cast<T>(0.0);
  for (int i = 0; i < size; i++)
  {
    res = res + conj(v1[i])*v2[i];
  }
  return real(res);
}

// Sum over the contents.
template <typename T> inline T sum_vector(T* v1, int size)
{
  T res = static_cast<T>(0.0);
  for (int i = 0; i < size; i++)
  {
    res = res + v1[i];
  }
  return res;
}

// Computes the vector norm squared.
template <typename T>
inline T norm2sq(T* v1, int size)
{
  T res = static_cast<T>(0.0);
  for (int i = 0; i < size; i++)
  {
    res = res + v1[i]*v1[i];
  }
  return res;
}

template <typename T>
inline T norm2sq(complex<T>* v1, int size)
{
  T res = static_cast<T>(0.0);
  for (int i = 0; i < size; i++)
  {
    res = res + real(conj(v1[i])*v1[i]);
  }
  return res;
}

// Return the infinity norm (max abs element)
template <typename T>
inline T norminf(T* v1, int size)
{
  T max_abs = static_cast<T>(0.0);
  T tmp = static_cast<T>(0.0);
  for (int i = 0; i < size; i++)
  {
    tmp = abs(v1[i]);
    if (tmp > max_abs)
      max_abs = tmp;
  }
  return max_abs;
}

template <typename T>
inline T norminf(complex<T>* v1, int size)
{
  T max_abs = static_cast<T>(0.0);
  T tmp = static_cast<T>(0.0);
  for (int i = 0; i < size; i++)
  {
    tmp = abs(v1[i]);
    if (tmp > max_abs)
      max_abs = tmp;
  }
  return max_abs;
}

// Return |v1 - v2|^2
template <typename T>
inline T diffnorm2sq(T* v1, T* v2, int size)
{
  T res = static_cast<T>(0.0);
  for (int i = 0; i < size; i++)
  {
    res = res + (v1[i] - v2[i])*(v1[i] - v2[i]);
  }
  return res;
}

template <typename T>
inline T diffnorm2sq(complex<T>* v1, complex<T>* v2, int size)
{
  T res = static_cast<T>(0.0);
  for (int i = 0; i < size; i++)
  {
    res = res + real(conj(v1[i] - v2[i])*(v1[i] - v2[i]));
  }
  return res;
}
  

template <typename T>
inline void normalize(T* v1, int size)
{
  T res = 1.0/sqrt(norm2sq<T>(v1, size));
  if (res > 0.0)
  {
    for (int i = 0; i < size; i++)
    {
      v1[i] *= res;
    }
  }
}

template <typename T>
inline void normalize(complex<T>* v1, int size)
{
  T res = 1.0/sqrt(norm2sq<T>(v1, size));
  if (res > 0.0)
  {
    for (int i = 0; i < size; i++)
    {
      v1[i] *= res;
    }
  }
}

template <typename T>
inline void conj_vector(T* v1, int size)
{
  return; // Trivial, it's real.
}

template <typename T>
inline void conj_vector(complex<T>* v1, int size)
{
  for (int i = 0; i < size; i++)
  {
    v1[i] = conj(v1[i]);
  }
}

// Make vector v1 orthogonal to vector v2. 
template <typename T>
inline void orthogonal(T* v1, T* v2, int size)
{
  T alpha = -dot<T>(v2, v1, size)/norm2sq<T>(v2, size);   
  for (int i = 0; i < size; i++)
  {
    v1[i] = v1[i] + alpha*v2[i];
  }
}

// Make vector v1 orthogonal to vector v2. 
template <typename T>
inline void orthogonal(complex<T>* v1, complex<T>* v2, int size)
{
  complex<T> alpha = -dot<T>(v2, v1, size)/norm2sq<T>(v2, size);   
  for (int i = 0; i < size; i++)
  {
    v1[i] = v1[i] + alpha*v2[i];
  }
}

#endif // GENERIC_VECTOR

