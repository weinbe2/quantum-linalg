// Copyright (c) 2017 Evan S Weinberg
// Header file for templated vector operations.

#include <complex>
#include <random>

#ifndef GENERIC_VECTOR
#define GENERIC_VECTOR

// Zeroes a vector.
template<typename T> inline void zero(T* v1, int size)
{
   int i;
   for (i = 0; i < size; i++)
   {
      v1[i] = 0.0;
   }
}

// Zeroes a complex vector. 
template <typename T> inline void zero(std::complex<T>* v1, int size)
{
  // Initialize.
  int i;
  
  for (i = 0; i < size; i++)
  {
    v1[i] = 0.0;
  }
  
}

// Random gaussian vector.
template<typename T> inline void gaussian(T* v1, int size, std::mt19937 &generator)
{
    // Generate a normal distribution.
    std::normal_distribution<> dist(0.0, 1.0);
    int i;
    
    for (i = 0; i < size; i++)
    {
        v1[i] = static_cast<T>(dist(generator));
    }
}

// Random gaussian vector, random in real and imag.
template <typename T> inline void gaussian(std::complex<T>* v1, int size, std::mt19937 &generator)
{
    // Generate a normal distribution.
    std::normal_distribution<> dist(0.0, 1.0);

    // Initialize.
    int i;

    for (i = 0; i < size; i++)
    {
        v1[i] = std::complex<T>(static_cast<T>(dist(generator)), static_cast<T>(dist(generator)));
    }
}

// Copy v2 into v1.
template<typename T> inline void copy(T* v1, T* v2, int size)
{
  // Initialize.
  int i;
  
  for (i = 0; i < size; i++)
  {
    v1[i] = v2[i];
  }
}

// Copy v2 into v1.
template<typename T> inline void copy(std::complex<T>* v1, std::complex<T>* v2, int size)
{
  // Initialize.
  int i;
  
  for (i = 0; i < size; i++)
  {
    v1[i] = v2[i];
  }
}

// Implement cxpy, v2 += v1.
template<typename T> inline void cxpy(T* v1, T* v2, int size)
{
  // Initialize.
  int i;
  for (i = 0; i < size; i++)
  {
    v2[i] += v1[i];
  }
}

// Implement cxpy, v2 += v1.
template<typename T> inline void cxpy(std::complex<T>* v1, std::complex<T>* v2, int size)
{
  // Initialize.
  int i;
  for (i = 0; i < size; i++)
  {
    v2[i] += v1[i];
  }
}


// Computes v1 dot v2.
template<typename T> inline T dot(T* v1, T* v2, int size)
{
  // Initialize.
  int i;
  T res = (T)0.0;
  
  for (i = 0; i < size; i++)
  {
    res = res + v1[i]*v2[i];
  }
  
  return res;
}

// computes conj(v1) dot v2.
template <typename T> inline std::complex<T> dot(std::complex<T>* v1, std::complex<T>* v2, int size)
{
  // Initialize.
  int i;
  std::complex<T> res = (T)0.0;
  
  for (i = 0; i < size; i++)
  {
    res = res + conj(v1[i])*v2[i];
  }
  
  return res;
}

template <typename T>
inline T norm2sq(T* v1,  int size)
{
  // Initialize.
  int i;
  T res = (T)0.0;
  
  for (i = 0; i < size; i++)
  {
    res = res + v1[i]*v1[i];
  }
  
  return res;
}

template <typename T>
inline T norm2sq(std::complex<T>* v1, int size)
{
  // Initialize.
  int i;
  T res = (T)0.0;
  
  for (i = 0; i < size; i++)
  {
    res = res + real(conj(v1[i])*v1[i]);
  }
  
  return res;
}

// Return |v1 - v2|
template <typename T>
inline T diffnorm2sq(T* v1, T* v2, int size)
{
    int i;
    T res = (T)0.0;
    for (i = 0; i < size; i++)
    {
        res = res + (v1[i] - v2[i])*(v1[i] - v2[i]);
    }
    return res;
}

template <typename T>
inline T diffnorm2sq(std::complex<T>* v1, std::complex<T>* v2, int size)
{
    int i;
    T res = (T)0.0;
    for (i = 0; i < size; i++)
    {
        res = res + real(conj(v1[i] - v2[i])*(v1[i] - v2[i]));
    }
    return res;
}
    

template <typename T>
inline void normalize(T* v1, int size)
{
    int i;
    T res = static_cast<T>(0.0);
    
    res = 1.0/sqrt(norm2sq<T>(v1, size));
    if (res > 0.0)
    {
        for (i = 0; i < size; i++)
        {
            v1[i] *= res;
        }
    }
}

template <typename T>
inline void normalize(std::complex<T>* v1, int size)
{
    int i;
    T res = static_cast<T>(0.0);
    
    res = 1.0/sqrt(norm2sq<T>(v1, size));
    if (res > 0.0)
    {
        for (i = 0; i < size; i++)
        {
            v1[i] *= res;
        }
    }
}

template <typename T>
inline void conj(T* v1, int size)
{
    return; // Trivial, it's real.
}

template <typename T>
inline void conj(std::complex<T>* v1, int size)
{
    int i;
    
    for (i = 0; i < size; i++)
    {
        v1[i] = conj(v1[i]);
    }
}

// Make vector v1 orthogonal to vector v2. 
template <typename T>
inline void orthogonal(T* v1, T* v2, int size)
{
    int i;
    T alpha; 
    
    alpha = -dot<T>(v2, v1, size)/norm2sq<T>(v2, size);
    
    for (i = 0; i < size; i++)
    {
        v1[i] = v1[i] + alpha*v2[i];
    }
}

// Make vector v1 orthogonal to vector v2. 
template <typename T>
inline void orthogonal(std::complex<T>* v1, std::complex<T>* v2, int size)
{
    int i;
    std::complex<T> alpha; 
    
    alpha = -dot<T>(v2, v1, size)/norm2sq<T>(v2, size);
    
    for (i = 0; i < size; i++)
    {
        v1[i] = v1[i] + alpha*v2[i];
    }
}

#endif // GENERIC_VECTOR

