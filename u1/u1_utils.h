// Copyright (c) 2017 Evan S Weinberg

#ifndef U1_UTILS
#define U1_UTILS

#include <complex>
#include <string>
#include <random>
using std::complex;
using std::string;

// Different gauge field types. 
enum gauge_create_type
{
  GAUGE_LOAD = 0,             // Load a gauge field.
  GAUGE_RANDOM = 1,           // Create a gauge field with deviation 1/sqrt(beta)
  GAUGE_UNIT = 2              // Use a unit gauge field.
};

// Load complex gauge field from file. 
// Based on Rich Brower's u1 gauge routines. 
// Reads in a U1 phase lattice from file, returns complex fields. 
// Rich's code has 'y' as the fast direction. Need to transpose!
void read_gauge_u1(complex<double>* gauge_field, int x_len, int y_len, string input_file);

// Write complex gauge field to file.
// Based on Rich Brower's u1 gauge routines.
// Writes a U1 phase lattice from file from complex fields.
// Rich's code has 'y' as the fast direction. Need to transpose!
void write_gauge_u1(complex<double>* gauge_field, int x_len, int y_len, string output_file);

// Create a unit gauge field.
// Just set everything to 1!
void unit_gauge_u1(complex<double>* gauge_field, int x_len, int y_len);

// Create a hot gauge field, uniformly distributed in -Pi -> Pi.
// mt19937 can be created+seeded as: std::mt19937 generator (seed1);
void rand_gauge_u1(complex<double>* gauge_field, int x_len, int y_len, std::mt19937 &generator);

// Create a gaussian gauge field with variance = 1/beta
// beta -> 0 is a hot start, beta -> inf is a cold start. 
// Based on code by Rich Brower, re-written for C++11.
void gauss_gauge_u1(complex<double>* gauge_field, int x_len, int y_len, std::mt19937 &generator, double beta);

// Generate a random gauge transform.
// mt19937 can be created+seeded as: std::mt19937 generator (seed1);
void rand_trans_u1(complex<double>* gauge_trans, int x_len, int y_len, std::mt19937 &generator);

// Apply a gauge transform:
// u_i(x) = g(x) u_i(x) g^\dag(x+\hat{i})
void apply_gauge_trans_u1(complex<double>* gauge_field, complex<double>* gauge_trans, int x_len, int y_len);

// Apply ape smearing with parameter \alpha, n_iter times.
// Based on code from Rich Brower
void apply_ape_smear_u1(complex<double>* smeared_field, complex<double>* gauge_field, int x_len, int y_len, double alpha, int n_iter);

// Get average plaquette
complex<double> get_plaquette_u1(complex<double>* gauge_field, int x_len, int y_len);

// Get the topological charge.
double get_topo_u1(complex<double>* gauge_field, int x_len, int y_len);
	
#endif // U1_UTILS