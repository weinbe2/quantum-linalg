// Copyright (c) 2017 Evan S Weinberg

#include <fstream>
#include <cmath>
#include <complex>
#include <random>

using std::complex; 
using std::fstream;
using std::ios;
using std::ios_base;

#include "generic_vector.h"
#include "u1_utils.h"

#ifndef PI
#define PI 3.14159265358979323846
#endif

// Load complex gauge field from file. 
// Based on Rich Brower's u1 gauge routines. 
// Reads in a U1 phase lattice from file, returns complex fields. 
// Rich's code has 'y' as the fast direction. Need to transpose!
void read_gauge_u1(complex<double>* gauge_field, int x_len, int y_len, string input_file)
{
   double phase_tmp;   
   fstream in_file;
   
   in_file.open(input_file,ios::in); 
   for(int x =0;x< x_len;x++)
   {
      for(int y =0;y< y_len;y++)
      {
         for(int mu=0; mu<2; mu++)
         {
            in_file >> phase_tmp;
            gauge_field[y*2*x_len+x*2+mu] = polar(1.0,phase_tmp);
            //cout << polar(1.0, phase_tmp) << "\n";
         }
      }
   }
   in_file.close(); 
   
   return;
}

// Write complex gauge field to file.
// Based on Rich Brower's u1 gauge routines.
// Writes a U1 phase lattice from file from complex fields.
// Rich's code has 'y' as the fast direction. Need to transpose!
void write_gauge_u1(complex<double>* gauge_field, int x_len, int y_len, string output_file)
{
    double phase_tmp;
    fstream out_file;
    
    out_file.open(output_file, ios::in|ios::out|ios::trunc);
    out_file.setf(ios_base::fixed, ios_base::floatfield);
    out_file.precision(20);
    
    for(int x =0;x< x_len;x++)
    {
        for(int y =0;y< y_len;y++)
        {
            for(int mu=0; mu<2; mu++)
            {
                phase_tmp = arg(gauge_field[y*2*x_len+x*2+mu]);
                out_file << phase_tmp << "\n";
            }
        }
    }
    out_file.close(); 
}

// Create a unit gauge field.
// Just set everything to 1!
void unit_gauge_u1(complex<double>* gauge_field, int x_len, int y_len)
{
  constant<double>(gauge_field, 1.0, 2*x_len*y_len);
}

// Create a hot gauge field, uniformly distributed in -Pi -> Pi.
// mt19937 can be created+seeded as: std::mt19937 generator (seed1);
void rand_gauge_u1(complex<double>* gauge_field, int x_len, int y_len, std::mt19937 &generator)
{
  random_uniform<double>(gauge_field, 2*x_len*y_len, generator, -PI, PI);
  polar<double>(gauge_field, 2*x_len*y_len);
}

// Create a gaussian gauge field with variance = 1/beta
// beta -> 0 is a hot start, beta -> inf is a cold start. 
// Based on code by Rich Brower, re-written for C++11.
void gauss_gauge_u1(complex<double>* gauge_field, int x_len, int y_len, std::mt19937 &generator, double beta)
{
  // Take abs value of beta.
  if (beta < 0) { beta = -beta; }

  // If beta is 0, just call a hot start.
  if (beta == 0)
  {
    rand_gauge_u1(gauge_field, x_len, y_len, generator);
  }
  else
  {
    // create phase.
    gaussian<double>(gauge_field, 2*y_len*x_len, generator, 1.0/sqrt(beta));
    // promote to U(1).
    polar<double>(gauge_field, 2*y_len*x_len);
  } 
}

// Generate a random gauge transform.
// mt19937 can be created+seeded as: std::mt19937 generator (seed1);
void rand_trans_u1(complex<double>* gauge_trans, int x_len, int y_len, std::mt19937 &generator)
{
  random_uniform<double>(gauge_trans, x_len*y_len, generator, -PI, PI);
  polar<double>(gauge_trans, x_len*y_len);
}

// Apply a gauge transform:
// u_i(x) = g(x) u_i(x) g^\dag(x+\hat{i})
void apply_gauge_trans_u1(complex<double>* gauge_field, complex<double>* gauge_trans, int x_len, int y_len)
{
    for (int y = 0; y < y_len; y++)
    {
        for (int x = 0; x < x_len; x++)
        {
            // Update x direction.
            gauge_field[2*x_len*y + 2*x] = gauge_trans[x_len*y + x]*gauge_field[2*x_len*y + 2*x]*conj(gauge_trans[x_len*y + (x+1)%x_len]);
            
            // Update y direction.
            gauge_field[2*x_len*y + 2*x + 1] = gauge_trans[x_len*y + x]*gauge_field[2*x_len*y + 2*x + 1]*conj(gauge_trans[x_len*((y+1)%y_len) + x]);
        }
    }
}

// Apply ape smearing with parameter \alpha, n_iter times.
// Based on code from Rich Brower
void apply_ape_smear_u1(complex<double>* smeared_field, complex<double>* gauge_field, int x_len, int y_len, double alpha, int n_iter)
{
	int i, x, y; 
	complex<double>* smeared_tmp = new complex<double>[x_len*y_len*2];
	
  copy<double>(smeared_tmp, gauge_field, x_len*y_len*2);
	
	// APE smearing: project back on U(1)
	for (i = 0; i < n_iter; i++)
	{
		for (y = 0; y < y_len; y++)
		{
			for (x = 0; x < x_len; x++)
			{
				// x link.
				smeared_field[y*x_len*2 + x*2 + 0] = smeared_tmp[y*x_len*2 + x*2 + 0];
				smeared_field[y*x_len*2 + x*2 + 0] += alpha*smeared_tmp[y*x_len*2+x*2+1]*smeared_tmp[((y+1)%y_len)*x_len*2 + x*2 + 0]*conj(smeared_tmp[y*x_len*2+((x+1)%x_len)*2+1]);
				smeared_field[y*x_len*2 + x*2 + 0] += alpha*conj(smeared_tmp[((y-1+y_len)%y_len)*x_len*2 + x*2 + 1])*smeared_tmp[((y-1+y_len)%y_len)*x_len*2 + x*2 + 0]*smeared_tmp[((y-1+y_len)%y_len)*x_len*2 + ((x+1)%x_len)*2 + 1];
				
				// y link.
				smeared_field[y*x_len*2 + x*2 + 1] = smeared_tmp[y*x_len*2 + x*2 + 1];
				smeared_field[y*x_len*2 + x*2 + 1] += alpha*conj(smeared_tmp[y*x_len*2 + ((x-1+x_len)%x_len)*2 + 0])*smeared_tmp[y*x_len*2 + ((x-1+x_len)%x_len)*2 + 1]*smeared_tmp[((y+1)%y_len)*x_len*2 + ((x-1+x_len)%x_len)*2 + 0];
				smeared_field[y*x_len*2 + x*2 + 1] += alpha*smeared_tmp[y*x_len*2 + x*2 + 0]*smeared_tmp[y*x_len*2 + ((x+1)%x_len)*2 + 1]*conj(smeared_tmp[((y+1)%y_len)*x_len*2 + x*2 + 0]);
			}
		}
		
		// Project back to U(1).
    arg(smeared_field, x_len*y_len*2);
    polar(x, x_len*y_len*2);
    copy(smeared_tmp, smeared_field, x_len*y_len*2);
	}
	
  copy<double>(smeared_field, smeared_tmp, x_len*y_len*2);
	
	delete[] smeared_tmp; 
}

// Get average plaquette. 
complex<double> get_plaquette_u1(complex<double>* gauge_field, int x_len, int y_len)
{
    complex<double> plaq = 0.0;
    complex<double> tmp_plaq = 0.0;
    for (int y = 0; y < y_len; y++)
    {
        for (int x = 0; x < x_len; x++)
        {
            tmp_plaq = gauge_field[2*x_len*y + 2*x]*
                           gauge_field[2*x_len*y + 2*((x+1)%x_len) + 1]*
                      conj(gauge_field[2*x_len*((y+1)%y_len) + 2*x])*
                      conj(gauge_field[2*x_len*y + 2*x + 1]);
            plaq += tmp_plaq;
        }
    }
    return plaq / ((double)(x_len*y_len));
    
}

// Get the topological charge.
double get_topo_u1(complex<double>* gauge_field, int x_len, int y_len)
{
	complex<double> w;
	double top = 0.0;
	
	for (int y = 0; y < y_len; y++)
	{
		for (int x = 0; x < x_len; x++)
		{
			w = gauge_field[y*x_len*2 + x*2 + 0]*gauge_field[y*x_len*2 + ((x+1)%x_len)*2 + 1]*conj(gauge_field[((y+1)%y_len)*x_len*2 + x*2 + 0])*conj(gauge_field[y*x_len*2 + x*2 + 1]);
			// top += imag(w);
			top += arg(w); // Geometric value is an integer?
		}
	}
	
	return 0.5*top/PI;
}


