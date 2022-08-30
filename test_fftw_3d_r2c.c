/* Example showing the use of FFTW. 

ml FFTW
g++ -O2 test_fftw_3d_r2c.c -o test_fftw_3d_r2c -L $EBROOTFFTW/lib -lfftw3f -lfftw3

*/

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>

// includes, project
#include <fftw3.h>

typedef fftwf_complex Complex;
typedef float Real;


////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void runTest(int argc, char** argv);

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv)
{
    runTest(argc, argv);
}

////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
void runTest(int argc, char** argv)
{

    int SIGNAL_SIZE = 128;
    int NUM_ITERS = 10;
    if (argc >= 2) {
        SIGNAL_SIZE = atoi(argv[1]);
        if (argc >= 3) {
            NUM_ITERS = atoi(argv[2]);
        }
    }
    
    printf("FFTW 3d r2c size %d num iters %d...\n", SIGNAL_SIZE, NUM_ITERS);
    
    int ntot = SIGNAL_SIZE * SIGNAL_SIZE * SIGNAL_SIZE;
    int ntot2 = SIGNAL_SIZE * SIGNAL_SIZE * (SIGNAL_SIZE/2 + 1);
    int mem_size = ntot * sizeof(Real);
    int mem_size2 = ntot2 * sizeof(Complex);

    // Allocate host memory for the signal
    Real* h_signal = (Real*) fftwf_malloc(mem_size);
    Complex* d_signal = (Complex*) fftwf_malloc(mem_size2);
    Real* h_signal2 = (Real*) fftwf_malloc(mem_size);    

    // FFTW plan
    fftwf_plan p = fftwf_plan_dft_r2c_3d(SIGNAL_SIZE, SIGNAL_SIZE, SIGNAL_SIZE, h_signal, d_signal, FFTW_MEASURE);
    fftwf_plan p2 = fftwf_plan_dft_c2r_3d(SIGNAL_SIZE, SIGNAL_SIZE, SIGNAL_SIZE, d_signal, h_signal2, FFTW_MEASURE);
    
    // Initalize the memory for the signal
    for (unsigned int i = 0; i < ntot; ++i) {
        h_signal[i] = 0;
    }
    h_signal[0] = 1;

    clock_t time_beg = clock();
    
    for (int iter = 0; iter < NUM_ITERS; ++iter) {
        fftwf_execute(p);    
        fftwf_execute(p2);
        // Normalize
        for (unsigned int i = 0; i < ntot; ++i) {
            h_signal2[i] /= ntot;
        }
    }

    clock_t time_end = clock();
    printf("fwd -> bwd transform time: %lf secs\n", ((double)(time_end - time_beg))/CLOCKS_PER_SEC); 
        
    // Check
    float error = 0;
    for (unsigned int i = 0; i < ntot; ++i) {
        error += h_signal[i] - h_signal2[i];
    }
    printf("error: %g\n", error);

    //Destroy CUFFT context
    fftwf_destroy_plan(p2);
    fftwf_destroy_plan(p);

    // cleanup memory
    fftwf_free(h_signal);
    fftwf_free(d_signal);
    fftwf_free(h_signal2);
}

