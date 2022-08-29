/* Example showing the use of FFTW. 

ml FFTW
g++ -O4 test_fftw_3d.c -o test_fftw_3d -L $EBROOTFFTW/lib -lfftw3

*/

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>

// includes, project
#include <fftw3.h>

typedef fftw_complex Complex;


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
    if (argc >= 2) {
       SIGNAL_SIZE = atoi(argv[1]);
    }
    
    printf("FFTW 3d size %d...\n", SIGNAL_SIZE);
    
    int ntot = SIGNAL_SIZE * SIGNAL_SIZE * SIGNAL_SIZE;
    int mem_size = ntot * sizeof(Complex);

    // Allocate host memory for the signal
    Complex* h_signal = (Complex*) fftw_malloc(mem_size);
    Complex* d_signal = (Complex*) fftw_malloc(mem_size);
    Complex* h_signal2 = (Complex*) fftw_malloc(mem_size);

    // Initalize the memory for the signal
    for (unsigned int i = 0; i < ntot; ++i) {
        h_signal[i][0] = 0;
        h_signal[i][1] = 0;
    }
    h_signal[0][0] = 1;
    
    // Copy to the device
    

    // FFTW plan
    fftw_plan p = fftw_plan_dft_3d(SIGNAL_SIZE, SIGNAL_SIZE, SIGNAL_SIZE, h_signal, d_signal, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_plan p2 = fftw_plan_dft_3d(SIGNAL_SIZE, SIGNAL_SIZE, SIGNAL_SIZE, d_signal, h_signal2, FFTW_BACKWARD, FFTW_ESTIMATE);
    
    clock_t time_beg = clock();
    
    fftw_execute(p);
    fftw_execute(p2);
    
    clock_t time_end = clock();
    printf("fwd -> bwd transform time: %lf secs\n", ((double)(time_end - time_beg))/CLOCKS_PER_SEC); 
        
    // Check
    float error = 0;
    for (unsigned int i = 0; i < ntot; ++i) {
        error += h_signal[i][0] - h_signal2[i][0]/ntot; // note: normalization
        error += h_signal[i][1] - h_signal2[i][1]/ntot;
    }
    printf("error: %g\n", error);

    //Destroy CUFFT context
    fftw_destroy_plan(p2);
    fftw_destroy_plan(p);

    // cleanup memory
    fftw_free(h_signal);
    fftw_free(d_signal);
    fftw_free(h_signal2);
}

