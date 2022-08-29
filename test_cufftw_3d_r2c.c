/* Example showing the use of FFTW. 

ml shared NVHPC cuda11.0/fft
nvcc test_cufftw_3d_r2c.c -o test_cufftw_3d_r2c -L $EBROOTCUDA/lib -lcufftw -lcufft -lculibos
srun --gpus-per-node=A100:1 ./test_cufftw_3d_r2c

*/

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, project
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <cufftw.h>
#include <time.h>


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
    
    printf("cuFFTW 3d r2c size %d...\n", SIGNAL_SIZE);

    int ntot = SIGNAL_SIZE * SIGNAL_SIZE * SIGNAL_SIZE;
    int ntot2 = SIGNAL_SIZE * SIGNAL_SIZE * (SIGNAL_SIZE/2 + 1);
    int mem_size = ntot * sizeof(double);
    int mem_size2 = ntot2 * sizeof(Complex);

    // Allocate host memory for the signal
    double* h_signal = (double*) fftw_malloc(mem_size);
    double* h_signal2 = (double*) fftw_malloc(mem_size);

    // Initalize the memory for the signal
    for (unsigned int i = 0; i < ntot; ++i) {
        h_signal[i] = 0;
    }
    h_signal[0] = 1;    

    // Data on the device
    double* d_signal;
    cudaMalloc((void**)&d_signal, mem_size);

    Complex* c_signal;
    cudaMalloc((void**)&c_signal, mem_size2);

    // FFTW plan
    fftw_plan p = fftw_plan_dft_r2c_3d(SIGNAL_SIZE, SIGNAL_SIZE, SIGNAL_SIZE, d_signal, c_signal, FFTW_ESTIMATE);
    fftw_plan p2 = fftw_plan_dft_c2r_3d(SIGNAL_SIZE, SIGNAL_SIZE, SIGNAL_SIZE, c_signal, d_signal, FFTW_ESTIMATE);

    clock_t time_beg = clock();

    // Copy host memory to device
    cudaMemcpy(d_signal, h_signal, mem_size,
               cudaMemcpyHostToDevice);
    
    fftw_execute(p);
    fftw_execute(p2);
    
    // Copy device memory to host
    cudaMemcpy(h_signal2, d_signal, mem_size,
               cudaMemcpyDeviceToHost);
    
    clock_t time_end = clock();
    printf("fwd -> bwd transform time: %lf secs\n", ((double)(time_end - time_beg))/CLOCKS_PER_SEC); 
    
    // Check
    float error = 0;
    for (unsigned int i = 0; i < ntot; ++i) {
        error += h_signal[i] - h_signal2[i]/ntot; // note: normalization
    }
    printf("error: %g\n", error);

    //Destroy plans
    fftw_destroy_plan(p2);
    fftw_destroy_plan(p);

    // cleanup memory
    cudaFree(d_signal);
    cudaFree(c_signal);
    fftw_free(h_signal);
    fftw_free(h_signal2);

}

