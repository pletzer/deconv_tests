/* Example showing the use of FFTW. 

ml shared NVHPC cuda11.0/fft
nvcc test_cufftw.c -o test_cufftw -L $EBROOTCUDA/lib -lcufftw -lcufft -lculibos
srun --gpus-per-node=A100:1 ./test_cufftw

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


typedef fftw_complex Complex;


////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void runTest(int argc, char** argv);

#define SIGNAL_SIZE        16

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
    printf("[cuFFTW] is starting...\n");
    int mem_size = SIGNAL_SIZE * sizeof(Complex);

    // Allocate host memory for the signal
    Complex* h_signal = (Complex*) fftw_malloc(sizeof(Complex) * SIGNAL_SIZE);
    Complex* h_signal2 = (Complex*) fftw_malloc(sizeof(Complex) * SIGNAL_SIZE);

    // Initalize the memory for the signal
    for (unsigned int i = 0; i < SIGNAL_SIZE; ++i) {
        h_signal[i][0] = 0;
        h_signal[i][1] = 0;
    }
    h_signal[0][0] = 1;    

    // Data on the device
    Complex* d_signal;
    Complex* d_signal2;
    cudaMalloc((void**)&d_signal, sizeof(Complex) * SIGNAL_SIZE);
    cudaMalloc((void**)&d_signal2, sizeof(Complex) * SIGNAL_SIZE);

    // Copy host memory to device
    cudaMemcpy(d_signal, h_signal, sizeof(Complex) * SIGNAL_SIZE,
               cudaMemcpyHostToDevice);
    
    // FFTW plan
    fftw_plan p = fftw_plan_dft_1d(SIGNAL_SIZE, d_signal, d_signal2, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_plan p2 = fftw_plan_dft_1d(SIGNAL_SIZE, d_signal2, d_signal, FFTW_BACKWARD, FFTW_ESTIMATE);
    
    fftw_execute(p);
    fftw_execute(p2);
    
    // Copy device memory to host
    cudaMemcpy(h_signal2, d_signal, sizeof(Complex) * SIGNAL_SIZE,
               cudaMemcpyDeviceToHost);
    
    
    // Normalize 
    for (
        unsigned int i = 0; i < SIGNAL_SIZE; ++i) {
        h_signal2[i][0] /= SIGNAL_SIZE;
        h_signal2[i][1] /= SIGNAL_SIZE;
    }
    
    // Check
    float error = 0;
    for (unsigned int i = 0; i < SIGNAL_SIZE; ++i) {
        error += h_signal[i][0] - h_signal2[i][0];
        error += h_signal[i][1] - h_signal2[i][1];
        printf("i = %d original = %f + i*%f returned = %f + i*%f\n",  i, h_signal[i][0], h_signal[i][1], h_signal2[i][0], h_signal2[i][1]);
    }
    printf("error: %g\n", error);

    //Destroy plans
    fftw_destroy_plan(p2);
    fftw_destroy_plan(p);

    // cleanup memory
    cudaFree(d_signal);
    cudaFree(d_signal2);
    fftw_free(h_signal);
    fftw_free(h_signal2);

}

