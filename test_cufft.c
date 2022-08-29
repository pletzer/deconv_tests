/* Example showing the use of CUFFT for FFT. 

ml shared NVHPC cuda11.0/fft
nvcc test_cufft.c -o test_cufft -L $EBROOTCUDA/lib -lcufft_static -lculibos

*/

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, project
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <cufft.h>

typedef float2 Complex;


////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void runTest(int argc, char** argv);

// The filter size is assumed to be a number smaller than the signal size
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
    printf("[simpleCUFFT] is starting...\n");
    int mem_size = SIGNAL_SIZE * sizeof(Complex);

    // Allocate host memory for the signal
    Complex* h_signal = (Complex*)malloc(sizeof(Complex) * SIGNAL_SIZE);
    // Initalize the memory for the signal
    for (unsigned int i = 0; i < SIGNAL_SIZE; ++i) {
        h_signal[i].x = 0;
        h_signal[i].y = 0;
    }
    h_signal[0].x = 1;

    // Allocate device memory for signal
    Complex* d_signal;
    cudaMalloc((void**)&d_signal, mem_size);
    // Copy host memory to device
    cudaMemcpy(d_signal, h_signal, mem_size,
               cudaMemcpyHostToDevice);
    

    // CUFFT plan
    cufftHandle plan;
    cufftPlan1d(&plan, mem_size, CUFFT_C2C, 1);

    // Transform signal and kernel
    printf("Transforming signal cufftExecC2C\n");
    cufftExecC2C(plan, (cufftComplex *)d_signal, (cufftComplex *)d_signal, CUFFT_FORWARD);

    // Transform signal back
    printf("Transforming signal back cufftExecC2C\n");
    cufftExecC2C(plan, (cufftComplex *)d_signal, (cufftComplex *)d_signal, CUFFT_INVERSE);

    // Copy device memory to host
    cudaMemcpy(h_signal, d_signal, mem_size,
               cudaMemcpyDeviceToHost);

    //Destroy CUFFT context
    cufftDestroy(plan);

    // cleanup memory
    free(h_signal);
    cudaFree(d_signal);

}

