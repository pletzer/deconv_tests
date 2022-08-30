/* Example showing the use of FFTW. 

ml shared NVHPC cuda11.0/fft
nvcc -O2 test_cufftw_3d_r2c.cu -o test_cufftw_3d_r2c -L $EBROOTCUDA/lib -lcufftw -lcufft -lculibos
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


typedef fftwf_complex Complex;
typedef float Real;

// Normalization kernel
__global__ void normalize(int n, Real* arr) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n) arr[i] /= (Real)(n); 
}

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
    
    printf("cuFFTW 3d r2c size %d num iters %d...\n", SIGNAL_SIZE, NUM_ITERS);

    int ntot = SIGNAL_SIZE * SIGNAL_SIZE * SIGNAL_SIZE;
    int ntot2 = SIGNAL_SIZE * SIGNAL_SIZE * (SIGNAL_SIZE/2 + 1);
    int mem_size = ntot * sizeof(Real);
    int mem_size2 = ntot2 * sizeof(Complex);

    // Allocate host memory for the signal
    Real* h_signal = (Real*) fftwf_malloc(mem_size);
    Real* h_signal2 = (Real*) fftwf_malloc(mem_size);

    // Initalize the memory for the signal
    for (unsigned int i = 0; i < ntot; ++i) {
        h_signal[i] = 0;
    }
    h_signal[0] = 1;    

    // Data on the device
    Real* d_signal;
    cudaMalloc((void**)&d_signal, mem_size);

    Complex* c_signal;
    cudaMalloc((void**)&c_signal, mem_size2);

    // FFTW plan
    fftwf_plan p = fftwf_plan_dft_r2c_3d(SIGNAL_SIZE, SIGNAL_SIZE, SIGNAL_SIZE, d_signal, c_signal, FFTW_MEASURE);
    fftwf_plan p2 = fftwf_plan_dft_c2r_3d(SIGNAL_SIZE, SIGNAL_SIZE, SIGNAL_SIZE, c_signal, d_signal, FFTW_MEASURE);

    clock_t time_beg = clock();

    // Copy host memory to device
    cudaMemcpy(d_signal, h_signal, mem_size,
               cudaMemcpyHostToDevice);
               
    for (int iter = 0; iter < NUM_ITERS; ++iter) {
        fftwf_execute(p);
        fftwf_execute(p2);
        normalize<<<(ntot + 255)/256, 256>>>(ntot, d_signal);
    }
    
    // Copy device memory to host
    cudaMemcpy(h_signal2, d_signal, mem_size,
               cudaMemcpyDeviceToHost);
    
    clock_t time_end = clock();
    printf("fwd -> bwd transform time: %lf secs\n", ((double)(time_end - time_beg))/CLOCKS_PER_SEC); 
    
    // Check
    float error = 0;
    for (unsigned int i = 0; i < ntot; ++i) {
        error += h_signal[i] - h_signal2[i];
    }
    printf("error: %g\n", error);

    //Destroy plans
    fftwf_destroy_plan(p2);
    fftwf_destroy_plan(p);

    // cleanup memory
    cudaFree(d_signal);
    cudaFree(c_signal);
    fftwf_free(h_signal);
    fftwf_free(h_signal2);

}

