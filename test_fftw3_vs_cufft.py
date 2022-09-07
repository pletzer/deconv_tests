import pyfftw
import cupy
import numpy

n0, n1, n2 = 2, 3, 4
r = pyfftw.empty_aligned((n0, n1, n2), dtype='f4')
F = pyfftw.empty_aligned((n0, n1, n2//2 + 1), dtype='complex64')


# create plans
plan1 = pyfftw.FFTW(r, F, direction='FFTW_FORWARD') #, axes=range(len(r.shape)), flags=('FFTW_MEASURE',), threads=1)
plan2 = pyfftw.FFTW(F, r, direction='FFTW_BACKWARD') #, axes=range(len(r.shape)), flags=('FFTW_MEASURE',), threads=1)

# set the data
ntot = n0 * n1 * n2
r[:] = numpy.array(range(ntot)).reshape((n0, n1, n2))

# apply forward FFT
plan1.execute()

# apply backward FFT
plan2.execute()

# check
print(f'pyfftw checksum: {numpy.sum(r)}')

