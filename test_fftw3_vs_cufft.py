import pyfftw
import cupy
import numpy

n0, n1, n2 = 2, 3, 4
r = pyfftw.empty_aligned((n0, n1, n2), dtype='f4')
F = pyfftw.empty_aligned((n0, n1, n2//2 + 1), dtype='complex64')

# create plans
plan1 = pyfftw.FFTW(r, F, direction='FFTW_FORWARD', axes=(0,1,2)) #, flags=('FFTW_MEASURE',), threads=1)
plan2 = pyfftw.FFTW(F, r, direction='FFTW_BACKWARD', axes=(0,1,2)) #, flags=('FFTW_MEASURE',), threads=1)

# set the data
ntot = n0 * n1 * n2
r[:] = numpy.array(range(ntot)).reshape((n0, n1, n2))
d_r = cupy.array(r) # copy to device
print(f'r={r}\nd_r={d_r.get()}')

# apply forward FFT
plan1.execute()
d_F = cupy.fft.rfftn(d_r) #, norm='backward')
print(f'F={F}\nd_F={d_F.get()}')

# apply backward FFT
plan2.execute()
d_r[:] = cupy.fft.irfftn(d_F) #, norm='forward')

# remove the normalisation
d_r *= ntot

print(f'r={r}\nd_r={d_r.get()}')


# check
print(f'pyfftw checksum: {numpy.sum(r)}')
print(f'cupy checksum: {numpy.sum(d_r.get())}')


