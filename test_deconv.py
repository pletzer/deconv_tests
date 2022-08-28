import numpy
from matplotlib import pyplot
import defopt
import functools
import operator
import pyfftw
from scipy.fftpack import ifftshift

def plot_data(field, title):
    fig, ax = pyplot.subplots()
    pyplot.pcolor(field)
    ax.set_aspect('equal')
    pyplot.title(title)
    pyplot.show()

class FFTAbstract(object):
    def __init__(self, *args, **kwargs):
        raise(NotImplementedError, 'need to derive from this class')
    def forward(self, arr):
        raise(NotImplementedError, 'need to derive from this class')
    def backward(self, arr):
        raise(NotImplementedError, 'need to derive from this class')

class FFT_numpy(FFTAbstract):

    def __init__(self, *args, **kwargs):
        pass

    def forward(self, arr):
        return numpy.fft.fftn(arr)

    def backward(self, arr):
        return numpy.fft.ifftn(arr)

class FFT_fftw(FFTAbstract):

    def __init__(self, arr, *args, **kwargs):
        self._a = pyfftw.empty_aligned(arr.shape, dtype='complex64')

    def forward(self, arr):
        self._a[:] = arr
        return pyfftw.interfaces.numpy_fft.fftn(self._a)

    def backward(self, arr):
        self._a[:] = arr
        return pyfftw.interfaces.numpy_fft.ifftn(self._a)


def main(*, nsizes: str='(32, 32)', plot: bool=False):

    nsizes = eval(nsizes)

    field = numpy.zeros(nsizes, numpy.float32)

    # create a blob
    xs = [numpy.linspace(0., 1., n) for n in nsizes]
    radius = 0.1
    center = numpy.array([0.3, 0.6])
    xx = numpy.meshgrid(xs[0], xs[1])
    inds = numpy.where( (xx[0] - center[0])**2 + (xx[1] - center[1])**2 < radius**2 )
    field[inds] = 1.0

    # blurr it
    fft_obj = FFT_fftw(field) #FFT_numpy()
    fft_field = fft_obj.forward(field)
    print(f'...fft_field = {fft_field}')

    one_over_r = 1.0/numpy.sqrt((xx[0]**2 + xx[1]**2) + 0.1**2)
    kernel = fft_obj.forward(one_over_r)

    fft_field_blurred = fft_field * kernel

    # back to real space
    field_blurred = numpy.real(fft_obj.backward(fft_field_blurred)) # or should it be absolute?

    decon_field = numpy.real( fft_obj.backward(fft_field_blurred / kernel) )
    # plot
    if plot:
        plot_data(field, title='original')
        plot_data(field_blurred, title='blurred')
        plot_data(decon_field, 'after deconvolution')

    # save the arrays
    sizes = functools.reduce(operator.__add__, [f'{n}x' for n in nsizes])
    numpy.save(f'original_{sizes}.npy', field)
    numpy.save(f'blurred_{sizes}.npy', field_blurred)

if __name__ == '__main__':
    defopt.run(main)




