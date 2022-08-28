import numpy
import defopt
import pyfftw
import pandas
import time


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

    MODEL_NAME = 'numpy'

    def __init__(self, *args, **kwargs):
        pass

    def forward(self, arr):
        return numpy.fft.fftn(arr)

    def backward(self, arr):
        return numpy.fft.ifftn(arr)

class FFT_fftw(FFTAbstract):

    MODEL_NAME = 'fftw'

    def __init__(self, arr, *args, **kwargs):
        self._a = pyfftw.empty_aligned(arr.shape, dtype='complex64')

    def forward(self, arr):
        self._a[:] = arr
        return pyfftw.interfaces.numpy_fft.fftn(self._a)

    def backward(self, arr):
        self._a[:] = arr
        return pyfftw.interfaces.numpy_fft.ifftn(self._a)


def compute(Model, nsizes):

    field = numpy.zeros(nsizes, numpy.float32)

    # create a blob
    xs = [numpy.linspace(0., 1., n) for n in nsizes]
    radius = 0.1
    center = numpy.array([0.3, 0.6])
    xx = numpy.meshgrid(xs[0], xs[1])
    inds = numpy.where( (xx[0] - center[0])**2 + (xx[1] - center[1])**2 < radius**2 )
    field[inds] = 1.0

    # blurr it
    fft_obj = Model(field)
    fft_field = fft_obj.forward(field)

    one_over_r = 1.0/numpy.sqrt((xx[0]**2 + xx[1]**2) + 0.1**2)
    kernel = fft_obj.forward(one_over_r)

    fft_field_blurred = fft_field * kernel

    # back to real space
    decon_field = numpy.real( fft_obj.backward(fft_field_blurred / kernel) )

    return decon_field


def main(*, nsizes: str='(32, 32)'):

    nsizes = eval(nsizes)

    data = {'model': [],
            'checksum': [],
            'time_s': []}
    for Model in FFT_numpy, FFT_fftw:
        tic = time.time()
        decon_field = compute(Model, nsizes=nsizes)
        t = time.time() - tic
        data['model'].append(Model.MODEL_NAME)
        chcksum = numpy.sum(numpy.fabs(decon_field))
        data['checksum'].append(chcksum)
        data['time_s'].append(t)

    df = pandas.DataFrame(data)
    print(df)


if __name__ == '__main__':
    defopt.run(main)




