import numpy
from matplotlib import pyplot
import defopt
import functools
import operator

def plot_data(field, title):
    fig, ax = pyplot.subplots()
    pyplot.pcolor(field)
    ax.set_aspect('equal')
    pyplot.title(title)
    pyplot.show()


def main(*, nsizes: str='(32, 32)', plot: bool=False):

    nsizes = eval(nsizes)

    field = numpy.zeros(nsizes, numpy.float64)

    # create a blob
    xs = [numpy.linspace(0., 1., n) for n in nsizes]
    radius = 0.1
    center = numpy.array([0.3, 0.6])
    xx = numpy.meshgrid(xs[0], xs[1])
    inds = numpy.where( (xx[0] - center[0])**2 + (xx[1] - center[1])**2 < radius**2 )
    field[inds] = 1.0

    # blurr it
    fft_field = numpy.fft.fftn(field, )
    one_over_r = 1.0/numpy.sqrt((xx[0]**2 + xx[1]**2) + 0.1**2)
    kernel = numpy.fft.fftn(one_over_r)
    fft_field_blurred = fft_field * kernel

    # back to real space
    field_blurred = numpy.real(numpy.fft.ifftn(fft_field_blurred)) # or should it be absolute?

    # plot
    if plot:
        plot_data(field, title='original')
        plot_data(field_blurred, title='blurred')

    # save the arrays
    sizes = functools.reduce(operator.__add__, [f'{n}x' for n in nsizes])
    numpy.save(f'original_{sizes}.npy', field)
    numpy.save(f'blurred_{sizes}.npy', field_blurred)

if __name__ == '__main__':
    defopt.run(main)




