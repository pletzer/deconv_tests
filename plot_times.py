from matplotlib import pyplot as plt
import pandas

data =  {
    'n': [100, 200, 300, 400, 500],
	'broadwell': [0.470000,5.08, 19.47, 44.63, 101.33],
	'milan': [0.28, 2.53, 11.55, 21.87, 46.43],
	'A100': [0.02, 0.19, 0.48, 1.44, 3.50]
}

plt.plot(data['n'], data['broadwell'])
plt.plot(data['n'], data['milan'])
plt.plot(data['n'], data['A100'])
plt.legend(['broadwell/FFTW', 'milan/FFTW', 'A100/cuFFT'])
plt.title('3d FFT fwd & bwd 40 iters')
plt.xlabel('size')
plt.ylabel('time s')
plt.show()
