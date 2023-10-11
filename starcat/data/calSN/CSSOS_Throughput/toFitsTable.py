from astropy.table import Table
from pylab import *

d = loadtxt('y_throughput.txt')

d[:, 0] = d[:, 0]

d[:, 1] = d[:, 1]

t = Table(d, names=('WAVELENGTH', 'SENSITIVITY'))

t.write('y.Throughput.fits', format='fits')

# d[:,0] = d[:,0]

# d[:,1] = d[:,1]*0.1

# t= Table(d, names=('WAVELENGTH', 'SENSITIVITY'))

# t.write('GI.Throughput.0st.fits', format='fits')


# d = loadtxt('GU_1st.dat')
