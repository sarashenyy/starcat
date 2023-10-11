import os

from pylab import *
from scipy import interpolate

module_dir = os.path.dirname(__file__)
data_dir = os.path.join(module_dir, 'data/calSN/')


def fun(a, b):
    h = 6.626e-34;
    c = 3e8;
    n_y = b / (h * c * 1.0e12 / a);
    return n_y


filter = ['NUV', 'u', 'g', 'r', 'i', 'z', 'y']


def calculateSkyPix(fil='g', zodi_mag=21., es_ratio=0.5, aperture=2.0, p_ang_size=0.074):
    throughput_f = loadtxt(data_dir + 'CSSOS_Throughput/' + fil + '_throughput.txt')
    thr_i = interpolate.interp1d(throughput_f[:, 0] / 10, throughput_f[:, 1]);

    f_s = 200
    f_e = 1100;
    delt_f = 0.5;

    data_num = int((f_e - f_s) / delt_f + 1);

    eff = zeros([data_num, 2])
    eff[:, 0] = arange(f_s, f_e + delt_f, delt_f);
    eff[:, 1] = thr_i(eff[:, 0]);

    efile = data_dir + 'sky/earthShine.dat';
    zfile = data_dir + 'sky/zodiacal.dat';
    edata = loadtxt(efile);
    zdata = loadtxt(zfile);
    zd_u = fun(zdata[:, 0], zdata[:, 1]);
    ed_u = fun(edata[:, 0], edata[:, 1]);

    zdi = interpolate.interp1d(zdata[:, 0], zd_u);
    edi = interpolate.interp1d(edata[:, 0], ed_u);

    sky_data = zeros([data_num, 3]);
    sky_data[:, 0] = arange(f_s, f_e + delt_f, delt_f);
    sky_data[:, 1] = edi(sky_data[:, 0]);
    sky_data[:, 2] = zdi(sky_data[:, 0]);

    flux_sky = trapz((sky_data[:, 1] * es_ratio + sky_data[:, 2] * pow(10, -0.4 * (zodi_mag - 22.1))) * eff[:, 1],
                     sky_data[:, 0]);
    flux_sky = flux_sky * p_ang_size * p_ang_size * pi * (aperture * aperture / 4);

    return flux_sky


def calculateSN(aperture=2.0, readout=5.0, dark=0.02, p_ang_size=0.074, sky=0.2, ratio=0.80, r80=0.15, ex_num=2, t=150,
                fil='g', ABmag=22.):
    throughput_f = loadtxt(data_dir + 'CSSOS_Throughput/' + fil + '_throughput.txt')
    thr_i = interpolate.interp1d(throughput_f[:, 0] / 10, throughput_f[:, 1]);

    f_s = 200
    f_e = 1100;
    delt_f = 0.5;

    data_num = int((f_e - f_s) / delt_f + 1);

    eff = zeros([data_num, 2])
    eff[:, 0] = arange(f_s, f_e + delt_f, delt_f);
    eff[:, 1] = thr_i(eff[:, 0]);

    wave = arange(f_s, f_e + delt_f, delt_f);
    wavey = ones(wave.shape[0]);

    r_pix = r80 / p_ang_size
    cnum = pi * r_pix * r_pix

    flux_a = 54799275581.04437 * pow(10.0, -0.4 * ABmag) * trapz(wavey * eff[:, 1] / wave, wave) * t * ex_num * pi * (
            aperture / 2) * (aperture / 2);
    sn = flux_a * ratio / sqrt(
        flux_a * ratio + sky * t * ex_num * cnum + dark * t * ex_num * cnum + readout * readout * cnum * ex_num);

    return sn

# ?example : calculate SN at specific exposure condition
# band = 'g'  # filter system : [NUV, u, g, r, i, z, y]
# ex_time = 150  # exposure time, typical time for main survey is 150s
# ex_num = 1  # the number of exposures
# mag = 24  # the input magnitude in CSST photometric system
#
# skyPix = calculateSkyPix(fil=band, zodi_mag=21.0)
# sn = calculateSN(sky=skyPix, ex_num=ex_num, t=ex_time, fil=band, ABmag=mag)
