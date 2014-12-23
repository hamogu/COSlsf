# 
#  Copyright (C) 2011 Smithsonian Astrophysical Observatory
#
#
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License along
#  with this program; if not, write to the Free Software Foundation, Inc.,
#  51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
#
from sherpa.astro.ui import *
import numpy as np
import pytest
import COSlsf

def test_fwhm():
    '''Test the FWHM of convolved models 
    
    First generate some artifical data
    test need to be close to final
    to make sure Sherpa finds a good fit.
    
    The FWHM of the a convolved Gaussian should be smaller than that of a 
    stand-alone Gaussian. The difference has to be 6.5 pix.
    ''' 
    
    disp = 0.01223
    x = np.arange(1430.,1435., 0.01223)
    y = np.random.normal(1e-13,0.3e-13, x.shape)
    sigma = 0.2
    mu = 1433.
    ampl = 1e-12
    y += ampl*np.exp(-(x-mu)**2./(0.5 * sigma**2.))
    yerr = np.ones_like(y) * 3e-14
    #return x, y, yerr, disp

    set_method('simplex')
    x, y, yerr, disp = make_some_data()
    load_arrays(1,x,y,yerr)
    set_model('tabG160M(const1d.c+gauss1d.g)')
    g.fwhm = 2.* sigma
    g.pos = mu
    g.ampl = ampl
    fit(1)
    #plot_fit()

    load_arrays(2,x,y,yerr)
    set_model(2, 'const1d.c2+gauss1d.g2')
    g2.fwhm = 2. * sigma
    g2.pos = mu
    g2.ampl = ampl
    fit(2)
    #plot_fit(2, overplot = True)

    # Input value use np.random, so range of value is possible
    sigma_line = 6.5*disp / 2.35  #2.35 convert from FWHM to sigma
    # expected width of Gaussian LSF core is >6.5 pixel
    d_lsf = g2.fwhm.val - g.fwhm.val
    assert (d_lsf > (.9 * sigma_line)) and (d_lsf < (1.1 * sigma_line))



def test_errors():
    '''set model outside applicable range'''
    x = np.arange(100)
    load_arrays(1,x,x)
    set_model('tabG160M(const1d.c+gauss1d.g)')
    with pytest.raises(ValueError):
        fit()

def test_xbinwarning(capsys):
    x = np.arange(1550., 1552., 2.*0.01223)
    load_arrays(1,x,x)
    set_model('tabG160M(const1d.c+gauss1d.g)')
    fit()
    out, err = capsys.readouterr()
    assert 'LSF convolution requires x to be binned in pixels.' in out

asciitableshape = {'tabG130M': (102, 8), 'tabG160M': (102, 8),
                   'tabG140L': (102, 13), 'tabNUV': (304, 17),
                   'empG130M': (202, 8), 'empG160M': (202, 8)}

def test_asciitableshape():
    '''Do the right models have the right tables?'''
    for key in asciitableshape:
        assert get_model_component(key).kernel.lsf_tab.shape == asciitableshape[key]


def test_lsf():
    '''LSF amplitide and normalisation'''
    #data values types in from Tab 3 in COS ISR 2011-01
    lsfpeakval = {'tabG130M': [.0990, .1013, .1026, .1042, 1067., .1085], 
                  'empG130M': [.0961, .0984, .0997, .1013, .1038, .1057],
                  'tabG160M': [.1047, .1067, .1075, .1086, .1099, .1116, .1120], 
                  'empG160M': [.1020, .1041, .1051, .1063, .1077, .1095, .1199]}
    modrange = {'tabG130M': [1150., 1401.],
                'empG130M': [1150., 1401.],
                'tabG160M': [1450., 1751.],
                'empG160M': [1450., 1751.]}
    for mymodel in modrange.keys():
        m = get_model_component(mymodel)
        for i, wave in enumerate(np.arange(modrange[mymodel][0], modrange[mymodel][1], 50.)):
            x = np.arange(wave-2., wave+2., 0.01223)
            y = np.zeros_like(x)
            y[y.shape[0]/2] = 1.
            res = m.kernel.convolve(x,y)
            # check peak values
            assert abs(max(res-lsfpeakval[mymodel][i]) < 0.0001)
            # check flux conservation
            assert abs((res.sum() - y.sum()) < 0.0001)
