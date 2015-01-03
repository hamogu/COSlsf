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
import os
import glob
import numpy as np

from sherpa.models import Model, CompositeModel,  ArithmeticModel#, modelCacher1d
from sherpa.utils import interpolate
from sherpa.utils.err import PSFErr
from sherpa.ui import add_model, _session

from sherpa.instrument import ConvolutionKernel

# The following is implemeted in C and thus faster than python.
from sherpa.astro.utils import rmf_fold


#import logging
#warning = logging.getLogger(__name__).warning
#info = logging.getLogger(__name__).info

class Kernel(Model):
    '''contains the convolution kernal and the convolution code
    '''
    def __init__(self, lsf_tab, disp, name):
        self.lsf_tab = lsf_tab
        self.disp = disp
        # attributes need to be initialized as such in classes derived
        # from NoAttributesAfterInit
        self._rsp = 0.
        self._grp = 0.
        self._fch = 0.
        self._nch = 0.

        self.cache_x = np.zeros(1)
        # Make mean value to 0
        # needed, because some tables are given from -n to +n, others from 1 to 2n+1 
        self.shift = self.lsf_tab[1:,0] - np.mean(self.lsf_tab[1:,0])
        self.m = 2 * int(max(self.shift)) + 1
        Model.__init__(self, name)

    def calc(self, pl, pr, rhs, *args, **kwargs):

        args = list(args)    # tuples are immutable!
        # add m/2 elements to the wavelength array on each side
        # to avoid edge effects when folding with the LSF
        args[0] = interpolate(np.arange(-(self.m/2), len(args[0]) + self.m/2-0.1), np.arange(len(args[0])), args[0])

        flux = rhs(pr, *args, **kwargs)

        return self.convolve(args[0], flux)[self.m/2 : -(self.m/2)]


    def convolve(self, wave, flux):
        '''Convolve the `flux` with the COS LSF.
        
        The LSF depends on the wavelength, thus `np.convolve` is not
        sufficient. We make a band matrix which holds the value for the LSF
        from -n to +n pixels. We then shift the flux from -n to +n pixels
        and multiply with the LSF in each position.
        
        Parameters
        ----------
        wave : ndarray
            wavelength (assumed to be in Ang)
            
        flux : ndarray
            in arbitrary units
        
        Returns
        -------
        newflux : ndarray
            flux convolved with the appropriate HST/COS LSF
        '''
        if len(wave) != len(flux):
            raise ValueError('Wave and Flux vector must have equal number of elements')
        # cache the _rsp, if wave did not change compared to the last call
        # This will generally be the case in Sherpa fits
        if ((wave.shape != self.cache_x.shape) or 
             not np.allclose(wave, self.cache_x)):
            self.make_rsp(wave)
        
        return rmf_fold(flux, self._grp, self._fch, self._nch, self._rsp, len(wave), 1)
    
    def make_rsp(self, wave):
        '''Interpolate the COS LSF for all wavelengths
        
        Parameters
        ----------
        wave : ndarray
            wavelength (assumed to be in Ang)
        '''
        # test if delta_lambda in wavelengths is as expected for this grating 
        for disp in self.disp:
            if max(abs(np.diff(wave) - disp)) < (disp / 100.):
                break
        else:
            print('LSF convolution requires x to be binned in pixels.')
            print('delta(x) differs from value given in COS IHB.')
        # first line of table is header values
        wavelen = self.lsf_tab[0, 1:]
        if ((min(wave) < 0.9 * min(wavelen)) or (max(wave) > 1.1 * max(wavelen))):
            raise ValueError('COS LSF in this model is only tabulated {0:4.0f} to  {1:0.0f}'.format(min(wavelen), max(wavelen)))

        # rest of table is data values
        # 1. Interpolation: Get lsf_tab on pixel grid
        # interpolate to full pixels, if given in fractional pixels
        lsftab = np.zeros((self.m, len(wavelen)))
        for i in range(len(wavelen)):
            lsftab[:,i] = interpolate(np.arange(-(self.m/2), self.m/2+0.1, dtype = np.int), self.shift, self.lsf_tab[1:,i+1])

        # 2. Interpolation: make lsf for each lambda
        n = len(wave)
        self._rsp = np.zeros((n, self.m))
        for i in np.arange(-(self.m/2), self.m/2+0.1, dtype = np.int):
            pix = interpolate(wave, wavelen, lsftab[self.m/2+i,:])
            self._rsp[:, self.m/2 + i] = pix
        
        if np.isfinite(self._rsp).sum() != self._rsp.size:
            raise ValueError('Response matrix holds nan or inf values!')
        # 3. Reformat: Bring matix in same shape as Sherpa would do for RMFs
        # I use the same naming convention here as in Sherpa `DataRMF` objects
        self._grp = np.ones(n)
        self._fch = np.clip(np.arange(1,n+1)-(self.m/2),1, np.inf)
        
        # set unused matix elements to nan - that allows a very simple
        # way to flatten the array
        for i in range(self.m/2):
            self._rsp[i,0:self.m/2-i] = np.nan
            self._rsp[-(i+1),-(self.m/2)+i:self.m] = np.nan
        self._nch = np.sum(np.isfinite(self._rsp),axis = 1)
        self._rsp = self._rsp[np.isfinite(self._rsp)].flatten()

        self.cache_x = wave

class ConvolutionModel(CompositeModel, ArithmeticModel):
    '''This is a simplified version of sherpa.instrument.ConvolutionModel
    
    Really, sherpa.instrument.ConvolutionModel could be derived from this class.
    '''
    def __init__(self, lhs, rhs):
        self.lhs = lhs
        self.rhs = rhs
	CompositeModel.__init__(self,
                                ('%s(%s)' %
                                 (self.lhs.name, self.rhs.name)),
                                (self.lhs, self.rhs))


    def calc(self, p, *args, **kwargs):
        nlhs = len(self.lhs.pars)
	return self.lhs.calc(p[:nlhs], p[nlhs:],
                             self.rhs.calc, *args, **kwargs)


class CosLsf(ConvolutionKernel):
    '''Factory for COS LSF convolution models'''
    def __init__(self, kernel, name='COSLSF'):
        self.kernel = kernel
        self.name = name
        #self.__name__ = name
        Model.__init__(self, name)
    
    def __call__(self, model):
        if self.kernel is None:
            raise PSFErr('notset')
        kernel = self.kernel
        return ConvolutionModel(kernel, model)


# From the COS IHB
# dispersion in Ang / pixel
dispersion = {'G130M': [0.00997], 'G140L': [0.0803], 'G160M': [0.01223], 'NUV': [0.033, 0.037, 0.040, 0.39], 'G185M': [0.037], 'G225M': [0.033], 'G285M': [0.04], 'G230L': [0.39]}

'''On import this module will detect all data files in lsf/
and automatically generate SHERPA models for that.
'''

# List of all data files in the lsf directory
datlist = glob.glob(os.path.join(os.path.dirname(__file__), 'lsf', '*.dat'))

lsf_names = []

for filename in datlist:
    
    modelname = os.path.basename(filename).split('.')[0]
    print('Adding '+ modelname +' to Sherpa')
    for grating in dispersion.keys():
        # If one grating matches, generate a class for that
        if grating in filename:
            disp = dispersion[grating]
            break
    else:
        print('Grating in ' + modelname + ' not recognized.')
    lsf_tab = np.loadtxt(filename)
    this_kern = Kernel(lsf_tab, disp, modelname)
    lsf = CosLsf(this_kern, modelname)
    _session._add_model_component(lsf)
    lsf_names.append(modelname)

    # Add to module level namespace
    globals()[modelname] = lsf
