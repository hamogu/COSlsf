'''This module defines classes, that can be used as SHERPA models.
They fold the modeled spectrum with the COS line spread function (LSF).
Different classes are supplied for indidivual gratings.

CosG130M, CosG140L, CosG160M and CosNUV are based on the 
Tabulated Theoretical Line Spread Functions  
(COS ISR 2009-01, Ghavamian et al.).

CosG130Memp and G160Memp are based on the 
Tabulated Empirical Line Spread Functions
(COS ISR 2011-01, Kriss).

See http://www.stsci.edu/hst/cos/performance/spectral_resolution
for more details.

Format of input tables::
    
    - The header line has `nan` as first element and wavelength values as
        column titles.
    - Tabulated Theoretical Line Spread Functions: These tables go from
        -n to +n pixels, in the case of NUV in fractional pixels
    - Tabulated Empirical Line Spread Functions: These go from 1 to 2n.
    
The tables are read such that the the resulting LSF is tabulated from
-n to +n assuming that the tables *give a symmetric region aroound the line center*.
'''

from sherpa.models import CompositeModel
from sherpa.utils import interpolate
from sherpa.ui import load_user_model

#import logging
#warning = logging.getLogger(__name__).warning
#info = logging.getLogger(__name__).info
import numpy as np
import os
import glob

class CosLsf(CompositeModel):
    '''Base class for all COS LSF model classes.

    An example usage in python could be:
    
        import sherpa.ui as ui
        import COSlsf 
        x = np.arange(2500,2700)
        y = np.ones(200)
        ui.load_arrays(1, x,y)
        ui.set_full_model(tabNUV(ui.const1d.c1 + ui.gauss1d.g1))

    `tabNUV` is a derived class for the LSF of the NUV gratings.
    Classes are generated dynamically upon import, a list of model classes
    is printed.
    '''
    def __init__(self, model, name = 'CosLsf'):
        self.model = model
        # attributes need to be initialized as such in classes derived
        # from NoAttributesAfterInit
        self.lsf_mat = 0.
        self.cache_x = np.zeros(1)
        # Make mean value to 0
        # needed, because some tables are given from -n to +n, others from 1 to 2n+1 
        self.shift = self.lsf_tab[1:,0] - np.mean(self.lsf_tab[1:,0])
        self.m = 2 * int(max(self.shift)) + 1

        CompositeModel.__init__(self, '%s * %s' % (name, model.name), model)


    def calc(self, *args, **kwargs):
        '''This method is called be SHERPA to evaluate models.
        
        The LSF has no free parameters, thus all arguments are passed
        to the enclosed models.
        
        However, to avoid edge effects, the wavelength vector is extended
        to both sides, before it is passed to the inner models. Thus,
        we can convolve the flux vector with the LSF and then crop the
        outer parts, which might be effected by edge effects.
        This requires, that the inner models can be calcualted for a larger
        range of x values.
        
        Parameters
        ----------
        All parameters are passed othe the inner models. See Sherpa
        documentation for details on which parameters are expected.
        `args[1]` is the wave vector.
        
        Returns
        -------
        flux : ndarray
            flux vector after appliing the LSF
        '''
        args = list(args)    # tuples are immutable!
        # add m/2 elements to the wavelength array on each side
        # to avoid edge effects when folding with the LSF
        args[1] = interpolate(np.arange(-(self.m/2), len(args[1]) + self.m/2-0.1), np.arange(len(args[1])), args[1])
        flux = self.model.calc(*args, **kwargs)
        #convolve with LSF and return only the original part of the flux
        return self.convolve(args[1], flux)[self.m/2 : -(self.m/2)]

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
        # cache the lsf_mat, if wave did not change compared to the last call
        # This will generally be the case in Sherpa fits
        if not np.allclose(wave, self.cache_x):
            self.make_lsf_mat(wave)
        newflux  = np.zeros_like(flux)
        cols = np.vsplit(self.lsf_mat, self.lsf_mat.shape[0])
        n = len(wave)
        m = min(n, len(cols))
        
        for i in np.arange(-(m/2), m/2+0.1, dtype = np.int):
            newflux[max(0, i): min(n, n+i)] += (cols[m/2+i].ravel() * flux)[max(0, -i):min(n, n-i)]

        return newflux
    
    def make_lsf_mat(self, wave):
        '''Interpolate the COS LSF for all wavelength
        
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
            print 'Warning: LSF convolution requires x to be binned in pixels.'
            print 'Warning: delta(x) differs from value given in COS IHB.'
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
        self.lsf_mat = np.zeros((self.m, n))
        for i in np.arange(-(self.m/2), self.m/2+0.1, dtype = np.int):
            pix = interpolate(wave, wavelen, lsftab[self.m/2+i,:])
            self.lsf_mat[self.m/2 + i,:] = pix
        
        self.cache_x = wave

# From the COS IHB
# dispersion in Ang / pixel
dispersion = {'G130M': [0.00997], 'G140L': [0.0803], 'G160M': [0.01223], 'NUV': [0.033, 0.037, 0.040, 0.39], 'G185M': [0.037], 'G225M': [0.033], 'G285M': [0.04], 'G230L': [0.39]}

'''On import this module will detect all data files in lsf/
and automatically generate SHERPA models for that.
'''
def factory(filename, modelname):

    for grating in dispersion.keys():
        # If one grating matches, generate a class for that
        if grating in filename:
            disp = dispersion[grating]
            break
    else:
        print 'Grating in ' + modelname + ' not recogized.'
            
    class NewClass(CosLsf):
        def __init__(self, p, model, name = modelname):
            '''
            Parameters
            ----------
            p : Sherpa parameters
                This model has not parameters, but accepts anything here to
                conferm to the Sherpa model API
            model : Sherpa model
                see Sherpa model API
            name : string, optional
                name of model
            '''
            self.disp = disp
            self.lsf_tab =  np.loadtxt(filename)
            CosLsf.__init__(self, model, name)

    NewClass.__name__ = modelname
    return NewClass

# List of all data files in the lsf directory
datlist = glob.glob(os.path.join(os.path.dirname(__file__), 'lsf', '*.dat'))

for filename in datlist:
    
    modelname = os.path.basename(filename).split('.')[0]
    print 'Adding '+ modelname +' to Sherpa'
    load_user_model(factory(filename, modelname), modelname)
