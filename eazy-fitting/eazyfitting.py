"""
Code for fitting images with EAZY and obtain physical parameters (rest-frame
colours, masses, SFR...)

AUTHOR:
      Clara Giménez Arteaga, University of Copenhagen, clara.arteaga_at_nbi.ku.dk
      Gabriel Brammer, University of Copenhagen
      
PURPOSE:
      Obtain physical parameters of target objects with SED fitting by EAZY.
      For more information on EAZY go to https://github.com/gbrammer/eazy-py
      
FUNCTIONS USED:
      EAZY_FITTING  -- Fit an input catalog and output physical parameters.
         
"""

def eazy_fitting(catalog_file='catalog.fits',target='galaxy',im, seg_im, mw_ebv=0.0375, plot=False, image_space=False):
    
    """
    Function fitting the input target and obtain physical parameters.

    INPUTS:
     catalog_file: Catalog file with the fluxes and errors to be fitted.
           target: The name of the target, to be used in the output files.
               im: Image to be used as reference to convert output from
                   table space into image space. [optional]
           seg_im: Segmentation image. [optional]
           mw_ebv: Galactic extinction. Value can be found here:
                   https://irsa.ipac.caltech.edu/applications/DUST/
       
    KEYWORDS:
         PLOT: Set this keyword to produce a plot of the two-dimensional
               continuum subtracted image.   
  IMAGE_SPACE: Set this keyword to produce the output in image space.
         
    OUTPUTS:
         EAZY derived physical parameters.

    """

    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import eazy
    import warnings
    from grizli import prep
    from astropy.utils.exceptions import AstropyWarning
    import astropy.io.fits as pyfits
    from astropy.table import Table
    from astropy.cosmology import WMAP9
    from astropy import units as u

    np.seterr(all='ignore')
    warnings.simplefilter('ignore', category=AstropyWarning)

    image = pyfits.open(im)
    sci = np.cast[np.float32](image['SCI'].data)
    seg = pyfits.open(seg_im)[0].data
    tab = Table.read(catalog_file)
    
    # Parameters 
    params = {}
    params['CATALOG_FILE'] = os.path.join(os.getenv('EAZYCODE'), catalog_file)
    params['MAIN_OUTPUT_FILE'] = 'hdfn.eazypy'

    # Galactic extinction
    params['MW_EBV'] = mw_ebv 
    params['SYS_ERR'] = 0.05

    params['Z_STEP'] = 0.001
    params['Z_MIN'] = 0.005
    params['Z_MAX'] = 0.1

    params['PRIOR_ABZP'] = 23.9 
    params['PRIOR_FILTER'] = 241 # K
    params['PRIOR_FILE'] = 'templates/prior_K_TAO.dat'
    params['TEMPLATES_FILE'] = 'templates/fsps_full/tweak_fsps_QSF_12_v3.param'
    params['FIX_ZSPEC'] = True

    translate_file = os.path.join(os.getenv('EAZYCODE'), 'inputs/zphot.translate')
    self = eazy.photoz.PhotoZ(param_file=None, translate_file=translate_file, zeropoint_file=None, 
                          params=params, load_prior=True, load_products=False)

    # Now fit the whole catalog
    # Turn off error corrections derived above
    self.efnu = self.efnu_orig*1

    # Full catalog
    sample = np.isfinite(self.cat['z_spec'])
    t = self.cat
   # sel = (t['xmin']>(x0-size))&(t['ymin']>(y0-size))&(t['xmax']<(x0+size))&(t['ymax']<(y0+size))
   # sample = sel
    self.fit_parallel(self.idx[sample], n_proc=8)

    # Derived parameters (z params, RF colors, masses, SFR, etc.)
    zout, hdu = self.standard_output(rf_pad_width=0.5, rf_max_err=2, 
                                     prior=False, beta_prior=False, extra_rf_filters=[272,273,274])

    # 'zout' also saved to [MAIN_OUTPUT_FILE].zout.fits

    if plot:
        # Show UVJ diagram
        uv = -2.5*np.log10(zout['restU']/zout['restV'])
        vj = -2.5*np.log10(zout['restV']/zout['restJ'])
        ssfr = zout['SFR']/zout['mass']

        plt.scatter(vj[sel], uv[sel], c=np.log10(ssfr)[sel], cmap = 'Spectral',
                    vmin=-13, vmax=-8, alpha=0.5)
        plt.colorbar()
        plt.xlabel(r'$(V-J)_0$'); plt.ylabel(r'$(U-V)_0$') 
        
        t['x'] = (t['xmin']+t['xmax'])/2
        t['y'] = (t['ymin']+t['ymax'])/2
        
        # sSFR
        fig = plt.figure(figsize=(10,8))
        plt.scatter(t['x'],t['y'],marker='.',alpha=0.5,c=np.log10(ssfr),cmap='Spectral',vmin=-12,vmax=-8)
        ax = plt.gca()
        ax.set_aspect(1)
        plt.colorbar()
        # Av
        fig = plt.figure(figsize=(10,8))
        plt.scatter(t['x'],t['y'],marker='.',alpha=0.5,c=zout['Av'],cmap='Spectral_r',vmin=0,vmax=3)
        ax = plt.gca()
        ax.set_aspect(1)
        plt.colorbar()
        
        plt.show()

    if image_space:
        # Av
        flux_Av = prep.get_seg_iso_flux(sci, seg, tab, fill=zout['Av'])
        fig = plt.figure(figsize=(10,8))
        plt.imshow(flux_dust[sly, slx],cmap='Spectral_r')
        ax = plt.gca()
        ax.set_aspect(1)
        plt.colorbar()

        primary_extn = pyfits.PrimaryHDU()
        sci_extn = pyfits.ImageHDU(data=flux_dust,name='SCI')
        hdul = pyfits.HDUList([primary_extn, sci_extn])
        hdul.writeto('Av_{0}.fits'.format(target), overwrite=True)

        # ssfr
        flux_ssfr = prep.get_seg_iso_flux(sci, seg, tab, fill=np.log10(ssfr))
        fig = plt.figure(figsize=(10,8))
        plt.imshow(flux_ssfr[sly, slx],cmap='Spectral')
        ax = plt.gca()
        ax.set_aspect(1)
        plt.colorbar()

        primary_extn = pyfits.PrimaryHDU()
        sci_extn = pyfits.ImageHDU(data=flux_ssfr,name='SCI')
        hdul = pyfits.HDUList([primary_extn, sci_extn])
        hdul.writeto('sSFR_{0}.fits'.format(target), overwrite=True)
        
        plt.show()

    return zout
