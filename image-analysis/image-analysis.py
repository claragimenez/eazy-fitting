
"""
Code for analysing images and perform different calculations on them.

AUTHOR:
      Clara Giménez Arteaga, University of Copenhagen, clara.arteaga_at_nbi.ku.dk
      Gabriel Brammer, University of Copenhagen
      
PURPOSE:
      Subtract continuum to obtain emission line images, calculate the Balmer
      decrement to obtain extinction maps, obtain an RGB image of the target,
      calculate the UVJ colours to plot the spatially resolved UVJ
      diagram of the target.
      
FUNCTIONS USED:
          SUBTRACT_CONTINUUM    -- Subtract the continuum in a line emission image.
          EXTINCTION_MAP        -- Calculate Balmer decrement to obtain extinction map.
          RGB_IMAGE             -- Obtain RGB image of the target.
          UVJ_DIAGRAM           -- Create UVJ diagram of the target.
"""


def subtract_continuum(line='f673n',continuum='f814w',target='galaxy',file_name='galaxy*.fits',line_name='ha',z=0.02,plot=False):
    
    """
    Function for subtracting the continuum in a line emission image

    INPUTS:
         line: Filter into which the emission line falls.
    continuum: Filter within which the emission line and broader continuum are contained.
       target: The name of the target, to be used in the output files.
    file_name: General form of the file names that contain the line and continuum images.
    line_name: State 'ha' or 'pab' to subtract Balmer-alpha or Paschen-beta respectively.
            z: Redshift of the target object.
       
    KEYWORDS:
         PLOT: Set this keyword to produce a plot of the two-dimensional
               continuum subtracted image.   
    OUTPUTS:
         Fits file with the continuum subtracted line emission image.

    """
    
    import pysynphot as S
    import numpy as np
    import glob
    from grizli import utils
    import astropy.io.fits as pyfits
    import matplotlib.pyplot as plt
    get_ipython().run_line_magic('matplotlib', 'inline')
    
    # restframe wavelength of the emission lines to subtract
    wave_pab = 1.2822e4 
    wave_ha = 6562.8
    
    print('Target =',target)
    files = glob.glob(file_name)
    files.sort()
    images = {}
    headers = {}
    bandpasses = {}

    for file in files:
        im = pyfits.open(file)
        filt = utils.get_hst_filter(im[0].header).lower()
        for ext in [0,1]:
            if 'PHOTMODE' in im[ext].header:
                photflam = im[ext].header['PHOTFLAM']
                headers[filt.lower()] = im[ext].header
                bandpasses[filt.lower()] = S.ObsBandpass(im[ext].header['PHOTMODE'].replace(' ',','))
                break
    
        flat_flam = S.FlatSpectrum(1., fluxunits='flam')
        obs = S.Observation(flat_flam, bandpasses[filt.lower()])
        my_photflam = 1/obs.countrate()
        flat_ujy = S.FlatSpectrum(1, fluxunits='ujy')
        obs = S.Observation(flat_ujy, bandpasses[filt.lower()])
        my_photfnu = 1/obs.countrate()
        images[filt.lower()] = [im['SCI'].data,im['ERR'].data]
    
    # Use PySynphot to compute flux calibration factors

    if line_name=='pab':
        # Pa-beta 
        cont_filter, line_filter, line_wave, name = cont, line, wave_pab, 'pab'
    elif line_name=='ha':
        # H-alpha
        cont_filter, line_filter, line_wave, name = cont, line, wave_ha, 'ha'

    ################
    # Continuum - flat spectrum
    cont = S.FlatSpectrum(1.e-19, fluxunits='flam')

    ###############
    # Continuum - slope spectrum
    cont_wave = np.arange(1000, 2.e4)

    slope = 0 # flat
    slope = 1 # red slope, increasing toward longer wavelengths
    cont_flux = (cont_wave/1.e4)**slope
    cont = S.ArraySpectrum(cont_wave, cont_flux, fluxunits='flam')

    ################
    # Continuum, galaxy model
    templ = utils.load_templates(full_line_list=[], line_complexes=False, alf_template=True)['alf_SSP.dat']
    cont = S.ArraySpectrum(templ.wave*(1+z), templ.flux, fluxunits='flam')

    # Gaussian line model
    ref_flux = 1.e-17
    line_model = S.GaussianSource(ref_flux, line_wave*(1+z), 10, waveunits='angstrom', fluxunits='flam')

    cont_contin_countrate = S.Observation(cont, bandpasses[cont_filter]).countrate()
    line_contin_countrate = S.Observation(cont, bandpasses[line_filter]).countrate()
    line_emline_countrate = S.Observation(line_model, bandpasses[line_filter]).countrate()

    # Continuum-subtracted, flux-calibrated
    line_calib = (images[line_filter][0] - images[cont_filter][0]*line_contin_countrate/cont_contin_countrate)
    line_calib /= line_emline_countrate
    
    # Propagated error of the subtraction
    err_sub = np.sqrt((images[line_filter][1]**2) + (images[cont_filter][1]*line_contin_countrate/cont_contin_countrate)**2)
    err_sub /= line_emline_countrate
    
    if plot:
        print("Continuum subtracted image")
        plt.figure()
        plt.imshow(line_calib,vmin=-0.5,vmax=0.5)
        plt.colorbar()
    
    primary_extn = pyfits.PrimaryHDU()
    sci_extn = pyfits.ImageHDU(data=line_calib,name='SCI')
    err_extn = pyfits.ImageHDU(data=err_sub,name='ERR') 
    hdul = pyfits.HDUList([primary_extn, sci_extn, err_extn])
    hdul.writeto('sub_{0}_{1}.fits'.format(line_name,target), output_verify='fix', overwrite=True)
    
    print(line_name,' Continuum Subtracted')

    
# -----------------------------------------------------------------------------------------------------

def extinction_map(line1='halpha.fits', line2='paschenbeta.fits', target="galaxy", min_sn=1, plot=False):
    
    """
    Function for calculating the Balmer decrement (H-alpha/Pa-beta)
    and creating an extinction map of the target.
    
    INPUTS:
         line1: File that contains one emission line image.
         line2: File that contains another emission line image.
        target: The name of the target, to be used in the output files.
        min_sn: Minimum signal-to-noise ratio for line2.
                 
    KEYWORDS:
          PLOT: Set this keyword to produce a plot of the two-dimensional
                continuum subtracted image.   
    OUTPUTS:
          Fits file with the extinction map (Balmer decrement).

    """
    
    import numpy as np
    import astropy.io.fits as pyfits
    import matplotlib.pyplot as plt
    
    im1 = pyfits.open(line1)
    im2 = pyfits.open(line2)
       
    sci1 = np.cast[np.float32](im1['SCI'].data)
    sci2 = np.cast[np.float32](im2['SCI'].data)
    error1 = np.cast[np.float32](im1['ERR'].data)
    error2 = np.cast[np.float32](im2['ERR'].data)
    
    if min_sn>0:
        sci2=np.maximum(sci2,error2*min_sn)
        
    ext = (sci1/sci2)/17.56    # 17.56 intrinsic ratio Ha/Pab
    err = np.sqrt((error1/sci2)**2+(error2*(sci1/(sci2**2)))**2)
    
    primary_extn = pyfits.PrimaryHDU()
    sci_extn = pyfits.ImageHDU(data=ext,name='SCI')
    err_extn = pyfits.ImageHDU(data=err,name='ERR') 
    hdul = pyfits.HDUList([primary_extn, sci_extn, err_extn])
    hdul.writeto('{0}_extinction_map.fits'.format(target), output_verify='fix', overwrite=True)    
 
    if plot:
        print("Extinction map Ha/Pab")
        plt.figure(figsize=(10,10))
        plt.imshow(ext,vmin=-20,vmax=20)
        plt.colorbar()

        
# -----------------------------------------------------------------------------------------------------

def rgb_image(target='galaxy',file_name='galaxy*.fits',blue_f='f435w',green_f='f814w',red_f='f110w', plot=False):
    
    """
    Function that creates an RGB image of an input target.
    
    INPUTS:
        target: The name of the target, to be used in the output files.
     file_name: General form of the file names that contain the RGB filter images.
        blue_f: Filter in the blue band.
       green_f: Filter in the green band.
         red_f: Filter in the red band.
                 
    KEYWORDS:
          PLOT: Set this keyword to produce a plot of the two-dimensional
                continuum subtracted image.   
    OUTPUTS:
          Fits file with the RGB image.

    """
    
    import numpy as np
    import matplotlib.pyplot as plt
    import astropy.io.fits as pyfits
    import glob
    from grizli import utils
    from astropy.visualization import lupton_rgb
    
    print('Target = ',target)
    files = glob.glob(file_name)
    files.sort()
    images = {}
    headers = {}
    bandpasses = {}

    for file in files:
        im = pyfits.open(file)
        filt = utils.get_hst_filter(im[0].header)
        for ext in [0,1]:
            if 'PHOTMODE' in im[ext].header:
                photflam = im[1].header['IM2FLAM']
                headers[filt.lower()] = im[1].header
                break
        images[filt.lower()] = im['SCI'].data
    
    blue = images[blue_f]*headers[blue_f]['IM2FLAM']/1.e-19
    green = images[green_f]*headers[green_f]['IM2FLAM']/1.e-19
    red = images[red_f]*headers[red_f]['IM2FLAM']/1.e-19

    rgb = lupton_rgb.make_lupton_rgb(red, green, blue, minimum=-0.1, stretch=1, Q=8)
    pyfits.writeto('rgb_{0}.fits'.format(target),data=rgb,overwrite=True) 
    print('RGB image created')
    
    if plot:
        fig = plt.figure(figsize=[6,6])
        ax = fig.add_subplot(111)
        imsh = ax.imshow(rgb, origin='lower')

    
# -----------------------------------------------------------------------------------------------------
 
def uvj_diagram(u='galaxy_u.fits',v='galaxy_v.fits',j='galaxy_j.fits',target='galaxy',file_name='galaxy*.fits',size=200,x0=1000,y0=1000):
    
    """
    Function for plotting a spatially resolved UVJ Diagram of an input target, 
    while comparing with EAZY templates.
    
    INPUTS:
            u: Fits file of the image in the U band.
            v: Fits file of the image in the V band.
            j: Fits file of the image in the J band.
       target: The name of the target, to be used in the output files.
    file_name: General form of the file names that contain the UVJ images.
         size: Size of the slice to plot only part of the original image.
           x0: Center of the x-coordinate of the slice.
           y0: Center of the y-coordinate of the slice.
        
    KEYWORDS:
          PLOT: Set this keyword to produce a plot of the two-dimensional
                continuum subtracted image.   
    OUTPUTS:
          Plot of the UVJ diagram.
    """
    
    from grizli import utils
    import numpy as np
    import astropy.io.fits as pyfits
    import matplotlib.pyplot as plt
    import glob
    import pysynphot as S
    
    slx = slice(x0-size, x0+size)
    sly = slice(y0-size, y0+size)
    
    u_im = pyfits.open(u)
    v_im = pyfits.open(v)
    j_im = pyfits.open(j)
    
    for ext in [0,1]:
        if 'PHOTPLAM' in u_im[ext].header:
            u_lam = u_im[ext].header['PHOTPLAM']
            break
    for ext in [0,1]:
        if 'PHOTPLAM' in v_im[ext].header:
            v_lam = v_im[ext].header['PHOTPLAM']
            break
    for ext in [0,1]:
        if 'PHOTPLAM' in j_im[ext].header:
            j_lam = j_im[ext].header['PHOTPLAM']
            break
    
    fu = u_im['SCI'].data[sly, slx]*u_im[1].header['IM2FLAM']*(u_lam**2)
    fv = v_im['SCI'].data[sly, slx]*v_im[1].header['IM2FLAM']*(v_lam**2)
    fj = j_im['SCI'].data[sly, slx]*j_im[1].header['IM2FLAM']*(j_lam**2)
    
    fu_error = u_im['ERR'].data[sly, slx]*u_im[1].header['IM2FLAM']*(u_lam**2)
    fv_error = v_im['ERR'].data[sly, slx]*v_im[1].header['IM2FLAM']*(v_lam**2)
    fj_error = j_im['ERR'].data[sly, slx]*j_im[1].header['IM2FLAM']*(j_lam**2)
    
    u_v_err = (2.5/np.log(10))*np.sqrt((fu_error/fu)**2+(fv_error/fv)**2)
    v_j_err = (2.5/np.log(10))*np.sqrt((fj_error/fj)**2+(fv_error/fv)**2)
    
    u_v = -2.5*np.log10(fu/fv)
    v_j = -2.5*np.log10(fv/fj)
    
    mask = (u_v_err<0.1) & (v_j_err<0.1)

    templ = utils.load_templates(full_line_list=[], line_complexes=False, fsps_templates=True, alf_template=True)

    files = glob.glob(file_name)
    files.sort()
    images = {}
    bandpasses = {}

    for file in files:
        im = pyfits.open(file)
        filt = utils.get_hst_filter(im[0].header)
        for ext in [0,1]:
            if 'PHOTMODE' in im[ext].header:
                bandpasses[filt.lower()] = S.ObsBandpass(im[ext].header['PHOTMODE'].replace(' ',','))
                break
        images[filt.lower()] = im['SCI'].data

    template_fluxes = {}
    for f in bandpasses:
        template_fluxes[f] = np.zeros(len(templ))

    # templates from EAZY
    for i, t in enumerate(templ):
        t_z = templ[t].zscale(0.03)
        for f in bandpasses:
            template_fluxes[f][i] = t_z.integrate_filter(bandpasses[f])    

    s = templ.keys()
    uv = -2.5*np.log10(template_fluxes[u]/template_fluxes[v])
    vj = -2.5*np.log10(template_fluxes[v]/template_fluxes[j])
   
    fig, ax = plt.subplots(figsize=(12,10))

    plt.plot(v_j[mask],u_v[mask],'kx',ms=1,zorder=-5)
    
    for i,lab in enumerate(s):
        plt.scatter(vj[i],uv[i],label=lab)
             
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.xlabel('V-J [observed] (AB mag)'.format(v,j))
    plt.ylabel('U-V [observed] (AB mag)'.format(u,v))
    plt.title('UVJ Diagram {0}'.format(target))
    plt.show()

#-----------------------------------------------------------------------------