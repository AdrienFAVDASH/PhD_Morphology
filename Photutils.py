#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np
from astropy.io import fits
import astropy.wcs as wcs
from astropy.stats import gaussian_fwhm_to_sigma
from astropy.convolution import convolve, Gaussian2DKernel, Tophat2DKernel
from photutils.segmentation import detect_sources, deblend_sources, detect_threshold, SourceCatalog
import statmorph
from scipy import ndimage
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import h5py
import warnings

#------------------------------------------------------------------------------
# QOL
#------------------------------------------------------------------------------

warnings.simplefilter("ignore")

#------------------------------------------------------------------------------
# Main procedures
#------------------------------------------------------------------------------

def process_image_morphology(image, morph_properties, nsigma, npixels, nlevels, contrast, labels=None, connectivity=8, mode='exponential', smooth_data=True, kernel_name='Tophat', smooth_fwhm=3, kernel_size=5, relabel=True, nproc=1, progress_bar=True, show_plots=True, psf=None, cutout_extent=2.5, min_cutout_size=48, n_sigma_outlier=10, annulus_width=1.0, eta=0.2, petro_fraction_gini=0.2, skybox_size=32, petro_extent_cas=1.5, petro_fraction_cas=0.25, boxcar_size_mid=3.0, niter_bh_mid=5, sigma_mid=1.0, petro_extent_flux=2.0, boxcar_size_shape_asym=3.0, sersic_fitting_args=None, sersic_model_args=None, sersic_maxiter=None, include_doublesersic=False, doublesersic_rsep_over_rhalf=2.0, doublesersic_tied_ellip=False, doublesersic_fitting_args=None, doublesersic_model_args=None, segmap_overlap_ratio=0.25, verbose=False):
    """
    Returns a dictionary containing the desired morphological properties of all the galaxies identified on the given image. The source_morphology object is also saved in self.morphologies.
    
    Parameters :
    - image : image object obtained using the image initializing classes
    - morph_properties : list of desired morphological properties (available properties : asymmetry, concentration, deviation, doublesersic_aic, doublesersic_amplitude1, doublesersic_amplitude2, doublesersic_bic, doublesersic_chi2_dof, doublesersic_ellip1, doublesersic_ellip2, doublesersic_n1, doublesersic_n2, doublesersic_rhalf1, doublesersic_rhalf2, doublesersic_theta1, doublesersic_theta2, doublesersic_xc, doublesersic_yc, ellipticity_asymmetry, ellipticity_centroid, flag, flag_sersic, flux_circ, flux_ellip, gini, gini_m20_bulge, gini_m20_merger, intensity, m20, multimode, nx_stamp, ny_stamp, orientation_asymmetry, orientation_centroid, outer_asymmetry, r20, r50, r80, rhalf_circ, rhalf_ellip, rmax_circ, rmax_ellip, rms_asymmetry2, rpetro_circ, rpetro_ellip, sersic_aic, sersic_amplitude, sersic_bic, sersic_chi2_dof, sersic_ellip, sersic_n, sersic_rhalf, sersic_theta, sersic_xc, sersic_yc, shape_asymmetry, sky_mean, sky_median, sky_sigma, smoothness, sn_per_pixel, xc_asymmetry, xc_centroid, xmax_stamp, xmin_stamp, yc_asymmetry, yc_centroid, ymax_stamp, ymin_stamp.)
    - nsigma : threshold value in units of standard deviation for detecting sources. Sources are pixels whose values are above nsigma times the local background standard deviation
    - npixels : minimum number of connected pixels above the threshold that an object must have to be deblended
    - nlevels : number of multi-thresholding levels to use for deblending. Each source will be re-thresholded at nlevels levels spaced between its minimum and maximum values (non-inclusive)
    - contrast : fraction of total source flux that a local peak must have (at any one of the multi-thresholds) to be deblended as a separate object
    - labels : label numbers to deblend. If None, then all labels in the segmentation image will be deblended
    - connectivity : type of pixel connectivity used in determining how pixels are grouped into a detected source. Available options are 8 and 4 where 8-connected pixels touch along their edges or corners while 4-connected pixels touch along their edges
    - mode : mode used in defining the spacing between the multi-thresholding levels during deblending. Available options are 'exponential', 'linear' and 'sinh'
    - smooth_data : bolean, whether to convolve the data with a smoothing kernel before source detection
    - kernel_name : name of kernel used to convolve data. Only Tophat and Gaussian are supported at the moment
    - smooth_fwhm : width of kernel
    - kernel_size : size of kernel
    - relabel : bolean, whether to relabel the segmentation image such that the labels are in consecutive order starting from 1.
    - nproc : number of processes to use for multiprocessing (if larger than 1). If set to 1, then a serial implementation is used instead of a parallel one. If None, then the number of processes will be set to the number of CPUs detected on the machine. Due to overheads, multiprocessing may be slower than serial processing. This is especially true if one only has a small number of sources to deblend. The benefits of multiprocessing require ~1000 or more sources to deblend, with larger gains as the number of sources increase.
    - progress_bar : bolean, whether to display a progress bar while the deblending is taking place
    - show_plots : bolean, whether to display plots of both the segmented and deblended images
    - psf : 2D array representing the PSF, where the central pixel corresponds to the center of the PSF. Typically, including this keyword argument will make the code run slower by a factor of a few, depending on the size of the PSF, but the resulting Sersic fits will be more correct
    - cutout_extent : target fractional size of the data cutout relative to the minimal bounding box containing the source. The value must be >= 1
    - min_cutout_size : minimum size of the cutout, in case cutout_extent times the size of the minimal bounding box is smaller than this. Any given value will be truncated to an even number
    - n_sigma_outlier : number of standard deviations that define a pixel as an outlier, relative to its 8 neighbors. Outlying pixels are removed as described in Lotz et al. (2004). If the value is zero or negative, outliers are not removed
    - annulus_width : The width of the annuli used to calculate the Petrosian radius and other quantities
    - eta : Petrosian eta parameter used to define the Petrosian radius. For a circular or elliptical aperture at the Petrosian radius, the mean flux at the edge of the aperture divided by the mean flux within the aperture is equal to eta
    - petro_fraction_gini : fraction of the Petrosian radius used as a smoothing scale in order to define the pixels that belong to the galaxy in the Gini calculation
    - skybox_size : target size of the skybox used to characterize the sky background
    - petro_extent_cas : radius of the circular aperture used for the asymmetry calculation, in units of the circular Petrosian radius
    - petro_fraction_cas : fraction of the Petrosian radius used as a smoothing scale in the CAS calculations
    - boxcar_size_mid : size of the constant kernel used to regularize the MID segmap in the MID calculations
    - niter_bh_mid : number of iterations in the basin-hopping stage of the maximization when calculating the multimode statistic. A value of at least 100 is recommended for “production” runs, but it can be time-consuming
    - sigma_mid : smoothing scale (in pixels) used to compute the intensity (I) statistic in the MID calculations
    - petro_extent_flux : number of Petrosian radii used to define the aperture over which the flux is measured. This is also used to define the inner radius of the elliptical aperture used to estimate the sky background in the shape asymmetry calculation
    - boxcar_size_shape_asym : size of the constant kernel used to regularize the segmap when calculating the shape asymmetry segmap
    - sersic_fitting_args : dictionary of keyword arguments passed to Astropy’s LevMarLSQFitter, which cannot include z or weights (these are handled internally by statmorph)
    - sersic_model_args : dictionary of keyword arguments passed to Astropy’s Sersic2D. By default, statmorph will make reasonable initial guesses for all the model parameters, although this functionality can be overridden for more customized fits. By default, statmorph also imposes a lower bound of n = 0.01 for the fitted Sersic index
    - sersic_maxiter : Deprecated. Use sersic_fitting_args instead
    - include_doublesersic : bolean, whether to also fit a double 2D Sersic model
    - doublesersic_rsep_over_rhalf : specifies the boundary used to separate the inner and outer regions of the image, which are used to perform single Sersic fits and thus construct an initial guess for the double Sersic fit. This argument is provided as a multiple of the (non-parametric) half-light semimajor axis rhalf_ellip
    - doublesersic_tied_ellip : bolean, if True both components of the double Sersic model share the same ellipticity and position angle. The same effect could be achieved using doublesersic_model_args in combination with the 'tied' argument, although the syntax would be slightly more involved
    - doublesersic_fitting_args : same as sersic_fitting_args, but for the double 2D Sersic fit
    - doublesersic_model_args : same as sersic_model_args, but for the double 2D Sersic fit
    - segmap_overlap_ratio : minimum ratio (in order to have a 'good' measurement) between the area of the intersection of the Gini and MID segmaps and the area of the largest of these two segmaps
    - verbose : bolean, If True this prints various minor warnings (which do not result in 'bad' measurements) during the calculations
    
    Documentation :
    - https://photutils.readthedocs.io/en/stable/api/photutils.segmentation.detect_sources.html#detect-sources
    - https://photutils.readthedocs.io/en/stable/api/photutils.segmentation.detect_threshold.html#detect-threshold
    - https://photutils.readthedocs.io/en/stable/api/photutils.segmentation.deblend_sources.html#deblend-sources
    - https://statmorph.readthedocs.io/en/latest/api.html#statmorph.source_morphology
    """
    
    segment_img = image.source_detection(image.sci, nsigma, npixels, connectivity, image.mask, 0.0, image.bkg_err, smooth_data, kernel_name, smooth_fwhm, kernel_size)
    deblend_img = image.source_deblending(image.sci, segment_img, npixels, labels, nlevels, contrast, mode, connectivity, relabel, nproc, progress_bar)
    
    if show_plots is True:
        fig1, ax1 = plt.subplots(figsize=(8, 8))
        ax1.imshow(segment_img, origin='lower', cmap=deblend_img.cmap, interpolation='nearest')
        ax1.set_title('Segmentation Image')
        fig2, ax2 = plt.subplots(figsize=(8, 8))
        ax2.imshow(deblend_img, origin='lower', cmap=deblend_img.cmap, interpolation='nearest')
        ax2.set_title('Deblended Segmentation Image')
        plt.show()
    
    morphologies = image.morphology(image.sci, deblend_img, image.mask, image.wht, image.gain, psf, cutout_extent, min_cutout_size, n_sigma_outlier, annulus_width, eta, petro_fraction_gini, skybox_size, petro_extent_cas, petro_fraction_cas, boxcar_size_mid, niter_bh_mid, sigma_mid, petro_extent_flux, boxcar_size_shape_asym, sersic_fitting_args, sersic_model_args, sersic_maxiter, include_doublesersic, doublesersic_rsep_over_rhalf, doublesersic_tied_ellip, doublesersic_fitting_args, doublesersic_model_args, segmap_overlap_ratio, verbose)

    results = {}
    for i in morph_properties:
        try:
            results[i] = [getattr(morphologies[j], i) for j in range(len(morphologies))]
        except AttributeError:
            results[i] = None
            print(f"Warning: Morphological property '{i}' not found.")

    return results
    
def process_image_photometry(image, phot_properties, nsigma, npixels, nlevels, contrast, kron_params, labels=None, connectivity=8, mode='exponential', local_bkg_width=0, apermask_method='correct', smooth_data=True, kernel_name='Tophat', smooth_fwhm=3, kernel_size=5, detection_cat=None, relabel=True, wcs='None', nproc=1, progress_bar=True, show_plots=True):
    """
    Returns a dictionary containing the desired photometric properties of all the galaxies identified on the given image. The SourceCatalog object is also saved in self.photometry_cat.
    
    Parameters :
    - image : image object obtained using the image initializing classes
    - phot_properties : list of desired photometric properties (available properties : area, background, background_centroid, background_ma, background_mean, background_sum, bbox, bbox_xmax, bbox_xmin, bbox_ymax, bbox_ymin, centroid, centroid_quad, centroid_win, convdata, convdata_ma, covar_sigx2, covar_sigy2, covariance, covariance_eigvals, cutout_centroid, cutout_centroid_quad, cutout_centroid_win, cutout_maxval_index, cutout_minval_index, cxx, cxy, cyy, data, data_ma, eccentricity, ellipticity, elongation, equivalent_radius, error, error_ma, extra_properties, fwhm, gini, inertia_tensor, isscalar, kron_aperture, kron_flux, kron_fluxerr, kron_radius, label, labels, local_background, local_background_aperture, max_value, maxval_index, maxval_xindex, maxval_yindex, min_value, minval_index, minval_xindex, minval_yindex, moments, moments_central, nlabels, orientation, perimeter, properties, segment, segment_area, segment_flux, segment_fluxerr, segment_ma, semimajor_sigma, semiminor_sigma, skybbox_ll, skybox_lr, skybox_ul, skybox_ur, sky_centroid, sky_centroid_icrs, sky_centroid_quad, sky_centroid_win, slices, xcentroid, xcentroid_quad, xcentroid_win, ycentroid, ycentroid_quad, ycentroid_win)
    - nsigma : threshold value in units of standard deviation for detecting sources. Sources are pixels whose values are above nsigma times the local background standard deviation
    - npixels : minimum number of connected pixels above the threshold that an object must have to be deblended
    - nlevels : number of multi-thresholding levels to use for deblending. Each source will be re-thresholded at nlevels levels spaced between its minimum and maximum values (non-inclusive)
    - contrast : fraction of total source flux that a local peak must have (at any one of the multi-thresholds) to be deblended as a separate object
    - kron_params : list of parameters used to determine the Kron aperture. The first item is the scaling parameter of the unscaled Kron radius and the second item represents the minimum value for the unscaled Kron radius in pixels. The optional third item is the minimum circular radius in pixels. If kron_params[0] * kron_radius * sqrt(semimajor_sigma * semiminor_sigma) is less than or equal to this radius, then the Kron aperture will be a circle with this minimum radius
    - labels : label numbers to deblend. If None, then all labels in the segmentation image will be deblended
    - connectivity : type of pixel connectivity used in determining how pixels are grouped into a detected source. Available options are 8 and 4 where 8-connected pixels touch along their edges or corners while 4-connected pixels touch along their edges
    - mode : mode used in defining the spacing between the multi-thresholding levels during deblending. Available options are 'exponential', 'linear' and 'sinh'
    - local_bkg_width : width of the rectangular annulus used to compute a local background around each source. If zero, then no local background subtraction is performed. The local background affects the min_value, max_value, segment_flux, kron_flux, and fluxfrac_radius properties. It is also used when calculating circular and Kron aperture photometry. It does not affect the moment-based morphological properties of the source.
    - apermask_method : method used to handle neighboring sources when performing aperture photometry. This parameter also affects the Kron radius. 'correct' replaces pixels assigned to neighboring sources by replacing them with pixels on the opposite side of the source center (equivalent to MASK_TYPE=CORRECT in SourceExtractor).'mask' masks pixels assigned to neighboring sources (equivalent to MASK_TYPE=BLANK in SourceExtractor). 'none' does not mask any pixels (equivalent to MASK_TYPE=NONE in SourceExtractor)
    - smooth_data : bolean, whether to convolve the data with a smoothing kernel before source detection
    - kernel_name : name of kernel used to convolve data. Only Tophat and Gaussian are supported at the moment
    - smooth_fwhm : width of kernel
    - kernel_size : size of kernel
    - detection_cat : SourceCatalog object for the detection image. The segmentation image used to create the detection catalog must be the same one input to segment_img. If input, then the detection catalog source centroids and morphological/shape properties will be returned instead of calculating them from the input data. The detection catalog centroids and shape properties will also be used to perform aperture photometry. If detection_cat is input, then the input wcs, apermask_method, and kron_params keywords will be ignored. This keyword affects circular_photometry (including returned apertures), all Kron parameters (Kron radius, flux, flux errors, apertures, and custom kron_photometry), and fluxfrac_radius (which is based on the Kron flux).
    - relabel : bolean, whether to relabel the segmentation image such that the labels are in consecutive order starting from 1.
    - wcs : A world coordinate system (WCS) transformation that supports the astropy shared interface for WCS (ex : astropy.wcs.WCS, gwcs.wcs.WCS). If None, then all sky-based properties will be set to None
    - nproc : number of processes to use for multiprocessing (if larger than 1). If set to 1, then a serial implementation is used instead of a parallel one. If None, then the number of processes will be set to the number of CPUs detected on the machine. Due to overheads, multiprocessing may be slower than serial processing. This is especially true if one only has a small number of sources to deblend. The benefits of multiprocessing require ~1000 or more sources to deblend, with larger gains as the number of sources increase.
    - progress_bar : bolean, whether to display a progress bar while the deblending is taking place
    - show_plots : bolean, whether to display plots of both the segmented and deblended images
       
    Documentation :
    - https://photutils.readthedocs.io/en/stable/api/photutils.segmentation.detect_sources.html#detect-sources
    - https://photutils.readthedocs.io/en/stable/api/photutils.segmentation.detect_threshold.html#detect-threshold
    - https://photutils.readthedocs.io/en/stable/api/photutils.segmentation.deblend_sources.html#deblend-sources
    - https://photutils.readthedocs.io/en/stable/api/photutils.segmentation.SourceCatalog.html#sourcecatalog
    """    
    
    segment_img = image.source_detection(image.sci, nsigma, npixels, connectivity, image.mask, 0.0, image.bkg_err, smooth_data, kernel_name, smooth_fwhm, kernel_size)
    deblend_img = image.source_deblending(image.sci, segment_img, npixels, labels, nlevels, contrast, mode, connectivity, relabel, nproc, progress_bar)

    if show_plots is True:
        fig1, ax1 = plt.subplots(figsize=(8, 8))
        ax1.imshow(segment_img, origin='lower', cmap=deblend_img.cmap, interpolation='nearest')
        ax1.set_title('Segmentation Image')
        fig2, ax2 = plt.subplots(figsize=(8, 8))
        ax2.imshow(deblend_img, origin='lower', cmap=deblend_img.cmap, interpolation='nearest')
        ax2.set_title('Deblended Segmentation Image')
        plt.show()
    
    if smooth_data==True:
        convolved_data = image.convolved_data
    elif smooth_data==False:
        convolved_data=None
    else:
        raise ValueError('invalid value given to "smooth_data"')

    photometry_cat = image.photometry(image.sci, deblend_img, convolved_data, image.err, image.mask, image.bkg, wcs, local_bkg_width, apermask_method, kron_params, detection_cat, progress_bar)
  
    results = {}
    for i in phot_properties:
        try:
            results[i] = getattr(photometry_cat, i)
        except AttributeError:
            results[i] = None
            print(f"Warning: Photometric property '{i}' not found.")

    return results

def process_image_number_counts(image, nsigma, npixels, nlevels, contrast, labels=None, connectivity=8, mode='exponential', smooth_data=True, kernel_name='Tophat', smooth_fwhm=3, kernel_size=5, relabel=True, nproc=1, progress_bar=True, show_plots=True):

    """
    Returns the number of galaxies identified on the given image.
    
    Parameters :
    - image : image object obtained using the image initializing classes
    - nsigma : threshold value in units of standard deviation for detecting sources. Sources are pixels whose values are above nsigma times the local background standard deviation
    - npixels : minimum number of connected pixels above the threshold that an object must have to be deblended
    - nlevels : number of multi-thresholding levels to use for deblending. Each source will be re-thresholded at nlevels levels spaced between its minimum and maximum values (non-inclusive)
    - contrast : fraction of total source flux that a local peak must have (at any one of the multi-thresholds) to be deblended as a separate object
    - labels : label numbers to deblend. If None, then all labels in the segmentation image will be deblended
    - connectivity : type of pixel connectivity used in determining how pixels are grouped into a detected source. Available options are 8 and 4 where 8-connected pixels touch along their edges or corners while 4-connected pixels touch along their edges
    - mode : mode used in defining the spacing between the multi-thresholding levels during deblending. Available options are 'exponential', 'linear' and 'sinh'
    - smooth_data : bolean, whether to convolve the data with a smoothing kernel before source detection
    - kernel_name : name of kernel used to convolve data. Only Tophat and Gaussian are supported at the moment
    - smooth_fwhm : width of kernel
    - kernel_size : size of kernel
    - relabel : bolean, whether to relabel the segmentation image such that the labels are in consecutive order starting from 1.
    - nproc : number of processes to use for multiprocessing (if larger than 1). If set to 1, then a serial implementation is used instead of a parallel one. If None, then the number of processes will be set to the number of CPUs detected on the machine. Due to overheads, multiprocessing may be slower than serial processing. This is especially true if one only has a small number of sources to deblend. The benefits of multiprocessing require ~1000 or more sources to deblend, with larger gains as the number of sources increase.
    - progress_bar : bolean, whether to display a progress bar while the deblending is taking place
    - show_plots : bolean, whether to display plots of both the segmented and deblended images    

    Documentation :
    - https://photutils.readthedocs.io/en/stable/api/photutils.segmentation.detect_sources.html#detect-sources
    - https://photutils.readthedocs.io/en/stable/api/photutils.segmentation.detect_threshold.html#detect-threshold
    - https://photutils.readthedocs.io/en/stable/api/photutils.segmentation.deblend_sources.html#deblend-sources
    """

    segment_img = image.source_detection(image.sci, nsigma, npixels, connectivity, image.mask, 0.0, image.bkg_err, smooth_data, kernel_name, smooth_fwhm, kernel_size)
    deblend_img = image.source_deblending(image.sci, segment_img, npixels, labels, nlevels, contrast, mode, connectivity, relabel, nproc, progress_bar)

    if show_plots is True:
        fig1, ax1 = plt.subplots(figsize=(8, 8))
        ax1.imshow(segment_img, origin='lower', cmap=deblend_img.cmap, interpolation='nearest')
        ax1.set_title('Segmentation Image')
        fig2, ax2 = plt.subplots(figsize=(8, 8))
        ax2.imshow(deblend_img, origin='lower', cmap=deblend_img.cmap, interpolation='nearest')
        ax2.set_title('Deblended Segmentation Image')
        plt.show()
    
    return len(deblend_img.labels)

def process_images_morphology(detection_image, filter_images, morph_properties, nsigma, npixels, nlevels, contrast, output_hdf5_filename, labels=None, connectivity=8, mode='exponential', smooth_data=True, kernel_name='Tophat', smooth_fwhm=3, kernel_size=5, relabel=True, nproc=1, progress_bar=True, show_plots=True, psf=None, cutout_extent=2.5, min_cutout_size=48, n_sigma_outlier=10, annulus_width=1.0, eta=0.2, petro_fraction_gini=0.2, skybox_size=32, petro_extent_cas=1.5, petro_fraction_cas=0.25, boxcar_size_mid=3.0, niter_bh_mid=5, sigma_mid=1.0, petro_extent_flux=2.0, boxcar_size_shape_asym=3.0, sersic_fitting_args=None, sersic_model_args=None, sersic_maxiter=None, include_doublesersic=False, doublesersic_rsep_over_rhalf=2.0, doublesersic_tied_ellip=False, doublesersic_fitting_args=None, doublesersic_model_args=None, segmap_overlap_ratio=0.25, verbose=False):

    """
    Returns a hdf5 file containing the desired morphological properties of all the galaxies identified for each given filter image. The file also contains the deblended segmentation map used to obtain these properties, derived from the given detection image. The source_morphology object of each filter image is saved in self.morphologies.
    The returned hdf5 file has the following format :
    Morph/Detection_Image/detection_image name/segmentation_map
    Morph/Filter_Images/filter_image name/morph_property
    
    Parameters :
    - detection_image : image object obtained using the image initializing classes of the detection image associated to the filter images
    - filter_images : list of image objects obtained using the image initializing classes of the filter images associated to the detection image
    - morph_properties : list of desired morphological properties (available properties : asymmetry, concentration, deviation, doublesersic_aic, doublesersic_amplitude1, doublesersic_amplitude2, doublesersic_bic, doublesersic_chi2_dof, doublesersic_ellip1, doublesersic_ellip2, doublesersic_n1, doublesersic_n2, doublesersic_rhalf1, doublesersic_rhalf2, doublesersic_theta1, doublesersic_theta2, doublesersic_xc, doublesersic_yc, ellipticity_asymmetry, ellipticity_centroid, flag, flag_sersic, flux_circ, flux_ellip, gini, gini_m20_bulge, gini_m20_merger, intensity, m20, multimode, nx_stamp, ny_stamp, orientation_asymmetry, orientation_centroid, outer_asymmetry, r20, r50, r80, rhalf_circ, rhalf_ellip, rmax_circ, rmax_ellip, rms_asymmetry2, rpetro_circ, rpetro_ellip, sersic_aic, sersic_amplitude, sersic_bic, sersic_chi2_dof, sersic_ellip, sersic_n, sersic_rhalf, sersic_theta, sersic_xc, sersic_yc, shape_asymmetry, sky_mean, sky_median, sky_sigma, smoothness, sn_per_pixel, xc_asymmetry, xc_centroid, xmax_stamp, xmin_stamp, yc_asymmetry, yc_centroid, ymax_stamp, ymin_stamp. Default : flag, concentration, asymmetry, smoothness)
    - nsigma : threshold value in units of standard deviation for detecting sources. Sources are pixels whose values are above nsigma times the local background standard deviation
    - npixels : minimum number of connected pixels above the threshold that an object must have to be deblended
    - nlevels : number of multi-thresholding levels to use for deblending. Each source will be re-thresholded at nlevels levels spaced between its minimum and maximum values (non-inclusive)
    - contrast : fraction of total source flux that a local peak must have (at any one of the multi-thresholds) to be deblended as a separate object
    - output_hdf5_filename : name of the output hdf5 file returned by the function
    - labels : label numbers to deblend. If None, then all labels in the segmentation image will be deblended
    - connectivity : type of pixel connectivity used in determining how pixels are grouped into a detected source. Available options are 8 and 4 where 8-connected pixels touch along their edges or corners while 4-connected pixels touch along their edges
    - mode : mode used in defining the spacing between the multi-thresholding levels during deblending. Available options are 'exponential', 'linear' and 'sinh'
    - smooth_data : bolean, whether to convolve the data with a smoothing kernel before source detection
    - kernel_name : name of kernel used to convolve data. Only Tophat and Gaussian are supported at the moment
    - smooth_fwhm : width of kernel
    - kernel_size : size of kernel
    - relabel : bolean, whether to relabel the segmentation image such that the labels are in consecutive order starting from 1.
    - nproc : number of processes to use for multiprocessing (if larger than 1). If set to 1, then a serial implementation is used instead of a parallel one. If None, then the number of processes will be set to the number of CPUs detected on the machine. Due to overheads, multiprocessing may be slower than serial processing. This is especially true if one only has a small number of sources to deblend. The benefits of multiprocessing require ~1000 or more sources to deblend, with larger gains as the number of sources increase.
    - progress_bar : bolean, whether to display a progress bar while the deblending is taking place
    - show_plots : bolean, whether to display plots of both the segmented and deblended images
    - psf : 2D array representing the PSF, where the central pixel corresponds to the center of the PSF. Typically, including this keyword argument will make the code run slower by a factor of a few, depending on the size of the PSF, but the resulting Sersic fits will be more correct
    - cutout_extent : target fractional size of the data cutout relative to the minimal bounding box containing the source. The value must be >= 1
    - min_cutout_size : minimum size of the cutout, in case cutout_extent times the size of the minimal bounding box is smaller than this. Any given value will be truncated to an even number
    - n_sigma_outlier : number of standard deviations that define a pixel as an outlier, relative to its 8 neighbors. Outlying pixels are removed as described in Lotz et al. (2004). If the value is zero or negative, outliers are not removed
    - annulus_width : The width of the annuli used to calculate the Petrosian radius and other quantities
    - eta : Petrosian eta parameter used to define the Petrosian radius. For a circular or elliptical aperture at the Petrosian radius, the mean flux at the edge of the aperture divided by the mean flux within the aperture is equal to eta
    - petro_fraction_gini : fraction of the Petrosian radius used as a smoothing scale in order to define the pixels that belong to the galaxy in the Gini calculation
    - skybox_size : target size of the skybox used to characterize the sky background
    - petro_extent_cas : radius of the circular aperture used for the asymmetry calculation, in units of the circular Petrosian radius
    - petro_fraction_cas : fraction of the Petrosian radius used as a smoothing scale in the CAS calculations
    - boxcar_size_mid : size of the constant kernel used to regularize the MID segmap in the MID calculations
    - niter_bh_mid : number of iterations in the basin-hopping stage of the maximization when calculating the multimode statistic. A value of at least 100 is recommended for “production” runs, but it can be time-consuming
    - sigma_mid : smoothing scale (in pixels) used to compute the intensity (I) statistic in the MID calculations
    - petro_extent_flux : number of Petrosian radii used to define the aperture over which the flux is measured. This is also used to define the inner radius of the elliptical aperture used to estimate the sky background in the shape asymmetry calculation
    - boxcar_size_shape_asym : size of the constant kernel used to regularize the segmap when calculating the shape asymmetry segmap
    - sersic_fitting_args : dictionary of keyword arguments passed to Astropy’s LevMarLSQFitter, which cannot include z or weights (these are handled internally by statmorph)
    - sersic_model_args : dictionary of keyword arguments passed to Astropy’s Sersic2D. By default, statmorph will make reasonable initial guesses for all the model parameters, although this functionality can be overridden for more customized fits. By default, statmorph also imposes a lower bound of n = 0.01 for the fitted Sersic index
    - sersic_maxiter : Deprecated. Use sersic_fitting_args instead
    - include_doublesersic : bolean, whether to also fit a double 2D Sersic model
    - doublesersic_rsep_over_rhalf : specifies the boundary used to separate the inner and outer regions of the image, which are used to perform single Sersic fits and thus construct an initial guess for the double Sersic fit. This argument is provided as a multiple of the (non-parametric) half-light semimajor axis rhalf_ellip
    - doublesersic_tied_ellip : bolean, if True both components of the double Sersic model share the same ellipticity and position angle. The same effect could be achieved using doublesersic_model_args in combination with the 'tied' argument, although the syntax would be slightly more involved
    - doublesersic_fitting_args : same as sersic_fitting_args, but for the double 2D Sersic fit
    - doublesersic_model_args : same as sersic_model_args, but for the double 2D Sersic fit
    - segmap_overlap_ratio : minimum ratio (in order to have a 'good' measurement) between the area of the intersection of the Gini and MID segmaps and the area of the largest of these two segmaps
    - verbose : bolean, If True this prints various minor warnings (which do not result in 'bad' measurements) during the calculations
    
    Documentation :
    - https://photutils.readthedocs.io/en/stable/api/photutils.segmentation.detect_sources.html#detect-sources
    - https://photutils.readthedocs.io/en/stable/api/photutils.segmentation.detect_threshold.html#detect-threshold
    - https://photutils.readthedocs.io/en/stable/api/photutils.segmentation.deblend_sources.html#deblend-sources
    - https://statmorph.readthedocs.io/en/latest/api.html#statmorph.source_morphology
    """    
    
    segment_img = detection_image.source_detection(detection_image.sci, nsigma, npixels, connectivity, detection_image.mask, 0.0, detection_image.bkg_err, smooth_data, kernel_name, smooth_fwhm, kernel_size)
    deblend_img = detection_image.source_deblending(detection_image.sci, segment_img, npixels, labels, nlevels, contrast, mode, connectivity, relabel, nproc, progress_bar)
    
    if show_plots is True:
        fig1, ax1 = plt.subplots(figsize=(8, 8))
        ax1.imshow(segment_img, origin='lower', cmap=deblend_img.cmap, interpolation='nearest')
        ax1.set_title('Segmentation Image')
        fig2, ax2 = plt.subplots(figsize=(8, 8))
        ax2.imshow(deblend_img, origin='lower', cmap=deblend_img.cmap, interpolation='nearest')
        ax2.set_title('Deblended Segmentation Image')
        plt.show()
    
    with h5py.File(output_hdf5_filename, 'w') as f:
        main = f.create_group('Morph')
        detection_img_group = main.create_group('Detection_Image')
        detection_img_name = detection_img_group.create_group(detection_image.img_name)
        detection_img_name.create_dataset('segmentation_map', data=deblend_img)
        
        filter_img_group = main.create_group('Filter_Images')
        for filter_image in filter_images:
            filter_img_name = filter_img_group.create_group(filter_image.img_name)
            morphologies = filter_image.morphology(filter_image.sci, deblend_img, filter_image.mask, filter_image.wht, filter_image.gain, psf, cutout_extent, min_cutout_size, n_sigma_outlier, annulus_width, eta, petro_fraction_gini, skybox_size, petro_extent_cas, petro_fraction_cas, boxcar_size_mid, niter_bh_mid, sigma_mid, petro_extent_flux, boxcar_size_shape_asym, sersic_fitting_args, sersic_model_args, sersic_maxiter, include_doublesersic, doublesersic_rsep_over_rhalf, doublesersic_tied_ellip, doublesersic_fitting_args, doublesersic_model_args, segmap_overlap_ratio, verbose)
            for morph_property in morph_properties:
                morph_data = [getattr(morph, morph_property) for morph in morphologies]
                filter_img_name.create_dataset(morph_property, data=np.array(morph_data))
                
def process_images_photometry(detection_image, filter_images, phot_properties, nsigma, npixels, nlevels, contrast, kron_params, output_hdf5_filename, labels=None, connectivity=8, mode='exponential', local_bkg_width=0, apermask_method='correct', smooth_data=True, kernel_name='Tophat', smooth_fwhm=3, kernel_size=5, detection_cat=None, relabel=True, wcs='None', nproc=1, progress_bar=True, show_plots=True):

    """
    Returns a hdf5 file containing the desired photometric properties of all the galaxies identified for each given filter image. The file also contains the deblended segmentation map used to obtain these properties, derived from the given detection image. The SourceCatalog object is also saved in self.photometry_cat.
    The returned hdf5 file has the following format :
    Photo/Detection_Image/detection_image name/segmentation_map
    Photo/Filter_Images/filter_image name/phot_property
    
    Parameters :
    - detection_image : image object obtained using the image initializing classes of the detection image associated to the filter images
    - filter_images : list of image objects obtained using the image initializing classes of the filter images associated to the detection image
    - phot_properties : list of desired photometric properties (available properties : area, background, background_centroid, background_ma, background_mean, background_sum, bbox, bbox_xmax, bbox_xmin, bbox_ymax, bbox_ymin, centroid, centroid_quad, centroid_win, convdata, convdata_ma, covar_sigx2, covar_sigy2, covariance, covariance_eigvals, cutout_centroid, cutout_centroid_quad, cutout_centroid_win, cutout_maxval_index, cutout_minval_index, cxx, cxy, cyy, data, data_ma, eccentricity, ellipticity, elongation, equivalent_radius, error, error_ma, extra_properties, fwhm, gini, inertia_tensor, isscalar, kron_aperture, kron_flux, kron_fluxerr, kron_radius, label, labels, local_background, local_background_aperture, max_value, maxval_index, maxval_xindex, maxval_yindex, min_value, minval_index, minval_xindex, minval_yindex, moments, moments_central, nlabels, orientation, perimeter, properties, segment, segment_area, segment_flux, segment_fluxerr, segment_ma, semimajor_sigma, semiminor_sigma, skybbox_ll, skybox_lr, skybox_ul, skybox_ur, sky_centroid, sky_centroid_icrs, sky_centroid_quad, sky_centroid_win, slices, xcentroid, xcentroid_quad, xcentroid_win, ycentroid, ycentroid_quad, ycentroid_win)
    - nsigma : threshold value in units of standard deviation for detecting sources. Sources are pixels whose values are above nsigma times the local background standard deviation
    - npixels : minimum number of connected pixels above the threshold that an object must have to be deblended
    - nlevels : number of multi-thresholding levels to use for deblending. Each source will be re-thresholded at nlevels levels spaced between its minimum and maximum values (non-inclusive)
    - contrast : fraction of total source flux that a local peak must have (at any one of the multi-thresholds) to be deblended as a separate object
    - kron_params : list of parameters used to determine the Kron aperture. The first item is the scaling parameter of the unscaled Kron radius and the second item represents the minimum value for the unscaled Kron radius in pixels. The optional third item is the minimum circular radius in pixels. If kron_params[0] * kron_radius * sqrt(semimajor_sigma * semiminor_sigma) is less than or equal to this radius, then the Kron aperture will be a circle with this minimum radius
    - output_hdf5_filename : name of the output hdf5 file returned by the function
    - labels : label numbers to deblend. If None, then all labels in the segmentation image will be deblended
    - connectivity : type of pixel connectivity used in determining how pixels are grouped into a detected source. Available options are 8 and 4 where 8-connected pixels touch along their edges or corners while 4-connected pixels touch along their edges
    - mode : mode used in defining the spacing between the multi-thresholding levels during deblending. Available options are 'exponential', 'linear' and 'sinh'
    - local_bkg_width : width of the rectangular annulus used to compute a local background around each source. If zero, then no local background subtraction is performed. The local background affects the min_value, max_value, segment_flux, kron_flux, and fluxfrac_radius properties. It is also used when calculating circular and Kron aperture photometry. It does not affect the moment-based morphological properties of the source.
    - apermask_method : method used to handle neighboring sources when performing aperture photometry. This parameter also affects the Kron radius. 'correct' replaces pixels assigned to neighboring sources by replacing them with pixels on the opposite side of the source center (equivalent to MASK_TYPE=CORRECT in SourceExtractor).'mask' masks pixels assigned to neighboring sources (equivalent to MASK_TYPE=BLANK in SourceExtractor). 'none' does not mask any pixels (equivalent to MASK_TYPE=NONE in SourceExtractor)
    - smooth_data : bolean, whether to convolve the data with a smoothing kernel before source detection
    - kernel_name : name of kernel used to convolve data. Only Tophat and Gaussian are supported at the moment
    - smooth_fwhm : width of kernel
    - kernel_size : size of kernel
    - detection_cat : SourceCatalog object for the detection image. The segmentation image used to create the detection catalog must be the same one input to segment_img. If input, then the detection catalog source centroids and morphological/shape properties will be returned instead of calculating them from the input data. The detection catalog centroids and shape properties will also be used to perform aperture photometry. If detection_cat is input, then the input wcs, apermask_method, and kron_params keywords will be ignored. This keyword affects circular_photometry (including returned apertures), all Kron parameters (Kron radius, flux, flux errors, apertures, and custom kron_photometry), and fluxfrac_radius (which is based on the Kron flux).
    - relabel : bolean, whether to relabel the segmentation image such that the labels are in consecutive order starting from 1.
    - wcs : A world coordinate system (WCS) transformation that supports the astropy shared interface for WCS (ex : astropy.wcs.WCS, gwcs.wcs.WCS). If None, then all sky-based properties will be set to None
    - nproc : number of processes to use for multiprocessing (if larger than 1). If set to 1, then a serial implementation is used instead of a parallel one. If None, then the number of processes will be set to the number of CPUs detected on the machine. Due to overheads, multiprocessing may be slower than serial processing. This is especially true if one only has a small number of sources to deblend. The benefits of multiprocessing require ~1000 or more sources to deblend, with larger gains as the number of sources increase.
    - progress_bar : bolean, whether to display a progress bar while the deblending is taking place
    - show_plots : bolean, whether to display plots of both the segmented and deblended images
       
    Documentation :
    - https://photutils.readthedocs.io/en/stable/api/photutils.segmentation.detect_sources.html#detect-sources
    - https://photutils.readthedocs.io/en/stable/api/photutils.segmentation.detect_threshold.html#detect-threshold
    - https://photutils.readthedocs.io/en/stable/api/photutils.segmentation.deblend_sources.html#deblend-sources
    - https://photutils.readthedocs.io/en/stable/api/photutils.segmentation.SourceCatalog.html#sourcecatalog
    """    
    
    segment_img = detection_image.source_detection(detection_image.sci, nsigma, npixels, connectivity, detection_image.mask, 0.0, detection_image.bkg_err, smooth_data, kernel_name, smooth_fwhm, kernel_size)
    deblend_img = detection_image.source_deblending(detection_image.sci, segment_img, npixels, labels, nlevels, contrast, mode, connectivity, relabel, nproc, progress_bar)
    
    if show_plots is True:
        fig1, ax1 = plt.subplots(figsize=(8, 8))
        ax1.imshow(segment_img, origin='lower', cmap=deblend_img.cmap, interpolation='nearest')
        ax1.set_title('Segmentation Image')
        fig2, ax2 = plt.subplots(figsize=(8, 8))
        ax2.imshow(deblend_img, origin='lower', cmap=deblend_img.cmap, interpolation='nearest')
        ax2.set_title('Deblended Segmentation Image')
        plt.show()    
    
    with h5py.File(output_hdf5_filename, 'w') as f:
        main = f.create_group('Photo')
        detection_img_group = main.create_group('Detection_Image')
        detection_img_name = detection_img_group.create_group(detection_image.img_name)
        detection_img_name.create_dataset('segmentation_map', data=deblend_img)
        
        filter_img_group = main.create_group('Filter_Images')
        for filter_image in filter_images:
            filter_img_name = filter_img_group.create_group(filter_image.img_name)
            if smooth_data==True:
                convolved_data = filter_image.smooth_data(filter_image.sci, kernel_name, smooth_fwhm, kernel_size)
            else:
                convolved_data=None        
            photometry_cat = filter_image.photometry(filter_image.sci, deblend_img, convolved_data, filter_image.err, filter_image.mask, filter_image.bkg, wcs, local_bkg_width, apermask_method, kron_params, detection_cat, progress_bar)
            for phot_property in phot_properties:
                phot_data = [getattr(photometry_cat, phot_property)]
                filter_img_name.create_dataset(phot_property, data=np.array(phot_data))    

def process_images_number_counts(images, nsigma, npixels, nlevels, contrast, labels=None, connectivity=8, mode='exponential', smooth_data=True, kernel_name='Tophat', smooth_fwhm=3, kernel_size=5, relabel=True, nproc=1, progress_bar=True, show_plots=True):

    """
    Returns a dictionary containing number of galaxies identified on each given image.
    
    Parameters :
    - images : list of image objects obtained using the image initializing classes
    - nsigma : threshold value in units of standard deviation for detecting sources. Sources are pixels whose values are above nsigma times the local background standard deviation
    - npixels : minimum number of connected pixels above the threshold that an object must have to be deblended
    - nlevels : number of multi-thresholding levels to use for deblending. Each source will be re-thresholded at nlevels levels spaced between its minimum and maximum values (non-inclusive)
    - contrast : fraction of total source flux that a local peak must have (at any one of the multi-thresholds) to be deblended as a separate object
    - labels : label numbers to deblend. If None, then all labels in the segmentation image will be deblended
    - connectivity : type of pixel connectivity used in determining how pixels are grouped into a detected source. Available options are 8 and 4 where 8-connected pixels touch along their edges or corners while 4-connected pixels touch along their edges
    - mode : mode used in defining the spacing between the multi-thresholding levels during deblending. Available options are 'exponential', 'linear' and 'sinh'
    - smooth_data : bolean, whether to convolve the data with a smoothing kernel before source detection
    - kernel_name : name of kernel used to convolve data. Only Tophat and Gaussian are supported at the moment
    - smooth_fwhm : width of kernel
    - kernel_size : size of kernel
    - relabel : bolean, whether to relabel the segmentation image such that the labels are in consecutive order starting from 1.
    - nproc : number of processes to use for multiprocessing (if larger than 1). If set to 1, then a serial implementation is used instead of a parallel one. If None, then the number of processes will be set to the number of CPUs detected on the machine. Due to overheads, multiprocessing may be slower than serial processing. This is especially true if one only has a small number of sources to deblend. The benefits of multiprocessing require ~1000 or more sources to deblend, with larger gains as the number of sources increase.
    - progress_bar : bolean, whether to display a progress bar while the deblending is taking place
    - show_plots : bolean, whether to display plots of both the segmented and deblended images    

    Documentation :
    - https://photutils.readthedocs.io/en/stable/api/photutils.segmentation.detect_sources.html#detect-sources
    - https://photutils.readthedocs.io/en/stable/api/photutils.segmentation.detect_threshold.html#detect-threshold
    - https://photutils.readthedocs.io/en/stable/api/photutils.segmentation.deblend_sources.html#deblend-sources
    """    
    
    number_counts = {}
    for i in images:
        segment_img = i.source_detection(i.sci, nsigma, npixels, connectivity, i.mask, 0.0, i.bkg_err, smooth_data, kernel_name, smooth_fwhm, kernel_size)
        deblend_img = i.source_deblending(i.sci, segment_img, npixels, labels, nlevels, contrast, mode, connectivity, relabel, nproc, progress_bar)
        number_counts[i.img_name] = len(deblend_img.labels)
        
        if show_plots is True:
            fig1, ax1 = plt.subplots(figsize=(8, 8))
            ax1.imshow(segment_img, origin='lower', cmap=deblend_img.cmap, interpolation='nearest')
            ax1.set_title(f"{i.img_name} Segmentation Image")
            fig2, ax2 = plt.subplots(figsize=(8, 8))
            ax2.imshow(deblend_img, origin='lower', cmap=deblend_img.cmap, interpolation='nearest')
            ax2.set_title(f"{i.img_name} Deblended Segmentation Image")
            plt.show()
    
    return number_counts              

def make_cutout(image, row=None, col=None, width=None, height=None, rowmin=None, rowmax=None, colmin=None, colmax=None, cutout_img_name='cutout_image'):
    """
    Returns a cutout of the given image, initialized using the ImageFromArrays class. The cutout can be created using row/col/width/height or rowmin/rowmax/colmin/colmax.
    """
    
    return image.cutout(row, col, width, height, rowmin, rowmax, colmin, colmax, cutout_img_name)
    
def make_img_panel(image, vmin=None, vmax=None, scaling=False, cmap=cm.magma):
    """
    Returns a representation of the given image.
    """
    
    fig, ax = plt.subplots()
    image.img_panel(ax, image.sci, vmin, vmax, scaling, cmap)
    plt.show()
        
def make_significance_panel(image, threshold = 2.5):
    """
    Returns a pixel significance plot of the given image.
    """
    
    fig, ax = plt.subplots()
    image.significance_panel(ax, threshold)
    plt.show()

#------------------------------------------------------------------------------
# Supporting functions
#------------------------------------------------------------------------------

class Image:

    def source_detection(self, data, nsigma, npixels, connectivity, mask, background, bkg_error, smooth_data, kernel_name, smooth_fwhm, kernel_size):

        threshold = detect_threshold(data, nsigma, background=background, error=bkg_error, mask=mask)

        if smooth_data==True:
            self.convolved_data = self.smooth_data(data, kernel_name, smooth_fwhm, kernel_size)
            segmentation_image = detect_sources(self.convolved_data, threshold, npixels, connectivity=connectivity, mask=mask)
        elif smooth_data==False:
            segmentation_image = detect_sources(data, threshold, npixels, connectivity=connectivity, mask=mask)
        else:
            raise ValueError('invalid value given to "smooth_data"')

        return segmentation_image

    def source_deblending(self, data, segment_img, npixels, labels, nlevels, contrast, mode, connectivity, relabel, nproc, progress_bar):

        deblended_image = deblend_sources(data, segment_img, npixels, labels=labels, nlevels=nlevels, contrast=contrast, mode=mode, connectivity=connectivity, relabel=relabel, nproc=nproc, progress_bar=progress_bar)

        return deblended_image

    def morphology(self, data, segmap, mask, weightmap, gain, psf, cutout_extent, min_cutout_size, n_sigma_outlier, annulus_width, eta, petro_fraction_gini, skybox_size, petro_extent_cas, petro_fraction_cas, boxcar_size_mid, niter_bh_mid, sigma_mid, petro_extent_flux, boxcar_size_shape_asym, sersic_fitting_args, sersic_model_args, sersic_maxiter, include_doublesersic, doublesersic_rsep_over_rhalf, doublesersic_tied_ellip, doublesersic_fitting_args, doublesersic_model_args, segmap_overlap_ratio, verbose):
        
        morphology_list = statmorph.source_morphology(data, segmap, mask=mask, weightmap=weightmap, gain=gain, psf=psf, cutout_extent=cutout_extent, min_cutout_size=min_cutout_size, n_sigma_outlier=n_sigma_outlier, annulus_width=annulus_width, eta=eta, petro_fraction_gini=petro_fraction_gini, skybox_size=skybox_size, petro_extent_cas=petro_extent_cas, petro_fraction_cas=petro_fraction_cas, boxcar_size_mid=boxcar_size_mid, niter_bh_mid=niter_bh_mid, sigma_mid=sigma_mid, petro_extent_flux=petro_extent_flux, boxcar_size_shape_asym=boxcar_size_shape_asym, sersic_fitting_args=sersic_fitting_args, sersic_model_args=sersic_model_args, sersic_maxiter=sersic_maxiter, include_doublesersic=include_doublesersic, doublesersic_rsep_over_rhalf=doublesersic_rsep_over_rhalf, doublesersic_tied_ellip=doublesersic_tied_ellip, doublesersic_fitting_args=doublesersic_fitting_args, doublesersic_model_args=doublesersic_model_args, segmap_overlap_ratio=segmap_overlap_ratio, verbose=verbose)
        self.morphologies = morphology_list
        
        return morphology_list        
    
    def photometry(self, data, segment_img, convolved_data, error, mask, background, wcs, localbkg_width, apermask_method, kron_params, detection_cat, progress_bar):
        
        if detection_cat is None:
            detection_catalog = SourceCatalog(data, segment_img, convolved_data=convolved_data, error=error, mask=mask, background=background, wcs=wcs, localbkg_width=localbkg_width, apermask_method=apermask_method, kron_params=kron_params, detection_cat=detection_cat, progress_bar=progress_bar)
            photometry_catalog = SourceCatalog(data, segment_img, convolved_data=convolved_data, error=error, mask=mask, background=background, wcs=wcs, localbkg_width=localbkg_width, apermask_method=apermask_method, kron_params=kron_params, detection_cat=detection_catalog, progress_bar=progress_bar)
        else:
            photometry_catalog = SourceCatalog(data, segment_img, convolved_data=convolved_data, error=error, mask=mask, background=background, wcs=wcs, localbkg_width=localbkg_width, apermask_method=apermask_method, kron_params=kron_params, detection_cat=detection_cat, progress_bar=progress_bar)
            
        self.photometry_cat = photometry_catalog
        
        return photometry_catalog

    def smooth_data(self, data, kernel_name, smooth_fwhm, kernel_size):

        if kernel_name == 'Gaussian':
            smooth_sigma = smooth_fwhm * gaussian_fwhm_to_sigma
            smooth_kernel = Gaussian2DKernel(smooth_sigma, x_size=kernel_size, y_size=kernel_size)
        elif kernel_name == 'Tophat':
            smooth_sigma = smooth_fwhm / np.sqrt(2)
            smooth_kernel = Tophat2DKernel(smooth_sigma, x_size=kernel_size, y_size=kernel_size)
        else :
            raise ValueError('Kernel not supported: {}'.format(kernel_name))

        smooth_kernel.normalize()
        convolved_data = convolve(data, smooth_kernel)
        self.convolved = convolved_data
        
        return convolved_data       

    def cutout(self, row=None, col=None, width=None, height=None, rowmin=None, rowmax=None, colmin=None, colmax=None, img_name='cutout_image'):

        if rowmin is not None and rowmax is not None and colmin is not None and colmax is not None:
            height = rowmax - rowmin
            width = colmax - colmin
        elif row is not None and col is not None and width is not None and height is not None:
            rowmin = row - height // 2
            rowmax = row + height // 2
            colmin = col - width // 2
            colmax = col + width // 2
        else:
            raise ValueError("Invalid arguments provided")

        rowmin = int(np.round(rowmin, 0))
        colmin = int(np.round(colmin, 0))

        rowstart = 0
        colstart = 0
        rowend = height
        colend = width

        if rowmin < 0:
            rowstart = -rowmin
            rowmin = 0
            print('Cutout rowmin below 0')
        if colmin < 0:
            colstart = -colmin
            colmin = 0
            print('Cutout colmin below 0')
        if rowmax > self.sci.shape[0]:
            rowend -= rowmax - self.sci.shape[0]
            rowmax = self.sci.shape[0]
            print('Cutout rowmax above image boundary')
        if colmax > self.sci.shape[1]:
            colend -= colmax - self.sci.shape[1]
            colmax = self.sci.shape[1]
            print('Cutout colmax above image boundary')

        data = np.zeros((height, width))
        bkg = np.zeros((height, width))
        bkg_err = np.zeros((height, width))
        wht = np.zeros((height, width))
        gain = np.zeros((height, width))
        err = np.zeros((height, width))
        mask = np.zeros((height, width), dtype=bool)   
            
        data[rowstart:rowend, colstart:colend] = self.sci[rowmin:rowmax, colmin:colmax]
        if self.bkg is None:
            bkg = None
        else:
            bkg[rowstart:rowend, colstart:colend] = self.bkg[rowmin:rowmax, colmin:colmax]
        if self.bkg_err is None:
            bkg_err = None
        else:
            bkg_err[rowstart:rowend, colstart:colend] = self.bkg_err[rowmin:rowmax, colmin:colmax]            
        if self.wht is None:
            wht = None
        else:
            wht[rowstart:rowend, colstart:colend] = self.wht[rowmin:rowmax, colmin:colmax]
        if self.gain is None:
            gain = None
        else:
            gain[rowstart:rowend, colstart:colend] = self.gain[rowmin:rowmax, colmin:colmax]
        if self.err is None:
            err = None
        else:
            err[rowstart:rowend, colstart:colend] = self.err[rowmin:rowmax, colmin:colmax]
        if self.mask is None:
            mask = None
        else:
            mask[rowstart:rowend, colstart:colend] = self.mask[rowmin:rowmax, colmin:colmax].astype(bool)

        return ImageFromArrays(data, img_name, bkg_sub=bkg, bkg_err=bkg_err, wht=wht, gain=gain, err_tot=err, mask=mask)

    def img_panel(self, ax, im, vmin=None, vmax=None, scaling=False, cmap=cm.magma):
        
        if vmin is None:
            vmin = np.min(im)
        if vmax is None:
            vmax = np.max(im)

        if scaling:
            im = scaling(im)

        ax.axis('off')
        ax.imshow(im, cmap=cmap, vmin=vmin, vmax=vmax, origin='lower')  # choose better scaling
        
        return ax
    
    def significance_panel(self, ax, threshold = 2.5):
        
        sig = (self.sci)/self.err

        ax.imshow(sig, cmap = cm.Greys, vmin = -threshold*2, vmax = threshold*2, origin = 'lower', interpolation = 'none')
        ax.imshow(np.ma.masked_where(sig <= threshold, sig), cmap = cm.plasma, vmin = threshold, vmax = 100, origin = 'lower', interpolation = 'none')
        ax.set_axis_off()

        return ax

#------------------------------------------------------------------------------
# Image initializing
#------------------------------------------------------------------------------

class ImageFromFITS(Image):

    def __init__(self, img_file, img_name, idata={'sci_bkgsub': 1, 'bkg_sub': 2, 'bkg_err': 3, 'wht': 4, 'gain': 5, 'err_tot': 6}, mask=None, mask_edge_thickness=10):
        """
        Generates an instance of image class from a FITS file.
        
        sci_bkgsub : background subtracted science data
        bkg_sub : subtracted background
        bkg_err : background error map
        wht : weight map
        gain : gain map
        err_tot : total error map
        """

        self.hdu = fits.open(img_file)
        self.img_name = img_name

        self.sci = self.hdu[idata['sci_bkgsub']].data
        self.bkg = self.hdu[idata['bkg_sub']].data if 'bkg_sub' in idata else None
        self.bkg_err = self.hdu[idata['bkg_err']].data if 'bkg_err' in idata else None
        self.wht = self.hdu[idata['wht']].data if 'wht' in idata else None
        self.gain = self.hdu[idata['gain']].data if 'gain' in idata else None
        self.err = self.hdu[idata['err_tot']].data if 'err_tot' in idata else None
        self.mask = mask

        if self.wht is not None and self.err is None:
            self.err = 1 / np.sqrt(self.wht)
        elif self.wht is None and self.err is not None:
            self.wht = (1/self.err)**2
        elif self.wht is None and self.err is None:
            raise ValueError('Image must have an error map or weight map')

        if self.mask is None and self.err is not None:
            self.mask = np.isnan(self.err)
            self.mask = ndimage.binary_dilation(self.mask, iterations=mask_edge_thickness)
     
class ImageFromArrays(Image):

    def __init__(self, data_bkgsub, img_name, bkg_sub = None, bkg_err = None, wht = None, gain = None, err_tot = None, mask = None, mask_edge_thickness=10):
        """
        Generates an instance of image class from different arrays.
                
        data_bkgsub : background subtracted science data
        bkg_sub : subtracted background
        bkg_err : background error map
        wht : weight map
        gain : gain map
        err_tot : total error map
        """

        self.img_name = img_name        
        self.sci = data_bkgsub
        self.bkg = bkg_sub
        self.bkg_err = bkg_err
        self.wht = wht
        self.gain = gain
        self.err = err_tot
        self.mask = mask
        
        if self.wht is not None and self.err is None:
            self.err = 1 / np.sqrt(self.wht)
        elif self.wht is None and self.err is not None:
            self.wht = (1/self.err)**2
        elif self.wht is None and self.err is None:
            raise ValueError('Image must have an error map or weight map')
                
        if self.mask is None:
            self.mask = np.isnan(self.err)
            self.mask = ndimage.binary_dilation(self.mask, iterations=mask_edge_thickness)            

class ImageFromDifferentSources(Image):
    
    def __init__(self, data_bkgsub_file, img_name, bkgsub_file = None, bkgerr_file = None, wht_file = None, gain_file = None, errtot_file = None, mask = None, mask_edge_thickness=10):
        """
        Generates an instance of image class from different files.
                
        data_bkgsub_file : background subtracted science data
        bkgsub_file : subtracted background
        bkgerr_file : background error map
        wht_file : weight map
        gain_file : gain map
        errtot_file : total error map
        """
        
        self.img_name = img_name
        self.sci = fits.open(data_bkgsub_file)[0].data
        self.bkg = fits.open(bkgsub_file)[0].data if bkgsub_file else None
        self.bkg_err = fits.open(bkgerr_file)[0].data if bkgerr_file else None
        self.wht = fits.open(wht_file)[0].data if wht_file else None
        self.gain = fits.open(gain_file)[0].data if gain_file else None
        self.err = fits.open(errtot_file)[0].data if errtot_file else None
        self.mask = mask
        
        if self.wht is not None and self.err is None:
            self.err = 1 / np.sqrt(self.wht)
        elif self.wht is None and self.err is not None:
            self.wht = (1/self.err)**2
        elif self.wht is None and self.err is None:
            raise ValueError('Image must have an error map or weight map')
                
        if self.mask is None:
            self.mask = np.isnan(self.err)
            self.mask = ndimage.binary_dilation(self.mask, iterations=mask_edge_thickness) 
        