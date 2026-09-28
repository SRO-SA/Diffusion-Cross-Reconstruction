import numpy as np
from subprocess import run

# Indices meanings in segmentation matrices
I_CSF = 1
I_GM = 2
I_WM = 3
#

def cjv(mri_mat, seg_mat):
    """
    Coefficient of joint variation. The doc says lower is better.
    -------
    PARAMETERS :
        mri_mat : ndarray -> brain MRI intensities
        seg_mat : ndarray -> segmentation matrice
    RETURN :
        cjv : float
    """
    wm_intensities = mri_mat[seg_mat == I_WM]
    gm_intensities = mri_mat[seg_mat == I_GM]
    wm_mean = wm_intensities.mean()
    gm_mean = gm_intensities.mean()
    wm_std = np.sqrt(np.square(wm_intensities-wm_mean).mean())
    gm_std = np.sqrt(np.square(gm_intensities-gm_mean).mean())
    return (wm_std+gm_std) / abs(wm_mean-gm_mean)

def efc(mri_mat, brain_mask):
    """
    Entropy-focus criterion, uses Shannon entropy of voxel intensities.
    This code follows the definition induced by MRIQC code, except we use only
    brain intensities here. The paper says that lower values are better.
    --------
    PARAMETERS :
        mri_mat : ndarray -> brain MRI intensities
        brain_mask : ndarray -> binary brain mask
    RETURN :
        efc : float
    """
    intensities = mri_mat[brain_mask==1]
    n = len(intensities)
    efc_max = n * np.log(1/np.sqrt(n)) / np.sqrt(n)
    epsilon = 1e-10
    x_max = np.sqrt(np.square(intensities).sum())
    return (intensities/x_max * np.log((intensities+epsilon)/x_max)).sum() / efc_max

def fwhm(mri_path, mask_path):
    """
    The FWHM of the spatial distribution of the image intensity values in units
    of voxels. Uses the AFNI command 3dFWMx.
    --------
    PARAMETERS :
        mri_path : str -> brain MRI filepath
        mask_path : str -> brain mask filepath (can use '_seg.nii.gz' of FSL-FAST)
    RETURN :
        fwhm : float -> average fwhm over the 3 dimensions
    """
    cmd = f"3dFWHMx -mask {mask_path} -out - {mri_path}"
    stdout = run(cmd.split(' '), capture_output=True).stdout
    stdout = stdout.decode("utf-8").split("\n")[1]
    return np.mean([float(v) for v in stdout.split('  ')])

def rpve(pves_csf, pves_gm, pves_wm):
    """
    Residual partial volume effect, see MRIQC doc. The paper says smaller is better.
    ------
    PARAMETERS :
        rpve_csf, rpve_gm, rpve_gm : ndarrays -> residual partial volume effects
    RETURN :
        rpve_csf, rpve_gm, rpve_wm : ndarray of floats
    """
    res = np.empty(3, dtype="float")
    for i,rpve in enumerate((pves_csf, pves_gm, pves_wm)):
        rpve = rpve[rpve > 0]
        p2 = np.percentile(rpve, 2)
        p98 = np.percentile(rpve, 98)
        res[i] = (rpve[(rpve>=0.5) & (rpve<=p98)].sum() + (1 - rpve[(rpve>=p2) & (rpve<0.5)]).sum()) / len(rpve)
    return res

def snr(mri_mat, seg_mat):
    """
    Signal-to-noise ratio.
    --------
    PARAMETERS :
        mri_mat : ndarray -> brain MRI intensities
        seg_mat : ndarray -> segmentation matrice
    RETURN :
        snr_csf, snr_gm, snr_wm: ndarray of floats
    """
    csf_mask = seg_mat == I_CSF
    gm_mask = seg_mat == I_GM
    wm_mask = seg_mat == I_WM

    res = np.empty(4, dtype="float")
    for i,mask in enumerate((csf_mask, gm_mask, wm_mask)):
        intensities = mri_mat[mask]
        mean = intensities.mean()
        std = np.sqrt(np.square(intensities-mean).mean())
        n = len(intensities)
        res[i] = mean / (std*np.sqrt(n/(n-1)))
    return res

def wm2max(mri_mat, seg_mat):
    """
    From MRIQC paper : 'The white-matter to maximum intensity ratio is the median
    intensity within the WM mask over the 95% percentile of the full intensity
    distribution.' Here we translate full distribution as brain intensity distribution.
    --------
    PARAMETERS :
        mri_mat : ndarray -> brain MRI intensities
        seg_mat : ndarray -> segmentation matrice
    RETURN :
        wm2max : float
    """
    wm_mask = seg_mat == I_WM
    brain_mask = np.isin(seg_mat, (I_CSF, I_GM, I_WM))
    median_wm = np.median(mri_mat[wm_mask])
    p95 = np.percentile(mri_mat[brain_mask], 95)
    return median_wm / p95

