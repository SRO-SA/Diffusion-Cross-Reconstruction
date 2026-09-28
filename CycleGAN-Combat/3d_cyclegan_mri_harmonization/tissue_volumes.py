import numpy as np

def icvs(pve_csf, pve_gm, pve_wm):
    """
    Estimation of intracranial volume of each tissue.
    --------
    PARAMETERS :
        pve_csf, pve_gm, pve_gm : ndarrays -> partial volume maps
    RETURN :
        icv_csf, icv_gm, icv_wm : ndarray of floats
    """
    res = np.empty(3, dtype="float")
    vol_total = 0
    for i,pve in enumerate((pve_csf, pve_gm, pve_wm)):
        res[i] = pve.sum()
        vol_total += res[i]
    return res / vol_total