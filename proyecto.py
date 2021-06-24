import lightkurve as lk
from astropy.table import Table 
from lightkurve.correctors import download_tess_cbvs
from lightkurve.correctors import CBVCorrector
from lightkurve.correctors import DesignMatrix
import numpy as np
import glob
import matplotlib.pyplot as plt

def correccion_curva_de_luz(archivo):

    tpf = lk.read(archivo)

    mask = np.array([[False, False, False, False, False, False, False, False, False, False],  # última fila
                            [False, False, False, False, False, False, False, False, False, False],
                            [False, False, False, False, False, False, False, True, False, False],
                            [False, False, False, True, True, True, True, True, False, False],
                            [False, True, True, True, True, True, True, True, False, False],
                            [False, True, True, True, True, True, True, True, False, False],
                            [False, False, True, True, True, True, True, True, False, False],
                            [False, False, True, True, True, True, True, True, False, False],
                            [False, False, True, True, True, True, True, True, False, False],
                            [False, False, False, False, True, True, True, False, False, False]]) # Primera fila

    #tpf.plot(aperture_mask=mask);
    lc = tpf.to_lightcurve(aperture_mask=mask)

    lc_clean = lc.remove_outliers(sigma=4)

    cbvs = download_tess_cbvs(sector=lc.sector, camera=lc.camera, ccd=lc.ccd, cbv_type='SingleScale')

    dm = DesignMatrix(tpf.flux[:, ~mask], name='pixel regressors').pca(5).append_constant()

    cbvcorrector = CBVCorrector(lc, interpolate_cbvs=True)

    cbvcorrector.correct_gaussian_prior(cbv_type=None, cbv_indices=None, ext_dm=dm, alpha=1e-4)
    #cbvcorrector.diagnose()


    # Select which CBVs to use in the correction
    cbv_type = ['SingleScale', 'Spike']
    # Select which CBV indices to use
    # Use the first 8 SingleScale and all Spike CBVS
    cbv_indices = [np.arange(1,9), 'ALL']


    cbvcorrector.correct(cbv_type=cbv_type, cbv_indices=cbv_indices, ext_dm=dm, alpha_bounds=[1e-6, 1e-2], target_over_score=0.8, target_under_score=-1)
    #cbvcorrector.diagnose();

    correcter_2 = cbvcorrector.corrected_lc
    
    t = Table([correcter_2["time"],correcter_2["flux"],correcter_2["flux_err"]])
    t.write(archivo + "_correccion.csv", format='csv', overwrite = True)
    #plt.plot(correcter_2["time"].value,correcter_2["flux"])
    #plt.show()
    
    
    
archivos = glob.glob('**/*.fits', recursive=True)

for archivo in archivos:
    correccion_curva_de_luz(archivo)   
                       


