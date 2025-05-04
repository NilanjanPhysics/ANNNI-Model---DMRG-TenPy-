############################################################################################
################ Generic Imports for a DMRG Simulations done using TenPy ###################
############################################################################################

import numpy as np
from tenpy.networks.mps import MPS
from tenpy.models.model import CouplingMPOModel
from tenpy.algorithms.dmrg import SingleSiteDMRGEngine, TwoSiteDMRGEngine
from tenpy.networks.site import SpinHalfSite
from tenpy.models.lattice import Chain
from pprint import pprint

########################################### Logging setup ##################################
import tenpy
import logging
logging.basicConfig(level=logging.INFO)
tenpy.tools.misc.setup_logging(to_stdout="INFO")

#################### Impports Required For Parallel Computing ##############################
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures import ThreadPoolExecutor, as_completed

############################ Plotting package [Matplotlib] #################################
import matplotlib.pyplot as plt 
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib as mpl
mpl.rcParams.update({"text.usetex":True})
plt.rcParams['figure.dpi'] = 100

np.set_printoptions(precision=5, suppress=True, linewidth=100)

############################################################################################
############################## Model Definition : SpinChainNNN2 ############################
############################################################################################

class SpinChainNNN2(CouplingMPOModel):
    def init_sites(self, model_params):
        conserve = model_params.get('conserve', 'parity', str)
        if conserve == 'best':
            conserve = 'parity'
        return SpinHalfSite(conserve=conserve, sort_charge=True)

    def init_terms(self, model_params):
        Jx = model_params.get('Jx', 1.)
        Jxp = model_params.get('Jxp', 1.)
        hz = model_params.get('hz', 0.)
        for u in range(len(self.lat.unit_cell)):
            self.add_onsite(-hz, u, 'Sigmaz')
        for u1, u2, dx in self.lat.pairs['nearest_neighbors']:
            self.add_coupling(Jx, u1, 'Sigmax', u2, 'Sigmax', dx)
        for u1, u2, dx in self.lat.pairs['next_nearest_neighbors']:
            self.add_coupling(Jxp, u1, 'Sigmax', u2, 'Sigmax', dx)


############################################################################################
################################### Parallel Computing Setup ###############################
############################################################################################

L = 50                                      ####### Length of the ANNNI Chain
k_vals = 0.25                               ####### Frustration Parameter : kappa
h_vals = 1.25                               ####### External Magnetic Field (Transverse) : h
r_vals = np.linspace(1, L, L).astype(int)   ####### array of distances of sites from the first lattice point

####################################### DMRG parameters ###################################
'''
In the Gapped Phase (Ferromagnetic, Paramagnetic) of the ANNNI Phase Diagram 
the Bond-Dimension ('chi_max' in the dmrg_params) of the MPS can be kept low
for example < 100; But in the Gapless phase (Floating)/crticial points the bond dimension
should be significantly higher as the entanglement entropy sacles as log L
'''
dmrg_params = {
    'trunc_params': {'chi_max': 1000, 'svd_min': 1e-12},
    'mixer': True,
    'combine': True,
    'max_E_err': 1e-12,
    'max_sweeps': 20,
    'min_sweeps': 5,
}

############################ DMRG Calculation of Ground State ##############################
params = {'L': L, 'Jx': -1, 'Jxp': k_vals, 'hz': h_vals, 'bc_MPS': 'finite', 'conserve': 'best'}
model = SpinChainNNN2(params)                                                          #### Model Definition
psi_guess = MPS.from_product_state(model.lat.mps_sites(), ['up'] * L, 'finite')        #### Guessed Psi
eng = SingleSiteDMRGEngine(psi_guess, model, dmrg_params)                            
E, psi_final = eng.run()                                                               #### Compute Energy and Psi_final after DMRG

############################# Worker function for one parameter set ########################
def Run_friedel(r):
    corr = psi_final.correlation_function('Sigmax', 'Sigmax', [0], [r-1]).item()           #### Compute Correlation Function
    return corr

############################################################################################
def Parallel_DMRG():
    """
    Runs DMRG sweeps in parallel over r_values using threads.
    Returns:
        corr_arr (np.ndarray): Sigmax correlations ordered by r.
    """
    results = []
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(Run_friedel, r): r for r in r_vals}
        for future in as_completed(futures):
            r_val = futures[future]                      # grab the r associated with this future
            try:
                c = future.result()
                results.append((r_val, c))               # store the pair (r, correlation)
                logging.info(f"Completed r={r_val:.4f}: corr={c:.4e}")
            except Exception as e:
                logging.error(f"Error during r={r_val:.4f}: {e}")

    # sort by r so that corr_arr[i] corresponds to r_vals[i]
    results.sort(key=lambda pair: pair[0])            # pair is (r, c)
    _, corr_list = zip(*results)                      # unzip into two sequences
    return np.array(corr_list)                        # just return correlations in order


###################### Execute and extract arrays at module import ########################
def _Initialize():
    global corr_arr
    corr_arr = Parallel_DMRG()

############## This Runs the whole code ###################
_Initialize()


############################################################################################
#################### Some Datas Which was produced by me in my PC ##########################
############################################################################################

#### k = 0.5 , h = 0.1 , chi_max = 500
Corr_50_k0p5 = np.array([ 1.     ,  0.97476,  0.25384, -0.58475, -0.82559, -0.42524,  0.31594,  0.72473,  0.53946,
       -0.08332, -0.59581, -0.59938, -0.11533,  0.43874,  0.60594,  0.27365, -0.26609, -0.56223,
       -0.38618,  0.09416,  0.47516,  0.4499 ,  0.06142, -0.35618, -0.46454, -0.18823,  0.22045,
        0.43294,  0.27797, -0.08466, -0.36198, -0.32615, -0.03572,  0.26301,  0.33157,  0.12867,
       -0.15124, -0.29624, -0.18643,  0.04379,  0.22586,  0.20507,  0.04302, -0.13022, -0.18304,
       -0.0961 ,  0.02144,  0.11866,  0.10786,  0.10442])

#### k = 0.48 , h = 0.1 , chi_max = 700
Corr_50_k0p48 = np.array([ 1.     ,  0.97312,  0.57115, -0.10188, -0.5925 , -0.7059 , -0.43881,  0.04966,  0.46179,
        0.58614,  0.3836 , -0.01794, -0.37864, -0.50507, -0.3456 , -0.00356,  0.31737,  0.44216,
        0.3142 ,  0.0179 , -0.26955, -0.38998, -0.28566, -0.02638,  0.23143,  0.34507,  0.25823,
        0.02956, -0.20116, -0.30555, -0.23086, -0.0275 ,  0.17786,  0.27034,  0.20269,  0.01978,
       -0.16143, -0.23884, -0.1728 , -0.00514,  0.15269,  0.21084,  0.13978, -0.01981, -0.15457,
       -0.18685, -0.10061,  0.07028,  0.17956,  0.17917])

#### k = 0.25 , h = 0.1 , chi_max = 100
Corr_50_k0p25 = np.array([1.     , 0.98945, 0.98869, 0.98867, 0.98867, 0.98867, 0.98867, 0.98867, 0.98867, 0.98867,
       0.98867, 0.98867, 0.98867, 0.98867, 0.98867, 0.98867, 0.98867, 0.98867, 0.98867, 0.98867,
       0.98867, 0.98867, 0.98867, 0.98867, 0.98867, 0.98867, 0.98867, 0.98867, 0.98867, 0.98867,
       0.98867, 0.98867, 0.98867, 0.98867, 0.98867, 0.98867, 0.98867, 0.98867, 0.98867, 0.98867,
       0.98867, 0.98867, 0.98867, 0.98867, 0.98867, 0.98867, 0.98867, 0.98869, 0.9891 , 0.98194])

############################################################################################
########################################### PLOTS ##########################################
############################################################################################

plt.plot(Corr_50_k0p5, marker = 'o', markersize = 2, linewidth = 0.8, linestyle = '--', label=r'$h = 0.1, \kappa = 0.5, \chi_{max} = 500$')
plt.plot(Corr_50_k0p48, marker = 'o', markersize = 2, linewidth = 0.8, linestyle = '--', label=r'$h = 0.1, \kappa = 0.48, \chi_{max}=700$')
plt.ylabel(r'$\langle \sigma^x_0 \sigma^x_j \rangle$', fontsize = 14)
plt.xlabel(r'Site Index : $j$', fontsize = 14)
plt.legend(fontsize = 14)
plt.grid(True, linestyle="--", alpha=1)
plt.tight_layout()
plt.show()


###################### ONE Can Fit any expoentialy decaying or alegebrically decaying function them #############
