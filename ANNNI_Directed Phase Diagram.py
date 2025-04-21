#######################################################################################
################################## PACKAGES ###########################################
################################### TenPy Packages ####################################
#######################################################################################

from tenpy.networks.mps import MPS
from tenpy.models.model import CouplingMPOModel
from tenpy.algorithms.dmrg import SingleSiteDMRGEngine, TwoSiteDMRGEngine
from tenpy.networks.site import SpinHalfSite
from tenpy.models.lattice import Chain

from pprint import pprint
import tenpy
tenpy.tools.misc.setup_logging(to_stdout="INFO")

#######################################################################################
################################### Basic Packages ####################################
#######################################################################################

import numpy as np
import matplotlib.pyplot as plt 
import matplotlib as mpl
mpl.rcParams.update({"text.usetex":True})

np.set_printoptions(precision=5, suppress=True, linewidth=100)
plt.rcParams['figure.dpi'] = 100

#######################################################################################
############################ Creating the Model : ANNNI Model #########################
#######################################################################################

class SpinChainNNN2(CouplingMPOModel):
  '''
  Here we create our own, customized model for 1D Transverse Field ANNNI Hamiltonian :
  ====================================================================================
    H = - Σ_j σ^x_j σ^x_{j+1} + κ Σ_j σ^x_j σ^x_{j+2} - h Σ_j σ^z_j
      ================================================================================
      With the Parameters :
      ===================================================================
      σ^x : Sigmax  (Pauli Matrices)
      σ^z : Sigmaz  (Pauli Matrices)
      Jx  : -1  (Nearest-Neighbor [NN] Ferromagnetic Coupling)
      Jxp : κ   (Next-Nearest-Neighbor [NNN] Antiferromagnetic Coupling)
      hz  : h   (Onsite Transverse Field)
      ===================================================================
  '''
    def init_sites(self, model_params):
        conserve = model_params.get('conserve', 'parity', str)
        assert conserve != 'Sigmaz'
        if conserve == 'best':
            conserve = 'parity'
            self.logger.info("%s: set conserve to %s", self.name, conserve)
        sort_charge = model_params.get('sort_charge', True, bool)
        site = SpinHalfSite(conserve = conserve, sort_charge=sort_charge)
        return site

    def init_terms(self, model_params):
        # (0) read out/set default parameters
        Jx = model_params.get('Jx', 1., 'real_or_array')
        Jxp = model_params.get('Jxp', 1., 'real_or_array')
        hz = model_params.get('hz', 0., 'real_or_array')
############### Loops for adding onsite, nearest-neighbour, next-nearest-neighbour terms! ##############
        for u in range(len(self.lat.unit_cell)):
            self.add_onsite(-hz, u, 'Sigmaz')
        for u1, u2, dx in self.lat.pairs['nearest_neighbors']:
            self.add_coupling(Jx, u1, 'Sigmax', u2, 'Sigmax', dx)
        for u1, u2, dx in self.lat.pairs['next_nearest_neighbors']:
            self.add_coupling(Jxp, u1, 'Sigmax', u2, 'Sigmax', dx)

#######################################################################################
############## Sign Changing term of the Order paramter : ANNNI Model #################
#######################################################################################

def sign(n):
  '''
  S(n) = √2 cos((n-1)π/2 - π/4) ; S = {+1, +1, -1, -1, +1, +1, ...}
  '''
    return np.sqrt(2) * np.cos( (n-1) * np.pi/2 - np.pi/4 )

#######################################################################################
############################ DMRG ALGORITHM Parameters ################################
#######################################################################################

dmrg_params = {
    'trunc_params': {
        'chi_max': 500,                #### Maximum Bond Dimension
        'svd_min': 1.e-10,             #### Singular value cutoff
    },
    'mixer': True,                     #### Enable mixer for SingleSiteDMRG
    'combine': True,
    'max_E_err': 1e-10,                #### Energy accuracy threshold
    'max_sweeps': 20,                  #### Maximum DMRG sweeps
    'min_sweeps': 5,                   #### Minimum DMRG sweeps
    # 'verbose': 1,                    #### Verbose output
}

##################################################
############ PARAMETERS for the Plot #############
##################################################
################## First Plot ####################
##################################################

h1 = np.linspace(1.75, 0.75, 5)
h2 = np.linspace(0.7, 0.35, 30)
h3 = np.linspace(0.36, 0.1, 5)
h_values1  = np.concatenate((h1, h2, h3))

##################################################
################## Second Plot ###################
##################################################

k1 = np.linspace(0.25, 0.4, 5)
k2 = np.linspace(0.42, 0.6, 30)
k3 = np.linspace(0.65, 0.75, 5)
k_values = np.concatenate((k1, k2, k3))

##################################################
################## Third Plot ####################
##################################################

h11 = np.linspace(0.1, 0.22, 5)
h22 = np.linspace(0.24, 0.75, 30)
h33 = np.linspace(0.8, 1.75, 5)
h_values2 = np.concatenate((h11, h22, h33))

#######################################################################################
############### Parameters required for ANNNI Model DMRG Simulations! #################
#######################################################################################

L = 51         ### System Size 

# k = 0.25     ### for 1st plot (with fixed kappa)
h = 0.1        ### for 2nd plot (with fixed h)
# k = 0.75     ### for 3rd plot (with fixed kappa)

entropyk2_annni = []
corrs_XXk2_annni = []

for k in k_values:
#### SpinChainNNN2 model parameters
    model_params = {
        'L': L,                  ##### System Size
        'Jx': -1,                ##### Nearest-neighbor Interaction
        'Jxp': k,                ##### Next-nearest-neighbor Interaction
        'hz': h,                 ##### Transverse Field
        'bc_MPS' : 'finite',     ##### Open Boundary Conditions (OBC) for finite 1D Chain
        'conserve': 'best'       ##### Conservation
    }
    annni_model = SpinChainNNN2(model_params) 
    ##### Initialize MPS
    psi = MPS.from_product_state(annni_model.lat.mps_sites(), ["up"] * L, "finite")

    ##### Run DMRG
    eng = SingleSiteDMRGEngine(psi, annni_model, dmrg_params)
    E, psi = eng.run()
    
    entropyk2_annni.append(np.max(psi.entanglement_entropy()))                                      ### Entanglement Entropy
    corrs_XXk2_annni.append(psi.correlation_function("Sigmax", 1 * "Sigmax", [0], [L//2]).item())   ### x-x Correlation Function

##############################################################################################################################################
#################################################### PLOTS for XX-Correlation Function #######################################################
##############################################################################################################################################

fig, ax = plt.subplots(1, 3, figsize=(17, 6))
fig.suptitle(r"Plots for Correlation function : $\langle\sigma^x_0 \sigma^x_{L/2}\rangle$ : for different System Sizes", fontsize=16)

####### Plot 1 ###########
ax[0].plot(h_values1[::-1], corrs_XX48k1_annni[::-1], marker='o', markersize = 2, label=r'$L = 48$')
ax[0].plot(h_values1[::-1], corrs_XX49k1_annni[::-1], marker='o', markersize = 2, label=r'$L = 49$')
ax[0].plot(h_values1[::-1], corrs_XX50k1_annni[::-1], marker='o', markersize = 2, label=r'$L = 50$')
ax[0].plot(h_values1[::-1], corrs_XX51k1_annni[::-1], marker='o', markersize = 2, label=r'$L = 51$')
ax[0].legend()
ax[0].invert_xaxis()
ax[0].grid(True, linestyle="--", alpha=0.5)
ax[0].set_title(r'$\kappa = 0.25$')
ax[0].set_xlabel(r'Magnetic Field : $h$', fontsize = 14)

####### Plot 2 ##########
ax[1].plot(k_values, corrs_XX48k2_annni, marker='o', markersize = 2, label=r'$L = 48$')
ax[1].plot(k_values, corrs_XX49k2_annni, marker='o', markersize = 2, label=r'$L = 49$')
ax[1].plot(k_values, corrs_XX50k2_annni, marker='o', markersize = 2, label=r'$L = 50$')
ax[1].plot(k_values, corrs_XX51k2_annni, marker='o', markersize = 2, label=r'$L = 51$')
ax[1].legend()
ax[1].set_yticks(np.arange(-0.75, 1.25, 0.25))
ax[1].grid(True, linestyle="--", alpha=0.5)
ax[1].set_title(r'$h=0.1$')
ax[1].set_xlabel(r'Frustration Parameter : $\kappa$', fontsize = 14)

####### Plot 3 #############
ax[2].plot(h_values2, corrs_XX48k3_annni, marker='o', markersize = 2, label=r'$L = 48$')
ax[2].plot(h_values2, corrs_XX49k3_annni, marker='o', markersize = 2, label=r'$L = 49$')
ax[2].plot(h_values2, corrs_XX50k3_annni, marker='o', markersize = 2, label=r'$L = 50$')
ax[2].plot(h_values2, corrs_XX51k3_annni, marker='o', markersize = 2, label=r'$L = 51$')
ax[2].legend()
ax[2].set_yticks(np.arange(-0.75, 1.25, 0.25))
ax[2].grid(True, linestyle="--", alpha=0.5)
ax[2].set_title(r'$\kappa = 0.75$')
ax[2].set_xlabel(r'Magnetic Field : $h$', fontsize = 14)

plt.tight_layout()
# plt.savefig("XX-Corelation_Ushaped.pdf", bbox_inches="tight")
plt.show()

##############################################################################################################################################
################################################# PLOTS for Half-Chain Entanglement Entropy ##################################################
##############################################################################################################################################

fig, ax = plt.subplots(1, 3, figsize=(17, 6))
fig.suptitle(r"Plots for Entanglement Entropy for different System Sizes", fontsize=16)

####### Plot 1 #############
ax[0].plot(h_values1[::-1], entropy48k1_annni[::-1], marker='o', markersize = 2, label=r'$L = 48$')
ax[0].plot(h_values1[::-1], entropy49k1_annni[::-1], marker='o', markersize = 2, label=r'$L = 49$')
ax[0].plot(h_values1[::-1], entropy50k1_annni[::-1], marker='o', markersize = 2, label=r'$L = 50$')
ax[0].plot(h_values1[::-1], entropy51k1_annni[::-1], marker='o', markersize = 2, label=r'$L = 51$')
ax[0].legend()
ax[0].invert_xaxis()
ax[0].set_yticks(np.arange(0, 1.5, 0.2))
ax[0].grid(True, linestyle="--", alpha=0.5)
ax[0].set_title(r'$\kappa = 0.25$')
ax[0].set_xlabel(r'Magnetic Field : $h$', fontsize = 14)

####### Plot 2 ############
ax[1].plot(k_values, entropy48k2_annni, marker='o', markersize = 2, label=r'$L = 48$')
ax[1].plot(k_values, entropy49k2_annni, marker='o', markersize = 2, label=r'$L = 49$')
ax[1].plot(k_values, entropy50k2_annni, marker='o', markersize = 2, label=r'$L = 50$')
ax[1].plot(k_values, entropy51k2_annni, marker='o', markersize = 2, label=r'$L = 51$')
ax[1].legend()
ax[1].set_yticks(np.arange(0, 1.5, 0.2))
ax[1].grid(True, linestyle="--", alpha=0.5)
ax[1].set_title(r'$h=0.1$')
ax[1].set_xlabel(r'Frustration Parameter : $\kappa$', fontsize = 14)

####### Plot 3 #############
ax[2].plot(h_values2, entropy48k3_annni, marker='o', markersize = 2, label=r'$L = 48$')
ax[2].plot(h_values2, entropy49k3_annni, marker='o', markersize = 2, label=r'$L = 49$')
ax[2].plot(h_values2, entropy50k3_annni, marker='o', markersize = 2, label=r'$L = 50$')
ax[2].plot(h_values2, entropy51k3_annni, marker='o', markersize = 2, label=r'$L = 51$')
ax[2].legend()
ax[2].set_yticks(np.arange(0, 1.5, 0.2))
ax[2].grid(True, linestyle="--", alpha=0.5)
ax[2].set_title(r'$\kappa = 0.75$')
ax[2].set_xlabel(r'Magnetic Field : $h$', fontsize = 14)

plt.tight_layout()
plt.savefig("Ent_Entropy_Ushaped.pdf", bbox_inches="tight")
plt.show()
