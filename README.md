# 1D Transverse Field ANNNI Model

Axial-Next-Nearest-Neighbor-Ising Model (ANNNI Model) is specific type of 1D Spin Model, where we have two competing interactions accompained by a transverse field. The Hamiltonian for this Model 
(in dimensionless parameters) can be written as,

$$ H_{ANNNI} = - \sum_{j=1}^{N-1} \sigma_j^x \sigma_{j+1}^x + \kappa \sum_{j=1}^{N-2} \ \sigma_{j}^x \sigma_{j+2}^x -h \sum_{j=1}^N \sigma_j^z $$

Where, $N$ is the length of the 1D Chain, $\kappa$ is the frustration parameter, $h$ is the external magnetic field and $h,\kappa \geq 0$

$$ \lim_{\kappa \to 0} H_{ANNNI} =  - \sum_{j=1}^{N-1} \sigma_j^x \sigma_{j+1}^x -h \sum_{j=1}^N \sigma_j^z \quad : \quad |\textcolor{cyan}{\uparrow\uparrow\uparrow} \cdot\cdot\cdot \textcolor{cyan}{\uparrow\uparrow\uparrow}\rangle \longrightarrow \underbrace{|\textcolor{red}{\rightarrow\rightarrow\rightarrow} \cdot\cdot\cdot \textcolor{red}{\rightarrow\rightarrow\rightarrow} \rangle}_{Paramagnetic}  $$

$$  \lim_{h \to 0} H_{ANNNI} =  - \sum_{j=1}^{N-1} \sigma_j^x \sigma_{j+1}^x+\kappa \sum_{j=1}^{N-2} \ \sigma_{j}^x \sigma_{j+2}^x \quad : \quad |\textcolor{cyan}{\uparrow\uparrow\uparrow} \cdot\cdot\cdot \textcolor{cyan}{\uparrow\uparrow\uparrow}\rangle \longrightarrow \underbrace{|\textcolor{blue}{\uparrow\uparrow} \textcolor{red}{\downarrow\downarrow}\textcolor{blue}{\uparrow\uparrow} \cdot\cdot\cdot  {} \rangle}_{Antiphase}$$
