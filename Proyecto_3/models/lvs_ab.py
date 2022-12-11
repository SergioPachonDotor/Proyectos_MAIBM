import numpy as np                        
from numba import njit
                        
# @njit                        
def run_population(tmax=10.0, sampling_time=1.0, cells=10) -> np.array:
	antibiotico_inicial = 1500
	x1_i  = 400
	x2_i = 400
	tiempo_inicial = 0.0
	cells_arr = []
	dosis = 3
	tmax_arr = [i*tmax for i in range(dosis)]
	for ab in range(dosis):      
	# Constants                                       
		for cell_indx in range(1, cells + 1):                        

			species = np.array([0., 0., 400, 400, 1500], dtype=np.float64)                        
			species[1] = cell_indx      
    
			α1 = 1
			α2 = 0.25
			β1 = 0.0001
			β2 = 0.0001
			k1 = 500
			k2 = 500
			γ1 = 0.06666666666666667
			γ2 = 0.06666666666666667
			ε1 = 0.0005
			ε2 = 0.0001                        

			# Reaction matrix                        
			reaction_type = np.array([[0, 0, 1.0, 0, 0], [0, 0, -1.0, 0, 0], [0, 0, 0, 1.0, 0], [0, 0, 0, -1.0, 0], [0, 0, -1.0, 0, 0], [0, 0, 0, -1.0, 0], [0, 0, -1.0, 0, -1.0], [0, 0, 0, -1.0, -1.0]], dtype=np.int64)                        

			# Propensities initiation                        
			propensities = np.zeros(8, dtype=np.float64)                       
			tarr = np.arange(0, tmax,   sampling_time, dtype=np.float64)                        

			# Simulation Space                        
			sim  = np.zeros((len(tarr), len(species)), dtype=np.float64)                        
			sim[0] = species                        

			for indx_dt in range(0, len(tarr)):          # Revisar esto porque el len puede estar mal              
				species = sim[indx_dt - 1]                        

				while species[0] < tarr[indx_dt]:                        
					# Species                        			
					x1 = species[2]
					x2 = species[3]
					z  = species[4]                        

					# Propensities                        			
					propensities[0] = (1 - (x1/k1)) * x1 * α1
					propensities[1] = x1 * x2 * β1
					propensities[2] = (1 - (x2/k2)) * x2 * α2
					propensities[3] = x2 * x1 * β2
					propensities[4] = γ1 * x1
					propensities[5] = γ2 * x2
					propensities[6] = ε1 * z * x1
					propensities[7] = ε2 * z * x2                        

					τarr = np.zeros(len(propensities), dtype=np.float64)                        

					# Calculate tau times                        
					for indx_τ in range(len(propensities)):                        
						if propensities[indx_τ] > 0 :                        
							τarr[indx_τ] = -(1/propensities[indx_τ]) * np.log(np.random.rand())                        
						else:                        
							τarr[indx_τ] = np.inf                        

					τ = np.min(τarr)                        
					q = np.argmin(τarr)                        
					species = species + reaction_type[q] # if -1 not in species + reaction_type[q] else species                        
					species[0] = species[0] + τ                        
				sim[indx_dt] = species                        
			cells_arr.append(sim)
		if cell_indx == cells:
			x1_i  += sim[-1][2]
			x2_i  += sim[-1][3]
			antibiotico_inicial += sim[-1][4]

	return cells_arr