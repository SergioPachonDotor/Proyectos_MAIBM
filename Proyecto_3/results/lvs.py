import numpy as np                        
from numba import njit
                        
@njit                        
def run_population(tmax=100.0, sampling_time=1.0, cells=1000) -> np.array:                        
	cells_arr = []                        
	for cell_indx in range(1, cells + 1):                        

		species = np.array([0.0, 0.0, 150, 100], dtype=np.float64)                        
		species[1] = cell_indx                        

		# Constants                        		
		α1 = 0.051
		α2 = 0.089
		β1 = 0.0001
		β2 = 0.001
		k1 = 300
		k2 = 300                        

		# Reaction matrix                        
		reaction_type = np.array([[0, 0, 1.0, 0], [0, 0, -1.0, 0], [0, 0, 0, 1.0], [0, 0, 0, -1.0]], dtype=np.int64)                        

		# Propensities initiation                        
		propensities = np.zeros(4, dtype=np.float64)                        
		tarr = np.arange(0, tmax,   sampling_time, dtype=np.float64)                        

		# Simulation Space                        
		sim  = np.zeros((len(tarr), len(species)), dtype=np.float64)                        
		sim[0] = species                        

		for indx_dt in range(1, len(tarr)):                        
			species = sim[indx_dt - 1]                        

			while species[0] < tarr[indx_dt]:                        
				# Species                        			
				x1 = species[2]
				x2 = species[3]                        

				# Propensities                        			
				propensities[0] = (1 - (x1/k1))*x1*α1
				propensities[1] = x1*x2*β1
				propensities[2] = (1 - (x2/k2))*x2*α2
				propensities[3] = x2*x1*β2                        

				τarr = np.zeros(len(propensities), dtype=np.float64)                        

				# Calculate tau times                        
				for indx_τ in range(len(propensities)):                        
					if propensities[indx_τ] > 0 :                        
						τarr[indx_τ] = -(1/propensities[indx_τ]) * np.log(np.random.rand())                        
					else:                        
						τarr[indx_τ] = np.inf                        

				τ = np.min(τarr)                        
				q = np.argmin(τarr)                        
				species = species + reaction_type[q] if -1 not in species + reaction_type[q] else species                        
				species[0] = species[0] + τ                        
			sim[indx_dt] = species                        
		cells_arr.append(sim)                        
	return cells_arr