import numpy as np
import matplotlib.pyplot as plt

def Z(t, n_dosis, t0):
    k_a = 1.3
    k_e = 1/8
    
    d = 1000
    F = 0.031
    
    A = np.exp(-k_a*t0)
    B = np.exp(-k_e*t0)
    n = n_dosis
    
    V = 6
    parte_a = ((k_a * d * F)/(V * (k_a - k_e))) * ((1 - (B**n))/(1 - B)) * np.exp(-k_e * (t - ((n - 1) * t0)))
    parte_b = ((k_a * d * F)/(V * (k_a - k_e))) * ((1 - (A**n))/(1 - A)) * np.exp(-k_a * (t - ((n - 1) * t0)))
    z = parte_a - parte_b
    return z
  
if __name__ == '__main__':
    for i in range(0, 8):
        t = np.arange(8*i, 8 + 8*i, 0.01)
        plt.plot(t, Z(t, i+1, 8), color='black')
        plt.grid(True)
        plt.savefig('multidosis.jpg')
    
