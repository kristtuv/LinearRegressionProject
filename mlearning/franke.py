import numpy as np
class Franke:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def compute(self):
        term1 = 0.75*np.exp(-(0.25*(9*self.x-2)**2) - 0.25*((9*self.y-2)**2))
        term2 = 0.75*np.exp(-((9*self.x+1)**2)/49.0 - 0.1*(9*self.y+1))
        term3 = 0.5*np.exp(-(9*self.x-7)**2/4.0 - 0.25*((9*self.y-3)**2))
        term4 = -0.2*np.exp(-(9*self.x-4)**2 - (9*self.y-7)**2)
        return term1 + term2 + term3 + term4
        
