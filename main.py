from mlearning.LinRegSK import OLSsk, Ridgesk, Lassosk
from mlearning import Franke
import numpy as np
x = np.random.rand(100,1)
y = np.random.rand(100,1)
f = Franke(x,y).compute() 
# print(f)
OLSsk(x, y, f, 3).prettyprint(3)
Ridgesk(x, y, f, 3).prettyprint(3)
Lassosk(x, y, f, 3).prettyprint(3)

