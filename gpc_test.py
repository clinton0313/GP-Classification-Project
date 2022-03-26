from gpc import GPC
from utils import matern_kernel
import numpy as np

X = np.random.multivariate_normal([0, 0, 0], np.identity(3), 100)
Y = np.random.randint(0, 2, 100)

gpc = GPC(matern_kernel, [1, 1, 1.5], ["variance", "length", "nu"])

gpc.fit(X, Y, verbose=1)