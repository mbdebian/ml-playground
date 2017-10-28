# 
# Author    : Manuel Bernal Llinares
# Project   : ml-playground
# Timestamp : 26-10-2017 9:50
# ---
# © 2017 Manuel Bernal Llinares <mbdebian@gmail.com>
# All rights reserved.
# 

"""
Scratchpad / playground for the 4th chapter on the book
"""

import time
import pylab
import random

# Seed pseudo-random number generator
random.seed(time.time())

print("+++> Create a random 100x5 Matrix")
features = pylab.rand(100, 5)
print("+++> Create the prediction target")
target = pylab.rand(100) > 0.5

# First we're going to try the holdout method
print("[{} Holdout Method {}]".format("-" * 20, "-" * 20))
n = features.shape[0]
