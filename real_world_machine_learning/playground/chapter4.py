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