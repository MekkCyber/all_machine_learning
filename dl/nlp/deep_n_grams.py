import os
import shutil
import trax
import trax.fastmath.numpy as np
import pickle
import numpy
import random as rnd
from trax import fastmath
from trax import layers as tl

filename = 'shakespeare.txt'
lines = []

counter = 0

with open(filename) as lines:
    for line in lines:        
        # remove leading and trailing whitespace
        pure_line = line.strip()
        # if pure_line is not the empty string,
        if pure_line:
            lines.append(pure_line)
print(f"Number of lines: {len(lines)}")
print(f"Sample line at position 0 {lines[0]}")

for i, line in enumerate(lines):
    # convert to all lowercase
    lines[i] = line.lower()

eval_lines = lines[-1000:] # Create a holdout validation set
lines = lines[:-1000] # Leave the rest for training

def line_to_tensor(line, EOS_int=1):
    tensor = []
    for c in line:
        c_int = ord(c)
        tensor.append(c_int)
    tensor.append(EOS_int)
    return tensor

print(line_to_tensor('abc xyz'))