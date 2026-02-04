import h5py
import tifffile
import numpy as np
import os

# Input filename
os.chdir('/mnt/bigdata/BRUKER/TSeries-07132025-1042-002/')
h5_filename = "registered.h5"
output_filename = "registered.tif"

# Read HDF5 file
with h5py.File(h5_filename, 'r') as f:
    # Get the first dataset (adjust key as needed)
    data = f[list(f.keys())[0]][:]

# Convert to appropriate format and save as TIFF stack
data = np.asarray(data, dtype=np.uint16)
tifffile.imwrite(output_filename, data)

print(f"Converted {h5_filename} to {output_filename}")