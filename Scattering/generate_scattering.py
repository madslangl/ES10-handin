import numpy as np
from scipy import fft
import xarray as xr
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import netCDF4

from utils import (
    distance_between_points,
    speed_of_sound,
    planeWaveScatSphere,
    pointSourceScatSphere,
    fibonacci_points_on_sphere,
    dataset_overview)

""" Generate a dataset of scattering data using a Fibonacci lattice for mic positions.        
        Dataset structure:
            - ir: Impulse responses (time, mic, source)
            - source_xyz: Source positions (source, xyz)
            - mic_xyz: Microphone positions (mic, xyz)
            - mic_angles: Microphone angles (mic, angle_type)
            - source_distance: Distances of sources from the center (source)
            - relative_elevation: Relative elevation angles of microphones (mic)
            - relative_azimuth: Relative azimuth angles of microphones (mic)
    """


output_path=r"Scattering\outputs\fibonacci_scattering_data.nc"
fs = 16000
T = 20  # temp
c = speed_of_sound(T)

head_size = 0.0875  # radius
nFreq = 481
freqVec = np.linspace(20, fs/2, nFreq) 

# n_points = 1000
n_points = 10 # for testing 
points, angles = fibonacci_points_on_sphere(n_points=n_points, radius=1.0, degrees=False)
points = points * head_size  # Scale unit sphere to head radius

# distArray = np.array(np.linspace(0.2, 2, 30)) 
distArray = np.array([0.4,0.6,0.8]) # for testing (3 x 10 points)
nDist = len(distArray)

Ieps = 1e-6  # relaxed a bit to speed up calculations

pointSource = np.zeros((n_points, nFreq, nDist), dtype=complex)

Path(output_path).parent.mkdir(parents=True, exist_ok=True)

# check for existing files to resume from
completed_dists = []
for iDist in range(nDist):
    temp_file = Path(f"{output_path}_dist{iDist}.nc")
    if temp_file.exists():
        try:
            temp_ds = xr.open_dataset(temp_file)
            temp_ds.close()
            completed_dists.append(iDist)
        except Exception as e:
            print(f"Found corrupted file for distance {distArray[iDist]:.2f}m, will regenerate: {e}")
        
for iDist in tqdm(range(nDist), desc="Processing distances"):
    dist = distArray[iDist]
    
    if iDist in completed_dists:
        print(f"Skipping already processed distance {dist:.2f}m (index {iDist})")
        continue
        
    for iFreq in tqdm(range(nFreq), desc=f"Frequencies for dist {iDist+1}/{nDist}", leave=False):
        k = 2 * np.pi * freqVec[iFreq] / c  # wavenumber
        
        theta_angles = angles[:, 0]  
        
        for i in range(n_points):
            try:
                with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
                    theta_angle = angles[i, 1] 
                    
                    ptot, _ = pointSourceScatSphere(k, head_size, dist, [theta_angle], Ieps)
                    ptot_value = np.nan_to_num(ptot.item(), nan=0.0, posinf=0.0, neginf=0.0)
                    
                    pointSource[i, iFreq, iDist] = ptot_value
                    
            except Exception as e:
                print(f"Error at freq={freqVec[iFreq]}Hz, dist={dist}m, point={i}: {e}")
                pointSource[i, iFreq, iDist] = 0

    source = np.zeros((1, 3))
    source[0, 0] = dist 

    dist_data = pointSource[:, :, iDist:iDist+1]  
    temp_ir = np.real(fft.ifft(dist_data, 2*nFreq, axis=1))
    temp_irs = np.transpose(temp_ir, (1, 0, 2))

    # intermediate dataset
    temp_ds = xr.Dataset(
        {
            "ir": (["time", "mic", "source"], temp_irs),
            "source_xyz": (["source", "xyz"], source),
            "mic_xyz": (["mic", "xyz"], points),
            "mic_angles": (["mic", "angle_type"], angles),
            "source_distance": (["source"], np.array([dist])),
            "relative_elevation": (["mic"], angles[:, 1]),  # Elevation angles
            "relative_azimuth": (["mic"], angles[:, 0]),    # Azimuth angles
        },
        attrs={
            "sampling_rate": fs,
            "temperature": T,
            "speed_of_sound": c,
            "head_size": head_size,
            "n_fibonacci_points": n_points,
            "distance": dist,
            "frequencies": freqVec.tolist(),
            "angle_types": ["azimuth", "elevation"],
        }
    )

    temp_ds.to_netcdf(f"{output_path}_dist{iDist}.nc", engine='netcdf4')
    completed_dists.append(iDist)  

# see if all dists are processed
if len(completed_dists) == nDist:
    print("all dists processed. combining into final set.")
    
    # Now create the combined dataset
    sources = np.zeros((nDist, 3))
    sources[:, 0] = distArray  
    
    mics = points  

    combined_irs = np.zeros((2*nFreq, n_points, nDist))
    
    for iDist in tqdm(range(nDist), desc="loading intermediate files"):
        temp_file = f"{output_path}_dist{iDist}.nc"
        temp_ds = xr.open_dataset(temp_file)
        # combined_irs[:, :, iDist] = temp_ds.ir.values
        combined_irs[:, :, iDist] = temp_ds.ir.values.squeeze(-1) 
        temp_ds.close()
    
    ds = xr.Dataset(
        {
            "ir": (["time", "mic", "source"], combined_irs),
            "source_xyz": (["source", "xyz"], sources),
            "mic_xyz": (["mic", "xyz"], mics),
            "mic_angles": (["mic", "angle_type"], angles),
            "relative_elevation": (["mic"], angles[:, 1]),
            "relative_azimuth": (["mic"], angles[:, 0]),  
        },
        attrs={
            "sampling_rate": fs,
            "temperature": T,
            "speed_of_sound": c,
            "head_size": head_size,
            "n_fibonacci_points": n_points,
            "distances": distArray.tolist(),
            "frequencies": freqVec.tolist(),
            "angle_types": ["azimuth", "elevation"],
        }
    )
    
    ds.to_netcdf(output_path, engine='netcdf4')
    print("combined set created,  generation complete")
else:
    print(f"Processing incomplete. {len(completed_dists)}/{nDist} distances processed.")    
