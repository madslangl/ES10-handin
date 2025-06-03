import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from scipy import fft
import colorsys
import scienceplots
from sklearn.metrics import mean_squared_error
import pandas as pd

# plt.style.use('science')

###########################################################################
## PLEASE NOTE:
# I tried to optimise this script, which broke it unfortunately.. A fix is needed, but the interpolator should still work.
###########################################################################

from interpolator import FibonacciSphere

dataset_path = None 
verification_data = None

# please set correct path here. The scattering dataset can be downloaded from: 
# https://www.dropbox.com/scl/fo/ewmh94bew74lqy02p1fb0/AFNBsgEYmQkNmN37IpJuSaY?rlkey=neqra7zr25imcv43lg043ksqd&st=tzwd9lar&dl=0
dataset = xr.open_dataset(dataset_path)

interpolator = FibonacciSphere(dataset)

new_source_position = np.array([1,0,0])
head_el = 0
head_az = 0

left_ir, right_ir = interpolator.interpolate(new_source_position, head_az, head_el, method='phase_aligned', k_neighbors=5)

fs = float(dataset.sampling_rate)

def calculate_frequency_response(ir, fs):
    n = len(ir)
    ir_fft = fft.fft(ir)
    magnitude = np.abs(ir_fft[:n//2+1])
    magnitude_db = 20 * np.log10(magnitude + 1e-10) 
    frequency = np.arange(n//2+1) * fs / n
    return frequency, magnitude_db

left_freq, left_magnitude = calculate_frequency_response(left_ir, fs)
right_freq, right_magnitude = calculate_frequency_response(right_ir, fs)

head_size=0.0875

verification_data = xr.open_dataset(verification_data, engine='netcdf4')

def compare_with_verification_dataset_right_ear(source_idx=0, head_az=0, head_el=0, color=None):
    """Compare interpolated responses to verification dataset for right ear"""
    # Get source position from verification dataset
    source_pos = verification_data.source_xyz.values[source_idx]
    source_dist = np.round(np.linalg.norm(source_pos), 2)
    # Add these debug lines to your compare_with_verification_dataset_right_ear function
    print(f"Verification source: {source_pos}")
    source_distances = np.array([np.linalg.norm(source_pos - s) for s in interpolator.source_positions])
    nearest_source_idx = np.argmin(source_distances)
    nearest_source = interpolator.source_positions[nearest_source_idx]
    print(f"Nearest interpolator source: {nearest_source}, dist: {source_distances[nearest_source_idx]:.4f}m")
    
    # interpolated impulse responses 
    left_ir_k3, right_ir_k3 = interpolator.interpolate(
        source_pos, head_az, head_el, k_neighbors=1
    )
    
    # mic pos from verificaiton dataset
    verification_mic_positions = verification_data.mic_xyz.values
    
    # calculate ear positions based on head orientation
    right_ear_local = np.array([0, -head_size, 0])  # Right ear at -Y axis
    left_ear_local = np.array([0, head_size, 0]) 
    
    # rotation matrices for head azimuth and elevation
    azimuth = np.deg2rad(head_az)
    elevation = np.deg2rad(head_el)
    rot_matrix_y = np.array([
        [np.cos(elevation), 0, np.sin(elevation)],
        [0, 1, 0],
        [-np.sin(elevation), 0, np.cos(elevation)]
    ])
    rot_matrix_z = np.array([
        [np.cos(azimuth), -np.sin(azimuth), 0],
        [np.sin(azimuth), np.cos(azimuth), 0],
        [0, 0, 1]
    ])

    # combined_rotation = rot_matrix_y @ rot_matrix_z
    combined_rotation = rot_matrix_y @ rot_matrix_z  
    right_ear_pos = combined_rotation @ right_ear_local
    
    # closest mics in verification dataset to ear positions
    az_rotation = np.array([
        [np.cos(azimuth), -np.sin(azimuth), 0],
        [np.sin(azimuth), np.cos(azimuth), 0],
        [0, 0, 1]
    ])
    
    # apply rotation only around z-axis for verification dataset lookup
    right_ear_local_rotated =  right_ear_pos
    print(f"Rotated ear position for verification lookup: {right_ear_local_rotated}")

    # find closest mics using the azimuth-rotated positioncloses
    # right_distances = np.array([np.linalg.norm(right_ear_local_rotated - mic_pos) for mic_pos in verification_mic_positions])
    right_distances = np.array([np.linalg.norm(right_ear_pos - mic_pos) 
                           for mic_pos in verification_mic_positions])
    verif_right_mic_idx = np.argmin(right_distances)

    right_verif_ir = verification_data.ir.values[:, verif_right_mic_idx, source_idx]

    # use same position for taking the point we want in the interpolatr
    closest_mic_idx = np.argmin([np.linalg.norm(right_ear_pos - mic_pos) for mic_pos in interpolator.mic_positions])
    closest_source_idx = np.argmin([np.linalg.norm(source_pos - s) for s in interpolator.source_positions])
    direct_ir = interpolator.precomputed_irs[:, closest_mic_idx, closest_source_idx]
    
    #Calculate frequency responses
    freq, right_verif_mag = calculate_frequency_response(right_verif_ir, fs)
    freq, right_k3_mag = calculate_frequency_response(right_ir_k3, fs)
    
    freq, direct_mag = calculate_frequency_response(direct_ir, fs)
    direct_mag = direct_mag - np.max(direct_mag)

    # normalise
    direct_mag = direct_mag - np.max(direct_mag)
    right_verif_mag = right_verif_mag - np.max(right_verif_mag)
    right_k3_mag = right_k3_mag - np.max(right_k3_mag)

    print(f"Right ear calculated position: {right_ear_pos}")
    print(f"Selected reference mic position: {verification_mic_positions[verif_right_mic_idx]}")
    print(f"Distance from right ear to selected mic: {right_distances[verif_right_mic_idx]:.4f}m")
    plt.semilogx(freq, right_verif_mag, '-.', color=color, linewidth=2, label=f'Reference, angle={head_az}°')
    plt.semilogx(freq, right_k3_mag, '--', color=color, linewidth=1.5, label=f'Interpolated, angle={head_az}°')


plot_values = np.linspace(45,90+45,5)

plt.figure(figsize=(8.5, 4.2))

colors = ["#FF0000", '#ff7f0e', "#BB22C9FF", "#00719e", "#000000"]
from matplotlib.lines import Line2D
legend_elements = []

for i, head_az in enumerate(plot_values):
    head_el = 0
    compare_with_verification_dataset_right_ear(source_idx=0, head_az=head_az, head_el=head_el, color=colors[i])
    legend_elements.append(Line2D([0], [0], color=colors[i], lw=2,ls="-.",
                                 label=f'Reference, angle={head_az}°'))
    legend_elements.append(Line2D([0], [0], color=colors[i], lw=2, ls="--", 
                                 label=f'Interpolated, angle={head_az}°'))

plt.title('Reference vs Interpolated (Right Ear)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude (dB)')
plt.grid(True, alpha=0.3)
plt.xlim(20, fs/2-100)
plt.ylim(-10, 0)
plt.legend(handles=legend_elements)
plt.show()


# # Uncomment to plot the mic positions and calculated ear position
# plt.figure(figsize=(10, 10))
# ax = plt.subplot(111, projection='3d')

# # Plot verification mics
# ax.scatter(verification_mic_positions[:, 0], 
#           verification_mic_positions[:, 1], 
#           verification_mic_positions[:, 2], 
#           c='blue', marker='o', label='Verification Mics')

# # Plot interpolator mics
# ax.scatter(interpolator.mic_positions[:, 0], 
#           interpolator.mic_positions[:, 1], 
#           interpolator.mic_positions[:, 2], 
#           c='red', marker='.', alpha=0.3, label='Interpolator Mics')

# # Plot calculated ear position
# ax.scatter([right_ear_pos[0]], [right_ear_pos[1]], [right_ear_pos[2]], 
#           c='green', marker='x', s=100, label='Calculated Ear')

# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# ax.legend()
# plt.show()