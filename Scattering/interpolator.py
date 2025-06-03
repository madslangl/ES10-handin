import numpy as np
from xarray import Dataset
import xarray as xr
from abc import ABC, abstractmethod

import matplotlib.pyplot as plt


"""Framework for testing interpolating RIRs using spherical harmonics and analytic model for rigid sphere.

# Uses Bessel functions for generating data. Can be found in:
# ES10\automatic-rir-generation\generate_scattering.py

"""

class Interpolator(ABC):
    def __init__(self, drtf: Dataset, **options):
        self.drtf = drtf

    @abstractmethod
    def interpolate(self, source_position, head_azimuth, head_elevation, *args, **kwargs):
        pass


class FibonacciSphere(Interpolator):
    """Interpolates IRs using angular interpolation from a Fibonacci sphere dataset.
    Inputs:
        - drtf: Dataset containing the precomputed IRs and mic positions.
        - radius: Radius of the sphere (default: 0.0875 m).
        - options: Additional options for the interpolator.
    Outputs:
        - Interpolated IRs for the new source and mic positions.
    """
    
    def __init__(self, drtf: Dataset, radius: float=0.0875, **options):
        super().__init__(drtf, **options)
        self.radius = radius
        self.source_positions = self.drtf.source_xyz.values
        self.mic_positions = self.drtf.mic_xyz.values
        self.fs = float(self.drtf.sampling_rate)
        # self.mic_angles = self.drtf.mic_angles.values  # [mic, angle_type] array from Fibonacci dataset
        
        if hasattr(self.drtf, 'mic_angles'):
            self.mic_angles = self.drtf.mic_angles.values
        else:
            print("Dataset doesn't have mic_angles, calculating from positions...")
            self.mic_angles = np.zeros((len(self.mic_positions), 2))
            for i, mic_pos in enumerate(self.mic_positions):
                norm = np.linalg.norm(mic_pos)
                unit_vec = mic_pos / norm if norm > 0 else np.array([0, 0, 1])
                _, az, el = self.cart_to_sphere(unit_vec)
                self.mic_angles[i, 0] = az  
                self.mic_angles[i, 1] = el  
            print(f"Calculated {len(self.mic_angles)} mic angles from positions")
            

        self.precomputed_irs = self.drtf.ir.values     # [time, mic, source]
        self.angle_types = ["azimuth", "elevation"]
            
        self.angles_in_radians = True

        if np.any(np.abs(self.mic_angles) > 2*np.pi):
            self.angles_in_radians = False
            self.mic_angles_rad = np.deg2rad(self.mic_angles)
        else:
            self.mic_angles_rad = self.mic_angles
        
    
    def cart_to_sphere(self, xyz):
        """Convert from Cartesian to spherical coordinates (r, theta, phi).
        Allows for taking mic position given in cartesian coordinates and converting them to spherical coordinates.
        Args:
            xyz: Cartesian coordinates [x, y, z]
        
        Returns:
            Spherical coordinates [r, theta, phi]
        """
        x, y, z = xyz
        r = np.sqrt(x**2 + y**2 + z**2)
        theta = np.arctan2(y, x)             # azimuth angle [0, 2π]
        phi = np.arccos(z / r if r != 0 else 0)  # elevation angle [0, π]
        return np.array([r, theta, phi])
    
    
    
    def find_nearest_mics_combined(self, target_angle, k, alpha=0.7):
        """Find k nearest mic positions by combining angular and Euclidean distance.
        
        Args:
            target_angle: Target angle [azimuth, elevation]
            k: Number of nearest neighbors to find
            alpha: Weight for angular distance vs Euclidean (0-1)
                Higher alpha gives more importance to angular distance
                
        Returns:
            nearest_indices: Indices of the k nearest mic positions
            weights: Weights for the k nearest mic positions
        """
        target_az, target_el = target_angle
        
        # angular dist on unit sphere
        az_diffs = np.abs(self.mic_angles_rad[:, 0] - target_az)
        
        # account for periodicity in azimuth (no dublicates)
        az_diffs = np.minimum(az_diffs, 2*np.pi - az_diffs)
        el_diffs = np.abs(self.mic_angles_rad[:, 1] - target_el)
        
        angular_distances = np.sqrt(az_diffs**2 + el_diffs**2)
        
        # target angle to cartesian position (on unit sphere)
        x = np.cos(target_az) * np.sin(target_el)
        y = np.sin(target_az) * np.sin(target_el)
        z = np.cos(target_el)
        target_pos = np.array([x, y, z]) * self.radius
        
        # euclidean distances to all mic positions
        euclidean_distances = np.array([np.linalg.norm(target_pos - mic_pos) for mic_pos in self.mic_positions])
        
        norm_angular = angular_distances / np.max(angular_distances) if np.max(angular_distances) > 0 else angular_distances
        norm_euclidean = euclidean_distances / np.max(euclidean_distances) if np.max(euclidean_distances) > 0 else euclidean_distances
        
        # combine distances with alpha weigthing, angular = alpha
        combined_distances = alpha * norm_angular + (1 - alpha) * norm_euclidean
        
        # uses knn
        nearest_indices = np.argsort(combined_distances)[:k] # uses quicksort by default.
        nearest_distances = combined_distances[nearest_indices]
        
        # calculating weights of the nearest mics. doesnt really make sense when k=1.
        weights = 1 / (nearest_distances + 1e-6)
        weights = weights / np.sum(weights)
        
        return nearest_indices, weights

    def _phase_aligned_average(self, irs, weights):
        """Average IRs with phase alignment in frequency domain
        Uses the first IR as ref. and aligns the phase of all k IRs to it. (So they all have the same phase response)
        Not really working well."""
        n = len(irs[0])
        reference_ir = irs[0]  # Use the first IR as reference
        
        ref_fft = np.fft.rfft(reference_ir)
        # ref_mag = np.abs(ref_fft)
        ref_phase = np.angle(ref_fft)
        
        # Initialize weighted average in frequency domain
        avg_fft = np.zeros_like(ref_fft, dtype=complex)
        
        for i, ir in enumerate(irs):
            ir_fft = np.fft.rfft(ir)
            ir_mag = np.abs(ir_fft)
            
            # Use magnitude of each IR 
            aligned_fft = ir_mag * np.exp(1j * ref_phase)
            
            # Add weighted contribution
            avg_fft += weights[i] * aligned_fft
        
        # Convert back to time domain
        result_ir = np.fft.irfft(avg_fft, n)
        
        return result_ir

    

    def interpolate(self, source_position, head_azimuth, head_elevation, method='phase_aligned', k_neighbors=1, k_sources=1, head_size=0.0875):
        """
        Get interpolated RIRs for both ears based on source position and head orientation.
        
        Args:
            source_position: Source position [x, y, z]
            head_azimuth: Head azimuth angle in degrees (0 = facing forward)
            head_elevation: Head elevation angle in degrees (0 = level)
            method (str): regular or phase_aligned 
            k_neighbors: Number of nearest neighbors for interpolation
            k_sources: Number of nearest sources for interpolation
            head_size: Distance from head center to ears in meters
        """

        azimuth = np.deg2rad(head_azimuth)
        elevation = np.deg2rad(head_elevation)

        left_ear_local = np.array([0, head_size, 0]) 
        right_ear_local = np.array([0, -head_size, 0])

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

        # for rotation the head after placement.
        combined_rotation = rot_matrix_y @ rot_matrix_z
        # combined_rotation = rot_matrix_z @ rot_matrix_y
        left_ear_pos = combined_rotation @ left_ear_local
        right_ear_pos = combined_rotation @ right_ear_local

        _, left_az, left_el = self.cart_to_sphere(left_ear_pos)
        _, right_az, right_el = self.cart_to_sphere(right_ear_pos)

        source_distances = np.array([np.linalg.norm(source_position - s) for s in self.source_positions])
        nearest_source_indices = np.argsort(source_distances)[:k_sources]
        
        source_dists = source_distances[nearest_source_indices]
        
        # small epsilon
        source_weights = 1 / (source_dists + 1e-6)
        source_weights = source_weights / np.sum(source_weights)
        
        left_ear_rir = np.zeros_like(self.precomputed_irs[:, 0, 0])
        right_ear_rir = np.zeros_like(self.precomputed_irs[:, 0, 0])
        
        if method == 'phase_aligned' and k_neighbors > 1:
            for src_idx, src_weight in zip(nearest_source_indices, source_weights):
                # finds k nearest mics 
                left_indices, left_ang_weights = self.find_nearest_mics_combined([left_az, left_el], k_neighbors)
                right_indices, right_ang_weights = self.find_nearest_mics_combined([right_az, right_el], k_neighbors)
                
                # takes the irs of both
                left_irs = [self.precomputed_irs[:, idx, src_idx] for idx in left_indices]
                right_irs = [self.precomputed_irs[:, idx, src_idx] for idx in right_indices]
                
                # aligns the phase for all the irs and then does average
                left_ear_src_contribution = self._phase_aligned_average(left_irs, left_ang_weights)
                right_ear_src_contribution = self._phase_aligned_average(right_irs, right_ang_weights)
                
                # weighted source contribution
                left_ear_rir += src_weight * left_ear_src_contribution
                right_ear_rir += src_weight * right_ear_src_contribution
        
        else: # original weighting method: 
            for src_idx, src_weight in zip(nearest_source_indices, source_weights):
                # nearest mics for left ear and right ear
                left_indices, left_ang_weights = self.find_nearest_mics_combined([left_az, left_el], k_neighbors)
                right_indices, right_ang_weights = self.find_nearest_mics_combined([right_az, right_el], k_neighbors)
                
                # adding weighted contribution from this source position (regular method)
                for i, (idx, ang_weight) in enumerate(zip(left_indices, left_ang_weights)):
                    left_ear_rir += src_weight * ang_weight * self.precomputed_irs[:, idx, src_idx]
                    
                for i, (idx, ang_weight) in enumerate(zip(right_indices, right_ang_weights)):
                    right_ear_rir += src_weight * ang_weight * self.precomputed_irs[:, idx, src_idx]
        
        return left_ear_rir, right_ear_rir

