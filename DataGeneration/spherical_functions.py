import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# import scienceplots
# plt.style.use('science')

def generate_spherical_head(sphere_center, radius, azimuth=0,elevation=0):
    """ Generates a empty sphere with microphones placed on the surface (approximately where the ears are) 
    and a source in the front (approximately where the mouth is).

    Arguments:
    sphere_center: [x,y,z] coordinates of the center of the sphere
    radius: radius of the sphere
    azimuth: azimuthal rotation of the head around the z-axis. [degrees] - default = 0
    elevation : elevation of the mouth position. [degrees] - default = 0. ears are placed perfectly on side of sphere, so are assumed independent of elevation"""


    mic_positions = []

    # Ear positions (right=90°, left=270° in theta)
    theta_values_ears = [90+azimuth, 270+azimuth]
    phi_values_ears = [90, 90] # elevation not added since ears are modelled to be perfectly place on the side of the sphere.

    for theta, phi in zip(theta_values_ears, phi_values_ears):
        x = sphere_center[0] + radius * np.sin(np.deg2rad(phi)) * np.cos(np.deg2rad(theta))
        y = sphere_center[1] + radius * np.sin(np.deg2rad(phi)) * np.sin(np.deg2rad(theta))
        z = sphere_center[2] + radius * np.cos(np.deg2rad(phi))
        mic_positions.append([x, y, z])

    mic_positions = np.array(mic_positions).T

    theta_value_mouth = 180+azimuth  # Front of head (degrees)
    phi_value_mouth = 200 + elevation  # Bottom of head (degrees)

    # conv to cartiesain    
    mouth_x = sphere_center[0] + radius * np.sin(np.deg2rad(phi_value_mouth)) * np.cos(np.deg2rad(theta_value_mouth))
    mouth_y = sphere_center[1] + radius * np.sin(np.deg2rad(phi_value_mouth)) * np.sin(np.deg2rad(theta_value_mouth))
    mouth_z = sphere_center[2] + radius * np.cos(np.deg2rad(phi_value_mouth))
    mouth_position = np.array([mouth_x, mouth_y, mouth_z])

    return mic_positions, mouth_position


class Head:
    def __init__(self, center_position, radius=0.0875, azimuth=0, elevation=0):
        """ Creates spherical head model with microphones on the side, and a source in the front.
        Makes it easier to call the head in the data generation by Head.LeftEar etc. 
        Args: center_position: [x,y,z] coordinates of the center of the head 
              radius: radius of the head (default: 0.08 m)
              azimuth: azimuthal rotation of the head around the z-axis. [degrees] - default = 0
              elevation : elevation of the mouth position. [degrees] - default = 0. ears are placed perfectly on side of sphere, so are assumed independent of elevation"""
        
        
        
        self.center_position = np.array(center_position)
        self.radius = radius
        self.azimuth = azimuth
        self.elevation = elevation
        self._ears, self._mouth = generate_spherical_head(center_position, radius,azimuth,elevation)

    @property
    def leftEar(self):
        """Left Ear position as x,y,z"""
        return self._ears[:, 1]
    @property
    def rightEar(self):
        """Right Ear position as x,y,z"""
        return self._ears[:, 0]
    @property
    def mouth(self):
        """Mouth position as x,y,z"""
        return self._mouth
        
    def place_at(self, new_position):
        """Move the head to a new center position"""
        offset = np.array(new_position) - self.center_position
        self.center_position = np.array(new_position)
        
        # Update positions
        self._ears = self._ears + offset[:, np.newaxis]  # Add offset to each column
        self._mouth = self._mouth + offset
        return self