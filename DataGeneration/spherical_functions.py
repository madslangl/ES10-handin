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
    phi_values_ears = [90+elevation, 90+elevation] 

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


def plot_spherical_head(azimuth):
    sphere_center = [0, 0, 0]
    radius = 1  # Increased for better visibility
    mic_positions, mouth_position = generate_spherical_head(sphere_center, radius, azimuth=azimuth)
    
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the sphere
    u, v = np.mgrid[0:2*np.pi:30j, 0:np.pi:20j]
    x = sphere_center[0] + radius * np.sin(v) * np.cos(u)
    y = sphere_center[1] + radius * np.sin(v) * np.sin(u)
    z = sphere_center[2] + radius * np.cos(v)
    ax.plot_wireframe(x, y, z, color='k', alpha=0.4)
    
    # Plot ears (microphones)
    ax.scatter(mic_positions[0][0], mic_positions[1][0], mic_positions[2][0], 
               c='b', marker='o', s=100, label='Right ear')
    ax.scatter(mic_positions[0][1], mic_positions[1][1], mic_positions[2][1], 
               c='g', marker='o', s=100, label='Left ear')
    
    # Plot mouth
    ax.scatter(mouth_position[0], mouth_position[1], mouth_position[2], 
               c='r', marker='x', linewidths=3, s=100, label='Mouth')
    
    ax.set_box_aspect([1, 1, 1])
    
    # Explicitly set axis labels with larger padding
    ax.set_xlabel('x', fontsize=11)
    ax.set_ylabel('y' , fontsize=11)
    
    # Fix z-label with extreme padding
    ax.set_zlabel('z', fontsize=11, labelpad=20)
    
    # Adjust legend to be on one horizontal line (3 columns)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 0.85),
              ncol=3, fontsize=12, frameon=True)
    
    # View angle - adjust for better z-axis visibility
    ax.view_init(elev=15, azim=135)
    
    # Move the z-axis label more to the side
    ax._get_zlabel_position = lambda: (-5, -5, -5)
    ax._set_zlabel_position = lambda pos: None
    
    # Give more space for the plot
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.9)
    
    # Save with tight layout and padding
    plt.savefig(f"spherical_head_azimuth-{azimuth}.pdf", bbox_inches='tight', pad_inches=0.3, dpi=300)
    
    # plt.show()

# for azimuth in [0, 45, 90]:
#     plot_spherical_head(azimuth=azimuth)

