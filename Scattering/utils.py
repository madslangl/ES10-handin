import numpy as np 
import scipy as sc
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy import special 
#import scienceplots

"""
    Contains functions supporting data generation.
    Includes:
        - distance_between_points: calculates the distance between two points in 3D space.
        - speed_of_sound: calculates the speed of sound given the temperature.
        - planeWaveScatSphere: calculates the scattered field from a plane wave incident on a sphere.
        - pointSourceScatSphere: calculates the scattered field from a point source incident on a sphere.
        - fibonacci_points_on_sphere: distributes points on a sphere using the Fibonacci lattice method.

"""

def distance_between_points(p1, p2):
    """Calculate the Euclidean distance between two 3D points."""
    p1 = np.array(p1)
    p2 = np.array(p2)
    diff = p1-p2
    if diff.ndim == 1:
        return np.sqrt(np.sum(diff** 2))    
    else:
        return np.sqrt(np.sum(diff** 2, axis=1))

def speed_of_sound(T):
    """Speed of sound given the temperature."""
    return 331 * np.sqrt(1 + T / 273)

def planeWaveScatSphere(k,a,r,theta,Ieps):
    """
    Calculates the scattered field from a plane wave incident on a sphere.
    Uses Bessel functions. 
    """
    theta = np.array(theta).reshape(1, -1)
    r = np.array(r).reshape(-1, 1)
    
    ka = k * a
    kr = k * r
    n = 0
    ptot = np.zeros((len(r), len(theta[0])),dtype=complex)
    accuracy = []
    rel_accuracy = Ieps * 10
    
    while rel_accuracy > Ieps: 
        apri = (n * special.jv(n - 0.5, ka) - (n + 1) * special.jv(n + 1.5, ka)) / \
               (n * special.hankel1(n - 0.5, ka) - (n + 1) * special.hankel1(n + 1.5, ka))
        
        Pmatrix = special.legendre(n)(np.cos(theta[0]))
        P = Pmatrix # already 1 dim        
        
        factor = ((-1j) ** n) * (2 * n + 1) * np.sqrt(np.pi / (2 * kr))
        jn = special.jv(n + 0.5, kr)
        hn = special.hankel1(n + 0.5, kr)
        
        iter = np.outer(factor * (jn - apri * hn), P)
        
        ptot = ptot + iter
        
        # rel_accuracy = np.min(np.max(iter / ptot, axis=1))

        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = np.abs(iter) / np.abs(ptot)
            ratio = np.where(np.isfinite(ratio), ratio, 0)  # Replace NaN/Inf with 0
        rel_accuracy = np.min(np.max(ratio, axis=1))
        accuracy.append(rel_accuracy)        
        
        n += 1
    
    return ptot, accuracy

def pointSourceScatSphere(wavenumber,radius, distance ,theta,Ieps=1e-8):
    """
    Calculates the scattered field from a point source incident on a sphere.
    Uses Bessel functions. 
    """
    theta = np.array(theta).reshape(1, -1)
    distance = np.array(distance).reshape(-1, 1)
    
    ka = wavenumber * radius
    kr = wavenumber * distance
    n = 0
    ptot = np.zeros((len(distance), len(theta[0])), dtype=complex)
    accuracy = []  
    rel_accuracy = Ieps * 10
    
    while rel_accuracy > Ieps:
        hnR = np.sqrt(np.pi / (2 * kr)) * special.hankel1(n + 0.5, kr)
        
        hdA = np.sqrt(np.pi / (2 * ka)) * (special.hankel1(n - 0.5, ka) - 
                                         (special.hankel1(n + 0.5, ka) + 
                                          ka * special.hankel1(n + 1.5, ka)) / ka)
        
        Pmatrix = special.legendre(n)(np.cos(theta[0]))
        P = Pmatrix # already 1 dim

        factor = n + 0.5
        #Ps = 8./(2*pi*a.^2) * (factor.*1)*P.*hnR./hdA * exp(-1i*pi/2);
        Ps = (8 / (2 * np.pi * radius**2)) * factor * np.outer(hnR.flatten(), P) / hdA * np.exp(-1j * np.pi / 2)
        #P0 = k/(pi*r).* exp(1i*(-kr + pi/2));
        P0 = wavenumber / (np.pi * distance) * np.exp(1j * (-kr + np.pi / 2))
        
        iter = Ps / P0
        ptot = ptot + iter
    
        # rel_accuracy = np.min(np.max(np.abs(iter) / np.abs(ptot)))

        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = np.abs(iter) / np.abs(ptot)
            ratio = np.where(np.isfinite(ratio), ratio, 0)  # Replace NaN/Inf with 0
        rel_accuracy = np.min(np.max(ratio, axis=1))
        accuracy.append(rel_accuracy)
        
        n += 1
    return ptot, accuracy

def fibonacci_points_on_sphere(n_points, radius=0.0875, degrees=False):
    """
    Generate points on a sphere using the Fibonacci lattice method.

    Parameters:
    num_points (int): The number of points to generate on the sphere.
    radius (float): The radius of the sphere. Default is 0.0875 m (for a head in meters).
    degrees (bool): If True, angles are returned in degrees. Default is False.

    Returns:
    points = np.ndarray: An array of shape (num_points, 3) containing the x, y, z coordinates of the points.
    angles = np.ndarray: An array of shape (num_points, 2) containing the azimuth and elevation angles of the points (degrees).
    """

    points = np.zeros((n_points, 3))
    angles = np.zeros((n_points, 2)) # for [azimuth, elevation]
    golden_ratio = (1 + np.sqrt(5)) / 2
    golden_angle = 2 * np.pi * (1 - 1 / golden_ratio)  # in radians. approx 137.5 in degs.

    for i in range(n_points):
        theta = golden_angle * i
        z = 1 - (2 * i) / (n_points - 1)  # Map i to [-1, 1]
        radius = np.sqrt(1 - z * z)

        x = radius * np.cos(theta)
        y = radius * np.sin(theta)

        points[i] = [x, y, z]

        theta = np.arctan2(y, x)  # azimuth 
        phi = np.arccos(z)        # elevation
        
        if degrees:
            theta = np.degrees(theta)
            phi = np.degrees(phi)

        angles[i] = [theta, phi]

    return points, angles


def dataset_overview(data):
    print("=== DATASET OVERVIEW ===")
    print(data)

    print("\n=== DIMENSIONS ===")
    for dim_name, dim_size in data.dims.items():
        print(f"{dim_name}: {dim_size}")

    print("\n=== VARIABLES ===")
    for var_name, var in data.variables.items():
        print(f"{var_name}: {var.shape} {var.dtype}")

    print("\n=== ATTRIBUTES ===")
    for attr_name, attr_value in data.attrs.items():
        if isinstance(attr_value, list) and len(attr_value) > 10:
            print(f"{attr_name}: List with {len(attr_value)} items")
        else:
            print(f"{attr_name}: {attr_value}")

    print("\n=== SOURCE POSITIONS ===")
    print(data.source_xyz.values)
    print("\n len source_xyz:", len(data.source_xyz.values))

    print("\n=== MIC SAMPLES (first 5) ===")
    print("Positions (xyz):")
    print(data.mic_xyz.values[:5])
    print("\nAngles (radians):")
    print(data.mic_angles.values[:5])

    plt.figure(figsize=(6.3*1.9, 3.5*1.6))
    ax = plt.subplot(111, projection='3d')
    # ax.grid()
    sources = data.source_xyz.values
    ax.scatter(sources[:, 0], sources[:, 1], sources[:, 2], 
            color='red', s=50, label='Sources')

    mics = data.mic_xyz.values
    if len(mics) > 100:
        sample_idx = np.linspace(0, len(mics)-1, 100, dtype=int)
        mics_sample = mics[sample_idx]
        ax.scatter(mics_sample[:, 0], mics_sample[:, 1], mics_sample[:, 2], 
                color='blue', s=20, alpha=0.5, label='Microphones (samples)')
    else:
        ax.scatter(mics[:, 0], mics[:, 1], mics[:, 2], 
                color='blue', s=20, alpha=0.5, label='Microphones (samples)')

    head_size = data.attrs['head_size']
    u = np.linspace(0, 2*np.pi, 20)
    v = np.linspace(0, np.pi, 20)
    x = head_size * np.outer(np.cos(u), np.sin(v))
    y = head_size * np.outer(np.sin(u), np.sin(v))
    z = head_size * np.outer(np.ones(np.size(u)), np.cos(v))

    ax.plot_surface(x, y, z, color='gray', alpha=0.2)
    ax.set_xlabel('X (m)', fontsize=9)
    ax.set_ylabel('Y (m)', fontsize=9)
    ax.set_zlabel('Z (m)', fontsize=9)
    ax.set_xlim(-.5, .5)
    ax.set_ylim(-.5, .5)
    ax.set_zlim(-.5, .5)
    # ax.set_title('Positions of Sources and Microphones')
    ax.legend(fontsize=12)
    ax.view_init(elev=20, azim=35)
    ax.set_box_aspect([1, 1, 1])
    # plt.tight_layout()
    # plt.savefig(r"ES10\automatic-rir-generation\source_mic_positions-valid.pdf")
    # plt.show()


def plot_one_signal(data,output_path):
    import scienceplots
    from scipy import fft
    plt.style.use('science')

    # Choose a specific mic point and source
    mic_idx = 500  # Just picking a point in the middle of the array
    source_idx = 15  # Roughly in the middle of the distance array

    # Get the IR for that specific point
    ir = data.ir.values[:, mic_idx, source_idx]

    # Print information about this point
    mic_position = data.mic_xyz.values[mic_idx]
    mic_angle_rad = data.mic_angles.values[mic_idx]
    mic_angle_deg = np.rad2deg(mic_angle_rad)
    source_position = data.source_xyz.values[source_idx]
    source_distance = np.linalg.norm(source_position)

    print(f"Microphone #{mic_idx}:")
    print(f"  Position (xyz): {mic_position}")
    print(f"  Angle (degrees): Azimuth={mic_angle_deg[0]:.1f}°, Elevation={mic_angle_deg[1]:.1f}°")
    print(f"Source #{source_idx}:")
    print(f"  Position (xyz): {source_position}")
    print(f"  Distance: {source_distance:.2f}m")

    # calculate the FFT
    fs = float(data.attrs["sampling_rate"])
    n = len(ir)
    ir_fft = fft.fft(ir)
    magnitude = np.abs(ir_fft[:n//2+1])
    magnitude_db = 20 * np.log10(magnitude + 1e-10)
    frequency = np.arange(n//2+1) * fs / n

    # plot frequency response
    plt.figure(figsize=(10, 6))
    plt.semilogx(frequency, magnitude_db)
    plt.title("Frequency Response of simulated IR. Distance = 1.13 m, Azimuth = -6.1°")
    # plt.title(f'Frequency Response for Mic #{mic_idx}, Source #{source_idx}\n' +
            #   f'(Azimuth={mic_angle_deg[0]:.1f}°, Elevation={mic_angle_deg[1]:.1f}°, Distance={source_distance:.2f}m)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (dB)')
    plt.grid(True, which="both", ls="-")
    plt.axvline(20, color='r', linestyle='--', alpha=0.3)
    plt.axvline(20000, color='r', linestyle='--', alpha=0.3)
    plt.xlim(20, fs/2)
    plt.ylim(-35, 1)
    plt.tight_layout()
    plt.savefig(rf"{output_path}\frequency_response_actual.pdf")