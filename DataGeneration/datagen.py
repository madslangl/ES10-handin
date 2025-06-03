# directivity is not available using raytracing, only ISM.
# Using pra.ShoeBox forces ISM
# TODO: look into modifying engine to combine ism for early reflections + directivity and raytracing after that (no directivity).

import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from tqdm import tqdm
from pathlib import Path
import string
import h5py
import pyroomacoustics as pra
from pyroomacoustics.directivities import DirectionVector, Cardioid

from spherical_functions import Head


TESTING = True
PLOT_EVERY_ROTATION = False
EXAMPLE_DATASET = True # for generating a very small datast to check integrity.

fs = 16000



# more on directivities here:
# https://pyroomacoustics.readthedocs.io/en/pypi-release/pyroomacoustics.directivities.html

# number of positions to sample in each dimension pr. room (unique instances of placing source and receiver)
x_positions = 2
y_positions = 2
z_positions = 2


parameter_dict = {
    'azimuth': np.linspace(-90, 90, 5),  
    'elevation': np.linspace(-20, 20, 3), # usually, 90 is straight forward, but this is converted later in the code, so 0 elevation = looking straight forward 
    'absorption': [0.01, 0.1, 0.2, 0.3],
    'height': np.linspace(2, 6, 2),     
    'length': np.linspace(2, 15, 5),    
    'width': np.linspace(2, 6, 3),      
    'laptop_x_offset': np.linspace(0.3, 0.6, 3), 
    'laptop_y_offset': np.linspace(-.3, 0.3, 3), 
    'laptop_z_offset': np.linspace(-.3, 0, 2),   
}

if TESTING: 
    output_dir = r"DataGeneration\example_data"
else: 
    output_dir = None # this makes a lot of data. please set path accordingly if you want to run the full gen.

if EXAMPLE_DATASET:
    total_count = 0
    max_count = 20 # synthesise 20 RIRs in a room. mainly implemented to test functionality and data structure.


output_dir = Path("DataGeneration") / "example_data"
hdf5_path = output_dir / "synthesisedRIRs.h5"

def generate_room_id(index): # naming scheme for rooms: AA, AB, AC etc..
    first_letter = string.ascii_uppercase[index // 26]
    second_letter = string.ascii_uppercase[index % 26]
    return f"{first_letter}{second_letter}"

def rt60_sabine(absorption, room_width, room_depth, room_height, c=343.0):
    """Calculate T60. Used for finding required max order of reflections via inverse sabine"""
    return 0.161 * (room_width*room_depth*room_height) / (absorption * c)
    
def generateRoom(length,width,height,absorption,fs=fs):
    print("\n dims: L:",length, "W:", width, "H:", height, "abs:", absorption)
    rt60_calc = rt60_sabine(absorption, length, width, height)
    
    dimensions = f"{length}x{width}x{height}"

    volume = length * width * height
    
    # Case 1: Very small rooms (under 10 cubic meters). Sets the max_order low, as modelling a lot of reflections in a very absorbant room makes little sense.
    if volume < 10:
        print(f"Small room detected ({dimensions}). Using direct absorption.")
        max_order = 5

    # Case 2: Very large rooms with low absorption
    elif (absorption <= 0.01 and length >= 10 and width >= 6) or \
         (absorption <= 0.05 and length >= 15):
        print(f"Large low-absorption room detected. Using direct absorption.")
        max_order = 5
    
    # Case 3: Normal rooms - inverse_sabine but with error handling
    else:
        try:
            if absorption < 0.45:
                absorption, max_order = pra.inverse_sabine(rt60=rt60_calc, room_dim=[length, width, height])
            else:
                max_order = 5
        except ValueError:
            print(f"inverse_sabine failed for {dimensions}. Using direct values.")
            max_order = 5
    
    max_order = max(3, min(50, max_order))

    if TESTING:
        print("max_order of reflections: ", max_order)
    
    room = pra.ShoeBox([length, width, height], fs=fs, absorption=absorption, max_order=max_order)
    return room, max_order

file_exists = hdf5_path.is_file()

with h5py.File(hdf5_path, 'a') as hdf:
    # Get existing room IDs if file exists
    existing_rooms = set(hdf.keys()) if file_exists else set()
    print(f"Found {len(existing_rooms)} existing rooms in HDF5 file")
    
    room_index = 0  # for room ID gen

    for absorption, height, length, width in product(parameter_dict['absorption'],parameter_dict['height'],parameter_dict['length'],parameter_dict['width']):
        room_id = generate_room_id(room_index)
        room_index += 1
        
        # Skip room if it already exists
        if room_id in existing_rooms:
            print(f"Room {room_id} already exists, skipping...")
            continue
        
        room_group = hdf.create_group(room_id)
        
        dimensions = f"{length}x{width}x{height}"        

        # following is used to find and implement safe margins for the room dimensions such that mic and source positions are always within the room.
        max_x_offset = max(abs(np.min(parameter_dict['laptop_x_offset'])), 
                        abs(np.max(parameter_dict['laptop_x_offset'])))

        max_y_offset = max(abs(np.min(parameter_dict['laptop_y_offset'])), 
                        abs(np.max(parameter_dict['laptop_y_offset'])))

        max_z_offset = max(abs(np.min(parameter_dict['laptop_z_offset'])), 
                        abs(np.max(parameter_dict['laptop_z_offset'])))

        base_margin = 0.5  # Base margin from walls. so the receiver is not all of a sudden in the wall.
        x_margin = base_margin + max_x_offset + 0.05  
        y_margin = base_margin + max_y_offset + 0.05
        z_margin = base_margin + max_z_offset + 0.05

        x_min = x_margin
        x_max = length - x_margin
        y_min = y_margin  
        y_max = width - y_margin
        z_min = z_margin
        z_max = height - z_margin

        x_vals = np.linspace(x_min, x_max, x_positions)
        y_vals = np.linspace(y_min, y_max, y_positions)
        z_vals = np.linspace(z_min, z_max, z_positions)
        
        plotted_positions = []        
        rir_counter = 1
        
        for x in tqdm(x_vals):
            for y in y_vals:
                for z in z_vals:
                    for azimuth in parameter_dict['azimuth']:
                        for elevation in parameter_dict['elevation']:
                            for laptop_x_offset in parameter_dict['laptop_x_offset']:
                                for laptop_y_offset in parameter_dict['laptop_y_offset']:
                                    for laptop_z_offset in parameter_dict['laptop_z_offset']:
                                        if EXAMPLE_DATASET:
                                            if total_count >= max_count:
                                                break
                                            total_count += 1
                                            
                                        rir_id = f"{rir_counter:04d}"
                                        
                                        if room_id in existing_rooms and rir_id in hdf[room_id]:
                                            rir_counter += 1
                                            continue
                                        
                                        rir_group = room_group.create_group(rir_id)
                                                                
                                        head = Head([x, y, z], radius=0.0875, azimuth=azimuth, elevation=elevation)
                                        
                                        laptop_position = np.array([x + laptop_x_offset, y + laptop_y_offset, z + laptop_z_offset])
                                        
                                        for source in ['mouth', 'laptop']:
                                            source_group = rir_group.create_group(source)
                                            try:
                                                room, max_order_of_reflecs = generateRoom(length, width, height, absorption)
                                                if room is None:
                                                    print(f"Skipping room configuration for {source}")
                                                    # delete the empty source group just created
                                                    del rir_group[source]
                                                    continue

                                            except Exception as e:
                                                print(f"Error generating room ({length}x{width}x{height}, abs={absorption}): {e}")
                                                # delete the empty source group just created
                                                del rir_group[source]
                                                continue
                                                    
                                            
                                            if source == 'mouth':
                                                mouth_dir = DirectionVector(azimuth=azimuth, colatitude=90-elevation, degrees=True) #should be -90 since colatitude = 0 -> up, =180 -> down.
                                                mouth_pattern = Cardioid(orientation=mouth_dir)
                                                room.add_source(head.mouth, directivity=mouth_pattern)
                                            else:  # source == 'laptop'
                                                laptop_source_pattern = Cardioid(orientation=DirectionVector(azimuth=180, degrees=True))
                                                room.add_source(laptop_position, directivity=laptop_source_pattern)
                                                
                                                # TODO: add bessel implementation for scattering on sphere when sound hits head.
                                                # should be optional, i need to be able to generate both types of data.
                                                # but should only be used when sound comes from computer.

                                            if TESTING:
                                                print("source:", source)

                                            left_dir = DirectionVector(azimuth=(azimuth+75)%360, colatitude=90-elevation, degrees=True)
                                            left_pattern = Cardioid(orientation=left_dir)
                                            room.add_microphone(head.leftEar, directivity=left_pattern)

                                            right_dir = DirectionVector(azimuth=(azimuth-75)%360, colatitude=90-elevation, degrees=True)
                                            right_pattern = Cardioid(orientation=right_dir)
                                            room.add_microphone(head.rightEar, directivity=right_pattern)

                                            room.add_microphone(laptop_position)
                                            try:
                                                room.compute_rir()
                                            except ValueError as e:
                                                if "zero-size array to reduction operation maximum" in str(e):
                                                    print(f"Skipping invalid room configuration: {source} source at position {room.sources[0].position}, "
                                                        f"head at [{x}, {y}, {z}], azimuth {azimuth}, elevation {elevation}")
                                                    continue
                                                else:
                                                    # Re-raise other ValueError exceptions
                                                    raise
                             
                                            raw_channel_data = []
                                            max_length = 0
                                            global_max_value = 0
                                            
                                            if source == 'laptop':
                                                # necessary as laptop -> laptop rir is delta, and source is exactly on top of
                                                # receiver, so normalizing to max value will scew the results (repress the other rirs)
                                                # therefore, only save L/R when source = laptop.
                                                    # additionally, normalization should be independent of the other source channels.

                                                for i, ear in enumerate(['left','right']):
                                                    rir = room.rir[i][0]
                                                    raw_channel_data.append(rir)
                                                    max_length = max(max_length, len(rir))

                                                laptop_source_max = max([np.max(np.abs(rir)) for rir in raw_channel_data])
                                                
                                                channel_data = []
                                                for rir in raw_channel_data:
                                                    current_max = np.max(np.abs(rir))
                                                    global_max_value = max(global_max_value, current_max)

                                                    channel_data.append(rir)

                                            else:
                                                for i, ear in enumerate(['left', 'right', 'laptop']):
                                                    rir = room.rir[i][0]  # [mic_index][source_index]
                                                    raw_channel_data.append(rir)
                                                    max_length = max(max_length, len(rir))

                                                mouth_source_max = max([np.max(np.abs(rir)) for rir in raw_channel_data])
                                                
                                                channel_data = [] 
                                                for rir in raw_channel_data:
                                                    # normalized_rir = rir / mouth_source_max
                                                    channel_data.append(rir)


                                            source_group.attrs['left_channel_length'] = len(channel_data[0])
                                            source_group.attrs['right_channel_length'] = len(channel_data[1])

                                            if source == 'mouth':
                                                source_group.attrs['laptop_channel_length'] = len(channel_data[2])
                                            else:
                                                source_group.attrs['laptop_channel_length'] = len(channel_data[0])

                                            source_group.create_dataset('left_channel', data=channel_data[0], compression='gzip')
                                            source_group.create_dataset('right_channel', data=channel_data[1], compression='gzip')
                                            
                                            if source == 'mouth':
                                                source_group.create_dataset('laptop_channel', data=channel_data[2], compression='gzip')
                                            else: # sets channel to zero if source & receiver = laptop:
                                                source_group.create_dataset('laptop_channel', data=np.zeros_like(channel_data[0]), compression='gzip')
                                            
                                            t60_left = pra.experimental.measure_rt60(room.rir[0][0], fs=room.fs, plot=False)  
                                            t60_right = pra.experimental.measure_rt60(room.rir[1][0], fs=room.fs, plot=False)
                                            
                                            if source == 'mouth':
                                                t60_laptop = pra.experimental.measure_rt60(room.rir[2][0], fs=room.fs, plot=False)
                                            else:
                                                t60_laptop = 0

                                            source_group.attrs['sample_rate'] = fs
                                            source_group.attrs['roomLength'] = length
                                            source_group.attrs['roomWidth'] = width
                                            source_group.attrs['roomHeight'] = height
                                            source_group.attrs['cubic_meters'] = length * width * height

                                            source_group.attrs['absorption'] = absorption
                                            
                                            source_group.attrs['head_position_x'] = x
                                            source_group.attrs['head_position_y'] = y
                                            source_group.attrs['head_position_z'] = z

                                            source_group.attrs['laptop_x_offset'] = laptop_x_offset
                                            source_group.attrs['laptop_y_offset'] = laptop_y_offset
                                            source_group.attrs['laptop_z_offset'] = laptop_z_offset
                                            
                                            source_group.attrs['source'] = source
                                            
                                            source_group.attrs['max_order_of_reflecs'] = max_order_of_reflecs
                                            
                                            source_group.attrs['azimuth'] = azimuth
                                            source_group.attrs['elevation'] = elevation
                                            
                                            source_group.attrs['max_amplitude_lapt_source'] = global_max_value
                                            
                                            source_group.attrs['t60_left'] = t60_left
                                            source_group.attrs['t60_right'] = t60_right
                                            source_group.attrs['t60_laptop'] = t60_laptop
                                            
                                        
                                            rir_counter += 1

                                            # for writing periodically since the hpc has time limit: 
                                            hdf.flush()
                        if TESTING:
                            break
                    if TESTING:
                        break
                if TESTING:
                    break
            if TESTING:
                break


print("RIR generation complete")
