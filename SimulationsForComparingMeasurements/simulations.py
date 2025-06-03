import pyroomacoustics as pra
import numpy as np
import matplotlib.pyplot as plt
import scienceplots

"""Script for simulating the rooms recorded in the measurement journal.
The rooms were all measured using a tape measure, and the absorption coefficients were chosen based on most fitting.
At the bottom, theres a lot of commented code. This is the Sabine and Erying Calculations.
"""


room_1_dimensions = [5.7, 3.75, 2.75]
room_2_dimensions = [3.7, 2.75, 2.75]
room_3_dimensions = [3.7, 1.4, 2.75]
room_4_dimensions = [3.8, 2.85, 2.75]

# for comparing to measured rirs:
fs = 48000

wall_material = pra.Material(energy_absorption=0.071,scattering=0.1)

m_1 = pra.make_materials(
    ceiling="rough_concrete",
    floor="linoleum_on_concrete",
    east="plasterboard",
    west="plasterboard",
    north="plasterboard",
    south="rough_concrete",
)
m_2 = m_1

m_3 = pra.make_materials(
    ceiling="rough_concrete",
    floor="linoleum_on_concrete",
    east="smooth_brickwork_flush_pointing",
    west="plasterboard",
    north="plasterboard",
    south="rough_concrete",
)

m_4 = m_1

room1 = pra.ShoeBox(
    room_1_dimensions, fs=fs, materials=m_1, max_order=50, air_absorption=True, ray_tracing=True
)
room2 = pra.ShoeBox(
    room_2_dimensions, fs=fs, materials=m_1, max_order=30, air_absorption=True, ray_tracing=True
)
room3 = pra.ShoeBox(
    room_3_dimensions, fs=fs, materials=m_1, max_order=30, air_absorption=True, ray_tracing=True
)
room4 = pra.ShoeBox(
    room_4_dimensions, fs=fs, materials=m_1, max_order=30, air_absorption=True, ray_tracing=True
)


#just for plotting:
room1.add_source([1,1,1])
room1.add_microphone([1.5,1.5,1.5])

room1.add_microphone([room_1_dimensions[0]/2, room_1_dimensions[1]/2, room_1_dimensions[2]/2])

room1.add_source([(room_1_dimensions[0]/2)+(1.5/np.sqrt(2)),
                  (room_1_dimensions[1]/2)+(1.5/np.sqrt(2)),
                    1.5])
room1.add_microphone([room_1_dimensions[0]/2, room_1_dimensions[1]/2, 1.5])

room2.add_source([(room_2_dimensions[0]/2)+(1.5/np.sqrt(2)),
                  (room_2_dimensions[1]/2)+(1.5/np.sqrt(2)),
                    1.5])
room2.add_microphone([room_2_dimensions[0]/2, room_2_dimensions[1]/2, 1.5])

room3.add_source([2.9, 1.3, 1.5])
room3.add_source([(room_3_dimensions[0]/2)+(1.5/np.sqrt(2)),
                  (room_3_dimensions[1]/2)+(1.5/np.sqrt(2)),
                    1.5])
room3.add_microphone([room_3_dimensions[0]/2, room_3_dimensions[1]/2, 1.5])

room4.add_source([(room_4_dimensions[0]/2)+(1.5/np.sqrt(2)),
                  (room_4_dimensions[1]/2)+(0.5/np.sqrt(2)), 
                  1.5])
room4.add_microphone([room_4_dimensions[0]/2, room_4_dimensions[1]/2, 1.5])

room1.image_source_model()
room2.image_source_model()
room3.image_source_model()
room4.image_source_model()

room1.compute_rir()
room2.compute_rir()
room3.compute_rir()
room4.compute_rir()

t60_1 = room1.measure_rt60(plot=True)
t60_2 = room2.measure_rt60()
t60_3 = room3.measure_rt60()
# plt.figure(figsize=[6.12,3])
# plt.grid()
t60_4 = room4.measure_rt60()
# plt.show()

# changed the absorption coefficients to be same for all frequencies to get specific frequency t60 values
# in .venv\Lib\site-packages\pyroomacoustics\data\materials.json
print("t60 room 1, A1-204: ", t60_1)
print("t60 room 2, A1-213: ", t60_2)
print("t60 room 3, A1-219: ", t60_3)
print("t60 room 4, A1-221:", t60_4)

def save_normalized_rir(filename, fs, rir):
    # Normalize to range -1 to 1
    normalized_rir = rir / np.max(np.abs(rir))
    # Save as 32-bit float WAV
    wavfile.write(filename, fs, normalized_rir.astype(np.float32))
    print(f"Saved {filename}")

    
from scipy.io import wavfile
save_normalized_rir(r"SimulationsForComparingMeasurements\outputs\simu_room1_A1-204_rir.wav", fs, room1.rir[0][0])
save_normalized_rir(r"SimulationsForComparingMeasurements\outputs\simu_room2_A1-213_rir.wav", fs, room2.rir[0][0])
save_normalized_rir(r"SimulationsForComparingMeasurements\outputs\simu_room3_A1-219_rir.wav", fs, room3.rir[0][0])
save_normalized_rir(r"SimulationsForComparingMeasurements\outputs\simu_room4_A1-221_rir.wav", fs, room4.rir[0][0])

# --------------------------- below is sabine and eyring calculation ---------------------------


area_1_w1 = room_1_dimensions[0] * room_1_dimensions[2]
area_1_w2 = room_1_dimensions[1] * room_1_dimensions[2]
area_1_w3 = area_1_w1
area_1_w4 = area_1_w2
area_1_ceiling = room_1_dimensions[0] * room_1_dimensions[1]
area_1_floor = area_1_ceiling
absorp_1_1 = 0.05
absorp_1_2 = 0.02
absorp_1_3 =0.02
absorp_1_4 =0.07
absorp_1_ceiling = 0.06
absorp_1_floor = 0.03

volume1 = room_1_dimensions[0] * room_1_dimensions[1] * room_1_dimensions[2]
surface_area1 = (
    2 * (room_1_dimensions[0] * room_1_dimensions[1])
    + 2 * (room_1_dimensions[0] * room_1_dimensions[2])
    + 2 * (room_1_dimensions[1] * room_1_dimensions[2])
)

total_absorption_1 = (
    area_1_w1 * absorp_1_1
    + area_1_w2 * absorp_1_2
    + area_1_w3 * absorp_1_3
    + area_1_w4 * absorp_1_4
    + area_1_ceiling * absorp_1_ceiling
    + area_1_floor * absorp_1_floor
)

# Calculate average absorption coefficient
a_1_avg = total_absorption_1 / surface_area1

area_2_w1 = room_2_dimensions[0] * room_2_dimensions[2]
area_2_w2 = room_2_dimensions[1] * room_2_dimensions[2]
area_2_w3 = area_2_w1
area_2_w4 = area_2_w2
area_2_ceiling = room_2_dimensions[0] * room_2_dimensions[1]
area_2_floor = area_2_ceiling

absorp_2_1 = 0.07
absorp_2_2 = 0.02
absorp_2_3 = 0.02
absorp_2_4 = 0.07
absorp_2_ceiling = 0.06
absorp_2_floor = 0.03

volume2 = room_2_dimensions[0] * room_2_dimensions[1] * room_2_dimensions[2]
surface_area2 = (
    2 * (room_2_dimensions[0] * room_2_dimensions[1])
    + 2 * (room_2_dimensions[0] * room_2_dimensions[2])
    + 2 * (room_2_dimensions[1] * room_2_dimensions[2])
)

total_absorption_2 = (
    area_2_w1 * absorp_2_1
    + area_2_w2 * absorp_2_2
    + area_2_w3 * absorp_2_3
    + area_2_w4 * absorp_2_4
    + area_2_ceiling * absorp_2_ceiling
    + area_2_floor * absorp_2_floor
)

area_3_w1 = room_3_dimensions[0] * room_3_dimensions[2]
area_3_w2 = room_3_dimensions[1] * room_3_dimensions[2]
area_3_w3 = area_3_w1
area_3_w4 = area_3_w2
area_3_ceiling = room_3_dimensions[0] * room_3_dimensions[1]
area_3_floor = area_3_ceiling

absorp_3_1 = 0.09
absorp_3_2 = 0.02
absorp_3_3 = 0.02
absorp_3_4 = 0.07
absorp_3_ceiling = 0.06
absorp_3_floor = 0.03

volume3 = room_3_dimensions[0] * room_3_dimensions[1] * room_3_dimensions[2]

surface_area3 = (
    2 * (room_3_dimensions[0] * room_3_dimensions[1])
    + 2 * (room_3_dimensions[0] * room_3_dimensions[2])
    + 2 * (room_3_dimensions[1] * room_3_dimensions[2])
)

total_absorption_3 = (
    area_3_w1 * absorp_3_1
    + area_3_w2 * absorp_3_2
    + area_3_w3 * absorp_3_3
    + area_3_w4 * absorp_3_4
    + area_3_ceiling * absorp_3_ceiling
    + area_3_floor * absorp_3_floor
)

area_4_w1 = room_4_dimensions[0] * room_4_dimensions[2]
area_4_w2 = room_4_dimensions[1] * room_4_dimensions[2]
area_4_w3 = area_4_w1
area_4_w4 = area_4_w2

area_4_ceiling = room_4_dimensions[0] * room_4_dimensions[1]
area_4_floor = area_4_ceiling

absorp_4_1 = 0.1
absorp_4_2 = 0.02
absorp_4_3 = 0.02
absorp_4_4 = 0.07
absorp_4_ceiling = 0.06
absorp_4_floor = 0.03

volume4 = room_4_dimensions[0] * room_4_dimensions[1] * room_4_dimensions[2]
surface_area4 = (
    2 * (room_4_dimensions[0] * room_4_dimensions[1])
    + 2 * (room_4_dimensions[0] * room_4_dimensions[2])
    + 2 * (room_4_dimensions[1] * room_4_dimensions[2])
)

total_absorption_4 = (
    area_4_w1 * absorp_4_1
    + area_4_w2 * absorp_4_2
    + area_4_w3 * absorp_4_3
    + area_4_w4 * absorp_4_4
    + area_4_ceiling * absorp_4_ceiling
    + area_4_floor * absorp_4_floor
)





def t60_sabine(V, total_absorption):
    return 0.161 * V / total_absorption

def t60_eyring(V, total_absorption):
    return 0.161 * V / (total_absorption * np.log(10)) 

t60_sabine_room1 = t60_sabine(volume1, total_absorption_1)
t60_sabine_room2 = t60_sabine(volume2, total_absorption_2)
t60_sabine_room3 = t60_sabine(volume3, total_absorption_3)
t60_sabine_room4 = t60_sabine(volume4, total_absorption_4)

print("RT60 room 1 (Sabine formula):", t60_sabine_room1)
print("RT60 room 2 (Sabine formula):", t60_sabine_room2)
print("RT60 room 3 (Sabine formula):", t60_sabine_room3)
print("RT60 room 4 (Sabine formula):", t60_sabine_room4)

# print("RT60 room 1 (Eyring formula):", t60_eyring(volume1, total_absorption_1))
# print("RT60 room 2 (Eyring formula):", t60_eyring(volume2, total_absorption_2))
# print("RT60 room 3 (Eyring formula):", t60_eyring(volume3, total_absorption_3))
# print("RT60 room 4 (Eyring formula):", t60_eyring(volume4, total_absorption_4))





# room1.plot_rir()
# room2.plot_rir()
# room3.plot_rir()
# room4.plot_rir()

# plt.show()