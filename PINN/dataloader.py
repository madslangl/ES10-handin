import torch
import torch.nn.functional as F
import torchaudio as ta
from torch.utils.data import Dataset, DataLoader
from scipy.io import wavfile
import h5py 
import os
from pathlib import Path
import random
import numpy as np
from parser import get_parser
from tqdm import tqdm


class RIRDataset(Dataset):
    # TODO: implement callable audio transforms in this code too.
    """Loads dataset for RIR prediction. 
    Args: 
        rir_path (str): path to the RIR dataset (.h5 file)
        speech_path (str): path to the speech dataset
        split (str): train/valid/test split
        transform (optional): augmentation techniques to apply to the dataset. happens before convolution.
        speech_duration (int): duration of the speech signal in seconds.
        train_ratio (float): proportion of data for trainingn
        valid_ratio (float): proportion of data for validation
        seed (int): for reproduceability
    """

    def __init__(self, rir_path, speech_path, split, transform=None, 
                 speech_duration=3, train_ratio=0.8, valid_ratio=0.1, seed=42, max_samples=None):
        self.rir_path = rir_path
        self.speech_path = speech_path
        self.split = split

        self.transform = transform
        self.speech_duration = speech_duration
        self.train_ratio = train_ratio
        self.valid_ratio = valid_ratio
        self.max_samples = max_samples
        parser = get_parser()
        self.args = parser.parse_args()
        self.mono = self.args.mono
        self.model = self.args.model
        self.rir_length = self.args.rir_length
        self.sample_rate = self.args.sample_rate
        self.room_dims = self.args.room_dims
        self.laptop_pos = self.args.laptop_pos

        random.seed(seed)
        
        if self.speech_path is not None:
            self.speech_files = self._get_speech_files()
        else: 
            self.speech_files = []

        self._get_rir_pairs()
    

    def _get_speech_files(self):
        """Finds sentences in EARS dataset.""" 
        speech_files = []
        print("getting speech files from: ", self.speech_path)
        for subdir, dirs, files in os.walk(self.speech_path):
            for file in files:
                if file.endswith('.wav') and file.startswith('sentences'):
                    speech_files.append(os.path.join(subdir, file))
        return speech_files

    def _get_rir_pairs(self):
        # TODO: right now, metadata is stored in ~/data/cache/[filename]_metadata.pkl, but it would make sense to store with the dataset. 
        """Maps the rir dataset structure and generates metadata"""
        print(f"Getting RIR pairs from: {self.rir_path}")
        
        h5_name = os.path.basename(self.rir_path)
        cache_dir = os.path.join(os.path.dirname(self.rir_path), "cache")
        os.makedirs(cache_dir, exist_ok=True)
        metadata_path = os.path.join(cache_dir, f"{h5_name}_metadata.pkl")
        
        if os.path.exists(metadata_path):
            try:
                import pickle
                print(f"Found metadata at {metadata_path}, loading...")
                with open(metadata_path, 'rb') as f:
                    cache_data = pickle.load(f)
                    self.room_list = cache_data['room_list']
                    self.position_counts = cache_data['position_counts']
                    self.pair_count = cache_data['pair_count']
                    print(f"loaded from cache: {len(self.room_list)} rooms, {self.pair_count} total positions")
            except Exception as e:
                print(f"no metadata, {e}")
                print("Building metadata from scratch...")
                self._build_metadata_from_file()
        else:
            print("No metadata found, scanning H5 file (this will take some time, but only once pr. dataset)...")
            self._build_metadata_from_file()
            
            try:
                import pickle
                cache_data = {
                    'room_list': self.room_list,
                    'position_counts': self.position_counts,
                    'pair_count': self.pair_count
                }
                with open(metadata_path, 'wb') as f:
                    pickle.dump(cache_data, f)
                print(f"Metadata cached to {metadata_path} for faster future loading")
            except Exception as e:
                print(f"failed to save metadata: {e}")
        
        room_indices = {}
        for room_idx, room_id in enumerate(self.room_list):
            room_indices[room_id] = room_idx
        
        shuffled_rooms = list(room_indices.keys())
        random.shuffle(shuffled_rooms)

        n_rooms = len(shuffled_rooms)
        n_train = int(n_rooms * self.train_ratio)
        n_valid = int(n_rooms * self.valid_ratio)

        train_rooms = shuffled_rooms[:n_train]
        valid_rooms = shuffled_rooms[n_train:n_train+n_valid]
        test_rooms = shuffled_rooms[n_train+n_valid:]

        # select which rooms to use based on the split
        if self.split == 'train':
            selected_rooms = train_rooms
        elif self.split == 'valid':
            selected_rooms = valid_rooms
        else: 
            selected_rooms = test_rooms

        #all positions for rooms
        self.indices = []
        position_offset = 0
        for room_idx, room_id in enumerate(self.room_list):
            positions_in_room = self.position_counts[room_idx]
            if room_id in selected_rooms:
                # all position indices for this room
                for pos in range(positions_in_room):
                    self.indices.append(position_offset + pos)
            position_offset += positions_in_room

        print(f"Split: {self.split}, using {len(selected_rooms)} rooms with {len(self.indices)} positions")
        
        if self.max_samples is not None and len(self.indices) > self.max_samples:
            print(f"debug: limiting samples to {self.max_samples} for {self.split} split")
            self.indices = self.indices[:self.max_samples]

    def _build_metadata_from_file(self):
        """Scans the H5 file to build complete metadata. Is only done once per dataset."""
        room_list = []
        position_counts = []
        pair_count = 0
        
        with h5py.File(self.rir_path, 'r') as hdf:
            total_rooms = len(hdf.keys())
            print(f"Scanning {total_rooms} rooms in H5 file...")
            
            for i, room_id in enumerate(hdf.keys()):
                if i % 20 == 0 or i == total_rooms-1:
                    print(f"Scanning room {i+1}/{total_rooms}: {room_id}")
                
                room_list.append(room_id)
                positions_in_room = len(hdf[room_id].keys())
                position_counts.append(positions_in_room)
                pair_count += positions_in_room
        
        self.room_list = room_list
        self.position_counts = position_counts
        self.pair_count = pair_count
        print(f"scan complete: {len(room_list)} rooms with {pair_count} total positions")

    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        """Map an index to a specific room and position, then load that sample"""
        # Get the global position index from filtered indices
        if idx >= len(self.indices):
            raise IndexError(f"Index {idx} out of bounds for dataset with {len(self.indices)} samples")
        
        position_idx = self.indices[idx]
        
        # map to specific room, position
        for room_idx, pos_count in enumerate(self.position_counts):
            if position_idx < pos_count:
                room_id = self.room_list[room_idx]
                return self._load_sample(room_id, position_idx)
            position_idx -= pos_count        
    
    def _load_sample(self, room_id, position_idx):
        """Load a specific sample from the H5 file"""
        # if position_idx % 100 == 0:
            # print(f"Loading sample: room={room_id}, position={position_idx}")
        
        with h5py.File(self.rir_path, 'r') as hdf:
            # following handles if there should be any errors in the dataset.
            # i generated it using an hpc with a time limit of 12h, so there might be 
            # some errors in the samples where generation was stopped before flushing.
            try:
                position_keys = list(hdf[room_id].keys())
                if position_idx >= len(position_keys):
                    # handle idx error by choosing a random valid position
                    print(f"Warning: Position index {position_idx} out of bounds for room {room_id}")
                    position_idx = random.randint(0, len(position_keys)-1)
                
                rir_id = position_keys[position_idx]
                
                if 'laptop' not in hdf[room_id][rir_id]:
                    return self.__getitem__((position_idx + 1) % len(self))
                    
                if 'mouth' not in hdf[room_id][rir_id]:
                    return self.__getitem__((position_idx + 1) % len(self))
                
                # laptop data:
                group_laptop = hdf[room_id][rir_id]['laptop']
                
                head_x = group_laptop.attrs['head_position_x']
                head_y = group_laptop.attrs['head_position_y']
                head_z = group_laptop.attrs['head_position_z']
                
                if self.room_dims:
                    # check: if all required room dimensions exist
                    if ('roomLength' in group_laptop.attrs and 
                        'roomWidth' in group_laptop.attrs and 
                        'roomHeight' in group_laptop.attrs):
                        
                        room_x = float(group_laptop.attrs['roomLength'])
                        room_y = float(group_laptop.attrs['roomWidth'])
                        room_z = float(group_laptop.attrs['roomHeight'])
                    else:
                        print(f"Missing room dimensions in {room_id}/{rir_id} - skipping")
                        return self.__getitem__((position_idx + 7) % len(self))
                

                azimuth = group_laptop.attrs['azimuth']
                elevation = group_laptop.attrs['elevation']

                distance = torch.sqrt(torch.tensor(head_x**2 + head_y**2 + head_z**2, dtype=torch.float32))
                
                # laptop -> head RIR
                if 'laptop_channel' not in group_laptop:
                    return self.__getitem__((position_idx + 1) % len(self))
                
                rir_left = torch.tensor(group_laptop['left_channel'][:self.rir_length], dtype=torch.float32)
                rir_right = torch.tensor(group_laptop['right_channel'][:self.rir_length], dtype=torch.float32)


                group_mouth = hdf[room_id][rir_id]['mouth']
                
                rir_headset_left = torch.tensor(group_mouth['left_channel'][:self.rir_length], dtype=torch.float32)
                rir_headset_right = torch.tensor(group_mouth['right_channel'][:self.rir_length], dtype=torch.float32)

                if self.mono:
                    rir = rir_left
                    azimuth = azimuth - 90
                else: 
                    rir = torch.stack((rir_left, rir_right), axis=0)
                
                if self.room_dims: 

                    if self.laptop_pos:
                        laptop_x = head_x + group_laptop.attrs['laptop_x_offset']
                        laptop_y = head_y + group_laptop.attrs['laptop_y_offset']
                        laptop_z = head_z + group_laptop.attrs['laptop_z_offset']

                        head_params = torch.tensor([
                        # Head pos relative to laptop
                        head_x, head_y, head_z,
                        azimuth, elevation,
                        # Laptop pos in room coordinates
                        room_x, room_y, room_z,
                        laptop_x, laptop_y, laptop_z
                    ], dtype=torch.float32)
                    else:
                        head_params = torch.tensor([head_x,head_y,head_z, azimuth, elevation, room_x, room_y, room_z], dtype=torch.float32)
                else:
                    head_params = torch.tensor([head_x,head_y,head_z, azimuth, elevation], dtype=torch.float32)
                
                # rir length handling. 
                if self.mono:
                    # mono case - rir is a 1D tensor
                    if len(rir) > self.rir_length:
                        rir = rir[:self.rir_length]
                    else:
                        rir = F.pad(rir, (0, self.rir_length - len(rir)), 'constant')
                else:
                    # stereo case - rir is [2, length]
                    if rir.shape[1] > self.rir_length:
                        rir = rir[:, :self.rir_length]
                    #  this handles if rirs are shorter than required length. they shouldnt be, but as mentioned, there might be some errors in the dataset.
                    else:
                        rir = F.pad(rir, (0, self.rir_length - rir.shape[1]), 'constant')
                    
                
                rir_laptop = rir.clone().detach().to(dtype=torch.float32) 

                fs = group_laptop.attrs['sample_rate']

                if fs != self.sample_rate:
                    resampler = ta.transforms.Resample(orig_freq=fs, new_freq=self.sample_rate)
                    rir_left = resampler(rir_left)
                    rir_right = resampler(rir_right) 
                    rir_headset_left = resampler(rir_headset_left)
                    rir_headset_right = resampler(rir_headset_right) 
                    fs = self.sample_rate

                actual_rir_length = len(rir)
                
                # mouth data:
                if self.model == 'simple_pinn':
                    # TODO: this was implemented to ensure that the dataloader could be used for all models (incl. proposed model3). Consider removing dummy tensors for the speech input
                    dummy_convolved = torch.zeros((2, 48), dtype=torch.float32)
                    
                    if self.split == 'test':
                        return dummy_convolved, head_params, rir_laptop, room_id, position_idx
                    else:
                        return dummy_convolved, head_params, rir_laptop

                if self.model == 'feeb_pinn':
                    # this loads speech files, but only if the model is not simple_pinn (for the feeb_pinn implementation which isnt implemented)
                    speech_file = random.choice(self.speech_files)
                    
                    # had some issues with torchaudio, so added scipy as a backup.
                    # TODO: it seems like it uses scipy in quite a large percentage of cases,
                    # so maybe something is wrong with the ta handling? 
                    try:
                        speech_data = ta.load(speech_file, normalize=True)[0]
                    except Exception as e:
                        # print(f"Torchaudio failed, trying scipy: {e}") # commented because it happens in a lot of cases.
                        sample_rate, speech_numpy = wavfile.read(speech_file)

                        # conv to float and normalize speech sig to [-1, 1]
                        # this handles both int16 and int32 formats.
                        
                        if speech_numpy.dtype == np.int16:
                            speech_data = torch.tensor(speech_numpy, dtype=torch.float32) / 32768.0
                        elif speech_numpy.dtype == np.int32:
                            speech_data = torch.tensor(speech_numpy, dtype=torch.float32) / 2147483648.0
                        else:
                            speech_data = torch.tensor(speech_numpy, dtype=torch.float32)
                        
                        # reshape to match torchaudio format if scipy was used 
                        if len(speech_data.shape) == 1:
                            speech_data = speech_data.unsqueeze(0)
                        else:
                            speech_data = speech_data.t()  

                    # mono conversion if stereo (the anechoic speech should be mono)
                    # assuming that ch 1 is just as good as ch 2.
                    if speech_data.shape[0] > 1: 
                        # speech_data = speech_data.mean(dim=0)
                        speech_data = speech_data[0]   
                    else:
                        speech_data = speech_data[0]
                        
                    # Process speech data
                    max_samples = int(self.speech_duration * fs)
                    if len(speech_data) > max_samples:
                        speech_data = speech_data[:max_samples]
                    else:
                        pad_length = max_samples - len(speech_data)
                        speech_data = F.pad(speech_data, (0, pad_length), 'constant')
                        
                    # fade to avoid clicks in audio 
                    fade_duration = int(0.01*fs)
                    fade = torch.linspace(1,0,fade_duration)
                    speech_data[-fade_duration:] *= fade

                    # speech_data = hp_butter(speech_data, self.sample_rate, cutoff=50, order=4)  # hp filter. quite a lot of noise in the lower freqs of the EARS dataset.

                    speech_data_conv = speech_data.view(1, 1, -1)  # [1, 1, signal_length]

                    # Flip the RIR for convolution (kernel needs to be flipped compared to np.convolve)
                    rir_headset_left_kernel = rir_headset_left.flip(0).view(1, 1, -1)  # [1, 1, kernel_length]
                    rir_headset_right_kernel = rir_headset_right.flip(0).view(1, 1, -1)  # [1, 1, kernel_length]
                    
                    convolved_left = F.conv1d(speech_data_conv, rir_headset_left_kernel, padding=rir_headset_left.shape[0]-1).squeeze()
                    convolved_right = F.conv1d(speech_data_conv, rir_headset_right_kernel, padding=rir_headset_right.shape[0]-1).squeeze()
                    
                    max_val = torch.max(torch.max(torch.abs(convolved_left)), torch.max(torch.abs(convolved_right)))
                    convolved_left = convolved_left / max_val
                    convolved_right = convolved_right / max_val
                    
                    max_len = max(convolved_left.shape[0], convolved_right.shape[0])
                    if convolved_left.shape[0] < max_len:
                        padding = max_len - convolved_left.shape[0]
                        convolved_left = F.pad(convolved_left, (0, padding), "constant", 0)
                    elif convolved_right.shape[0] < max_len:
                        padding = max_len - convolved_right.shape[0]
                        convolved_right = F.pad(convolved_right, (0, padding), "constant", 0)

                    convolved = torch.stack((convolved_left, convolved_right), dim=0)
                    
                # return convolved, head_params, rir_laptop
                # this is where the dummy convolved is used.
                if self.split == 'test':
                    return convolved, head_params, rir_laptop, room_id, position_idx
                else:
                    return convolved, head_params, rir_laptop 

            except Exception as e:
                print(f"Error loading sample: {e}")
                # if sample is bad, load a new and print an error. I haven't seen this activated yet.
                return self.__getitem__(random.randint(0, len(self)-1))