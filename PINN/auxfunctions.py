import torch 
import numpy as np
import torch.nn.functional as F 
from scipy import signal
from parser import get_name
import csv

def pad_collate(batch):
    """Custom collate function that pads sequences to match the longest one in the batch."""
    reverbed_speeches, head_params_list, rir_laptops = zip(*batch)
    
    # Get the longest reverbed_speech length
    max_len = max([speech.shape[1] for speech in reverbed_speeches])
    
    # pad all sequences to the max length
    padded_speeches = []
    for speech in reverbed_speeches:
        if speech.shape[1] < max_len:
            padding = max_len - speech.shape[1]
            padded_speech = F.pad(speech, (0, padding), "constant", 0)
            padded_speeches.append(padded_speech)
        else:
            padded_speeches.append(speech)
            
    # Stack everything
    reverbed_speeches_batch = torch.stack(padded_speeches)
    head_params_batch = torch.stack(head_params_list)
    
    # rir laptops should should already be padded to max_rir_length in __getitem__
    rir_laptops_batch = torch.stack(rir_laptops)
    
    return reverbed_speeches_batch, head_params_batch, rir_laptops_batch

def scale_t(input, dim, c=343.0):
    """
    Scale time by multiplying by speed of sound to simplify the wave equation
    This transforms the wave equation from:
    ∂²p/∂t² = c² * (∂²p/∂x² + ∂²p/∂y² + ∂²p/∂z²) 
    to:
    ∂²p/∂t'² = ∂²p/∂x² + ∂²p/∂y² + ∂²p/∂z²
    where t' = c*t
    
    Args: 
        input: tensor of shape (N, dim) where N is the number of samples and dim is the dimension of the input (3 or 4)
        dim: dimension of the input (3 for [x,y,t] or 4 for [x,y,z,t])
        c: speed of sound in m/s (default is 343.0 m/s)
    """
    x_scaled = input.clone()
    if dim==4:
        x_scaled[:, 3] = x_scaled[:, 3] * c
    else:
        raise ValueError("Input dimension must be 4 ([x,y,z,t]).")
    return x_scaled


def targeted_collocation_points(head_params, radius, mono=False, device='cpu'):
    """Generate targeted collocation points for physics loss.
    
    Args:
        head_params: Tensor of shape [5] containing [x, y, z, azimuth, elevation]
        radius: Radius of the head (for computing impact points)
        device: Device to use for tensor operations
    """
    # pos and angles
    head_pos = head_params[:3]  # [x, y, z]
    azimuth = head_params[3]    # azimuth angle in degrees
    elevation = head_params[4]  # elevation angle in degrees
    
    # degs to rads
    azimuth_rad = azimuth * np.pi / 180.0 # cos and sin needs radians
    elevation_rad = elevation * np.pi / 180.0
    
    # time_points = torch.tensor([0, 0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.004, 0.005, 0.01, 0.03], device=device)  # unit: [seconds]
    # at fs=16e3, 480 samples = 30 ms = 0.03 s
    # time_points = torch.tensor([0, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.02, 0.03], device=device)  # unit: [seconds]
    time_points = torch.tensor([0, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.012, 0.014, 0.016, 0.018, 0.02, 0.022, 0.024, 0.026, 0.028, 0.03], device=device)  # unit: [seconds]

    if mono: 
        # direction vector (which direction the wave is coming from)
        direction = torch.tensor([
            torch.cos(elevation_rad) * torch.cos(azimuth_rad),
            torch.cos(elevation_rad) * torch.sin(azimuth_rad),
            torch.sin(elevation_rad)
        ], device=device)
    
    # impact point on head surface.
        impact_point = head_pos - radius * direction

        
        spatial_points = impact_point.unsqueeze(0).repeat(len(time_points), 1)
        t_points = time_points.unsqueeze(1)
        return torch.cat((spatial_points, t_points), dim=1)  # Shape: [num_time_points, 4]

    else:
        # Right ear (90° relative to azimuth)
        right_ear_azimuth = azimuth_rad + (np.pi / 2)
        right_ear_x = head_pos[0] + radius * torch.cos(elevation_rad) * torch.cos(right_ear_azimuth)
        right_ear_y = head_pos[1] + radius * torch.cos(elevation_rad) * torch.sin(right_ear_azimuth)
        right_ear_z = head_pos[2] + radius * torch.sin(elevation_rad)
        
        # Left ear (270° relative to azimuth)
        left_ear_azimuth = azimuth_rad - (np.pi / 2)
        left_ear_x = head_pos[0] + radius * torch.cos(elevation_rad) * torch.cos(left_ear_azimuth)
        left_ear_y = head_pos[1] + radius * torch.cos(elevation_rad) * torch.sin(left_ear_azimuth)
        left_ear_z = head_pos[2] + radius * torch.sin(elevation_rad)
        
        #  spatial points for both ears
        right_ear_pos = torch.tensor([right_ear_x, right_ear_y, right_ear_z], device=device)
        left_ear_pos = torch.tensor([left_ear_x, left_ear_y, left_ear_z], device=device)
        
        # collocation points at different time steps for both ears
        right_spatial_points = right_ear_pos.unsqueeze(0).repeat(len(time_points), 1)
        left_spatial_points = left_ear_pos.unsqueeze(0).repeat(len(time_points), 1)
        
        # spatial points from both ears
        all_spatial_points = torch.cat([right_spatial_points, left_spatial_points], dim=0)
        
        # time points for both ears
        all_time_points = time_points.repeat(2).unsqueeze(1)
        
        # spatial and time coordinates
        return torch.cat((all_spatial_points, all_time_points), dim=1)  # Shape: [2*num_time_points, 4]



def targeted_collocation_with_reflections(head_params, room_dims, radius, device='cpu'):
    """Generate collocation points for both direct and reflected paths.
    Used for the last model in the report, and is based on the section on image source method."""
    head_pos = head_params[:3]
    
    # orig. collocation points (direct path)
    direct_points = targeted_collocation_points(head_params, radius, device=device)
    
    # time-equivalent sources for first-order reflections
    reflection_points = []
    
    # first-order image sources (reflections off each wall)
    image_sources = []
    for dim in range(3):  # x, y, z
        # Reflection across each wall
        mirror_pos = head_pos.clone()
        mirror_pos[dim] = 2 * room_dims[dim] - mirror_pos[dim]  # Reflect across far wall
        image_sources.append(mirror_pos)
        
        mirror_pos = head_pos.clone()
        mirror_pos[dim] = -mirror_pos[dim]  # Reflect across near wall (at origin)
        image_sources.append(mirror_pos)
    
    # collocation points for each image source
    for img_src in image_sources:
        # Calculate distance from image source to head
        dist = torch.norm(img_src - head_pos)
        delay = dist / 343.0  # Speed of sound
        
        # delayed time points for this reflection
        reflection_times = direct_points[:, 3] + delay
        
        # same spatial points but delayed times
        refl_pts = direct_points.clone()
        refl_pts[:, 3] = reflection_times
        
        reflection_points.append(refl_pts)
    
    # comb. direct and reflection points
    all_points = torch.cat([direct_points] + reflection_points, dim=0)
    return all_points


# ---- checkpointing for time-limited hpc and running backups in general
def save_checkpoint(epoch, model, optimizer, loss_history, filename):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss_history': loss_history,
    }
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved: {filename}")

def load_checkpoint(filename, model, optimizer):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    loss_history = checkpoint['loss_history']
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    return start_epoch, loss_history


def save_training_metrics(metrics_path, epochs, total_losses, data_losses=None, physics_losses=None):
    """ Save training to csv for later plotting. Pre-wandb implementation."""
    headers = ['epoch', 'total_loss']
    rows = []
    
    if data_losses is not None:
        headers.append('data_loss')
    if physics_losses is not None: 
        headers.append('physics_loss')
    
    for i, total_loss in enumerate(total_losses):
        row = [i+1, total_loss]  # epochs start at 1
        if data_losses is not None and i < len(data_losses):
            row.append(data_losses[i])
        if physics_losses is not None and i < len(physics_losses):
            row.append(physics_losses[i])
        rows.append(row)
    
    with open(metrics_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)


def validate(model, loader, criterion, device, head_radius, alpha, max_samples=None):
    """Validate the model on the validation set."""
    model.eval()
    val_loss = 0.0
    val_data_loss = 0.0
    val_physics_loss = 0.0
    samples_processed = 0

    for reverbed_speech, head_params, rir_laptop in loader:
        if max_samples and samples_processed >= max_samples:
            break
        
        head_params = head_params.to(device)
        rir_laptop = rir_laptop.to(device)
        
        # forward pass (no grads)
        with torch.no_grad():
            if model.__class__.__name__ in ['simple_pinn', 'simple_lstm_pinn']:
                pred = model(head_params)
            else:
                reverbed_speech = reverbed_speech.to(device)
                pred = model(reverbed_speech, head_params)
            
            data_loss = criterion(pred, rir_laptop)
        
        # physics loss (requires grads)
        batch_physics_loss = 0.0
        for b in range(head_params.shape[0]):
            with torch.enable_grad():
                head_pos = head_params[b].detach()
                collocation_points = targeted_collocation_points(head_pos, head_radius, device=device)
                collocation_points.requires_grad_(True)
                
                try:
                    physics_loss_val = model.loss_PDE(collocation_points, head_params[b]).item()
                    batch_physics_loss += physics_loss_val
                except Exception as e:
                    print(f"Physics loss calculation failed: {e}")
                    batch_physics_loss += 0.0
        
        physics_loss = batch_physics_loss / head_params.shape[0]
        loss = data_loss + alpha * physics_loss
        
        val_loss += loss
        val_data_loss += data_loss
        val_physics_loss += physics_loss
        samples_processed += head_params.size(0)
    
    val_loss /= max(1, samples_processed)
    val_data_loss /= max(1, samples_processed)
    val_physics_loss /= max(1, samples_processed)
    
    return val_loss, val_data_loss, val_physics_loss



def weigthed_mse_loss(pred, target, weight_decay=0.8, early_loss_weight=5.0, early_samples=100):
    """Weighted MSE loss with focus on early reflections. Didn't really give any good results.
    It is also a problem that everything is hard-coded. 
    
    Args:
        pred: Predicted RIR [batch, seq_len] or [batch, channels, seq_len]
        target: Target RIR [batch, seq_len] or [batch, channels, seq_len]
        weight_decay: Exponential decay factor
        early_loss_weight: Weight multiplier for early samples loss
        early_samples: Number of samples considered as early reflections
    """
    batch_size = pred.shape[0]
    
    # both mono and stereo handling:
    if len(pred.shape) == 3:  # [batch, channels, seq_len]
        seq_len = pred.shape[2]
        # exponentially decaying weights (focuses on first samples but also makes smooth gradient)
        weights = torch.pow(weight_decay, torch.arange(seq_len, device=pred.device).float())
        weights = weights / weights.sum()
        
        # main weighted loss across full sequence
        squared_error = (pred - target)**2
        weighted_error = squared_error * weights.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len]
        full_loss = weighted_error.sum() / batch_size
        
        # early reflection loss (first early_samples). heavy weighting to focus on early reflections.
        early_error = torch.nn.functional.mse_loss(
            pred[:, :, :early_samples], 
            target[:, :, :early_samples]
        )
    else:  # [batch, seq_len]
        seq_len = pred.shape[1]
        # exponentially decaying weights
        weights = torch.pow(weight_decay, torch.arange(seq_len, device=pred.device).float())
        weights = weights / weights.sum()
        
        squared_error = (pred - target)**2
        weighted_error = squared_error * weights.unsqueeze(0)  # [1, seq_len]
        full_loss = weighted_error.sum() / batch_size
        
        early_error = torch.nn.functional.mse_loss(
            pred[:, :early_samples], 
            target[:, :early_samples]
        )
    
    return full_loss + early_loss_weight * early_error

def hp_butter(audio_data, fs, cutoff=50, order=4):
    """
    Apply a Butterworth high-pass filter to audio data. not used as it is now.
    
    Args:
        audio_data: Tensor containing audio data
        fs: Sample rate in Hz
        cutoff: Cutoff frequency in Hz (default: 50)
        order: Filter order (default: 4)
    
    Returns:
        Filtered audio data as a tensor
    """
    audio_numpy = audio_data.numpy()
    
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = signal.butter(order, normal_cutoff, btype='highpass', analog=False)
    
    filtered_audio = signal.filtfilt(b, a, audio_numpy)
    
    return torch.from_numpy(filtered_audio).to(dtype=audio_data.dtype)


def reflection_aware_loss(pred, target, peak_height=0.1, peak_distance=5, peak_weight=5.0):
    """Loss function that emphasizes reflection peaks in the target RIR, batch processing."""
    import numpy as np
    from scipy.signal import find_peaks
    import torch.nn.functional as F
    
    mse = F.mse_loss(pred, target)
    
    batch_size = pred.shape[0]
    total_peak_loss = 0.0
    
    with torch.no_grad():
        for i in range(batch_size):
            # stereo-handling:
            if len(pred.shape) == 3:  # [batch, channels, time]
                channels = pred.shape[1]
                example_peak_loss = 0.0
                
                for c in range(channels):
                    # convert to numpy array. #TODO is an alternative possible? would not hurt keeping it on gpu.
                    # edit: looks like it isnt https://discuss.pytorch.org/t/pytorch-argrelmax-or-c-function/36404
                    target_np = target[i, c].detach().cpu().numpy()
                    
                    peaks, _ = find_peaks(np.abs(target_np), height=peak_height, distance=peak_distance)
                    
                    if len(peaks) > 0:
                        peak_indices = torch.LongTensor(peaks).to(pred.device)
                        
                        pred_peaks = pred[i, c, peak_indices]
                        target_peaks = target[i, c, peak_indices]
                        
                        example_peak_loss += F.mse_loss(pred_peaks, target_peaks)
                
                if channels > 0:
                    example_peak_loss /= channels
                    total_peak_loss += example_peak_loss
            else:  # [batch, time]
                #mono handling:
                target_np = target[i].detach().cpu().numpy()

                peaks, _ = find_peaks(np.abs(target_np), height=peak_height, distance=peak_distance)
                
                if len(peaks) > 0:
                    peak_indices = torch.LongTensor(peaks).to(pred.device)
                    
                    pred_peaks = pred[i, peak_indices]
                    target_peaks = target[i, peak_indices]
                    
                    total_peak_loss += F.mse_loss(pred_peaks, target_peaks)
    
    if batch_size > 0:
        total_peak_loss = total_peak_loss * peak_weight / batch_size
    
    return mse + total_peak_loss



def create_compatible_params(head_params):
    """Convert parameters to format expected by model"""
    # For models trained with room dimensions, batch format is:
    # [x, y, z, azimuth, elevation, room_length, room_width, room_height]
    
    x, y, z = head_params[0, 0:3]
    azimuth, elevation = head_params[0, 3:5]
    
    if head_params.shape[1] >= 11:  # Check if we have laptop position
        room_length = head_params[0, 5]
        room_width = head_params[0, 6]
        room_height = head_params[0, 7]
        laptop_x = head_params[0, 8]
        laptop_y = head_params[0, 9]
        laptop_z = head_params[0, 10]
        
        # Create 11D input with laptop position
        model_input = torch.tensor([
            [x, y, z, azimuth, elevation, room_length, room_width, room_height, laptop_x, laptop_y, laptop_z]
        ], device=head_params.device)

    elif head_params.shape[1] >= 8:
        room_length = head_params[0, 5]
        room_width = head_params[0, 6]
        room_height = head_params[0, 7]
        model_input = torch.tensor([
            [x, y, z, azimuth, elevation, room_length, room_width, room_height, 0.0, 0.0, 0.0]
        ], device=head_params.device)
    else: 
    
        model_input = torch.tensor([
            [x, y, z, azimuth, elevation]
        ], device=head_params.device)
    return model_input


def analyse_reflection_accuracy(pred, target, model_name, output_path=r"PINN\outputs", k_reflections=6):
    """Analyze how error increases with each reflection number."""
    from scipy.signal import find_peaks
    
    is_stereo = len(target.shape) > 2  # Batch, channels, samples
    
    if is_stereo:
        # Process each channel separately
        left_results = _analyse_single_channel(pred[:,0,:], target[:,0,:], model_name + "_Left", output_path, k_reflections)
        right_results = _analyse_single_channel(pred[:,1,:], target[:,1,:], model_name + "_Right", output_path, k_reflections)
        
        # Combine results dictionary
        results = {
            'Left': left_results,
            'Right': right_results,
            'mean_time_error_ms': (left_results['mean_time_error_ms'] + right_results['mean_time_error_ms']) / 2
        }
        return results
    
    # for mono data, use the existing logic
    return _analyse_single_channel(pred, target, model_name, output_path, k_reflections)


def _analyse_single_channel(pred, target, channel_name, output_path, k_reflections=6):
    """Analyze a single channel (helper function to avoid code duplication)"""
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.signal import find_peaks
    
    batch_size = pred.shape[0]
    sr = 16000
    
    # Analysis params
    window_size = 15  # samples around each peak to analyze
    min_height = 0.05 # when is a peak a peak?
    min_distance = 5 # in order to not count a peak twice, there should be some hops.
    
    reflection_nums = np.arange(1, k_reflections + 1)
    
    # Metrics per reflection
    peak_qualities = [[] for _ in range(k_reflections)]
    timing_errors = [[] for _ in range(k_reflections)]
    amplitude_errors = [[] for _ in range(k_reflections)]
    detection_rate = np.zeros(k_reflections)
    
    for i in range(batch_size):
        # absolute values for peak detection
        target_abs = torch.abs(target[i]).cpu().numpy()
        pred_abs = torch.abs(pred[i]).cpu().numpy()
        
        # find peaks in target (ground truth)
        target_peaks, _ = find_peaks(target_abs, distance=min_distance, height=min_height * np.max(target_abs))
        if len(target_peaks) == 0:
            target_peaks = [np.argmax(target_abs)]
        target_peaks = sorted(target_peaks, key=lambda idx: target_abs[idx], reverse=True)[:k_reflections]
        target_peaks.sort()  # Sort by time
        
        # Find peaks in prediction
        pred_peaks, _ = find_peaks(pred_abs, distance=min_distance, height=min_height * np.max(pred_abs))
        if len(pred_peaks) == 0:
            pred_peaks = [np.argmax(pred_abs)]
        pred_peaks = sorted(pred_peaks, key=lambda idx: pred_abs[idx], reverse=True)[:k_reflections]
        pred_peaks.sort()
        
        # ground truth reflections: 
        for k, target_peak in enumerate(target_peaks):
            if k >= k_reflections:
                break
                
            # window around target peak:
            window_start = max(0, target_peak - window_size)
            window_end = min(len(target_abs), target_peak + window_size + 1)
            
            # check for peak within window
            window_pred = pred_abs[window_start:window_end]
            if len(window_pred) > 0:
                local_max_idx = np.argmax(window_pred) + window_start
                timing_error = local_max_idx - target_peak
                timing_errors[k].append(timing_error)
                
                # amplitude error
                target_amp = target_abs[target_peak]
                pred_amp = pred_abs[local_max_idx]
                amp_error = abs(target_amp - pred_amp) / max(target_amp, 1e-8)  # Store in variable first
                amplitude_errors[k].append(amp_error)
                
                max_amplitude_error = 0.3  # max 30% amplitude error for a peak to be considered detected
                if abs(pred_amp - target_amp) / target_amp <= max_amplitude_error:
                    detection_rate[k] += 1
            else:
                # No matching peak found
                pass
    
    # Normalize detection rate
    detection_rate = detection_rate / batch_size * 100
    
    # lists to arrays for statistics
    timing_error_means = np.array([np.mean(errors) if errors else np.nan for errors in timing_errors])
    timing_error_stds = np.array([np.std(errors) if errors else np.nan for errors in timing_errors])
    amplitude_error_means = np.array([np.mean(errors) if errors else np.nan for errors in amplitude_errors])

    # direct sound error:
    direct_sound_error = timing_errors[0] if timing_errors[0] else [0]
    mean_sample_error = np.mean(direct_sound_error)
    var_sample_error = np.var(direct_sound_error)
    mean_time_error_ms = mean_sample_error * 1000 / sr  # Convert to ms
    
    # lots of plotting:
    plt.figure(figsize=(12, 8))
    
    # Plot 1: Detection rate
    plt.subplot(2, 2, 1)
    plt.bar(reflection_nums, detection_rate, color='blue', alpha=0.7)
    plt.xlabel('Reflection Number')
    plt.ylabel('Detection Rate ($\%$)')
    plt.title('Detection Rate by Reflection')
    plt.xticks(reflection_nums)
    plt.grid(axis='y', alpha=0.3)
    
    # Plot 2: Timing error
    plt.subplot(2, 2, 2)
    valid_indices = ~np.isnan(timing_error_means) # filter out NaNs (~ isnan returns true for valid indices)
    plt.errorbar(reflection_nums[valid_indices], 
                 timing_error_means[valid_indices],
                 yerr=timing_error_stds[valid_indices],
                 marker='o', linestyle='-')
    plt.xlabel('Reflection Number')
    plt.ylabel('Timing Error (samples)')
    plt.title('Timing Error by Reflection')
    plt.xticks(reflection_nums)
    plt.grid(alpha=0.3)
    
    # Plot 3: Amplitude error
    plt.subplot(2, 2, 3)
    valid_indices = ~np.isnan(amplitude_error_means)
    plt.plot(reflection_nums[valid_indices], 
             amplitude_error_means[valid_indices], 
             marker='o', linestyle='-')
    plt.xlabel('Reflection Number')
    plt.ylabel('Amplitude Error (normalized)')
    plt.title('Amplitude Error by Reflection')
    plt.xticks(reflection_nums)
    plt.grid(alpha=0.3)
    
    # Plot 4: Example waveforms with numbered peaks
    plt.subplot(2, 2, 4)
    example_idx = 0  
    t = np.arange(len(target[example_idx]))
    plt.plot(t, target[example_idx].cpu().numpy(), 'b-', label='Target', alpha=0.7)
    plt.plot(t, pred[example_idx].cpu().numpy(), 'r-', label='Predicted', alpha=0.7)
    
    # find peaks for plotted example and number them
    target_abs = torch.abs(target[example_idx]).cpu().numpy()
    pred_abs = torch.abs(pred[example_idx]).cpu().numpy()
    
    target_peaks, _ = find_peaks(target_abs, distance=min_distance, height=min_height * np.max(target_abs))
    if len(target_peaks) == 0:
        target_peaks = [np.argmax(target_abs)]
    target_peaks = sorted(target_peaks, key=lambda idx: target_abs[idx], reverse=True)[:k_reflections]
    target_peaks.sort()
    
    for i, peak in enumerate(target_peaks):
        if i < 5:  # Only label first 5 to avoid clutter
            plt.text(peak, target_abs[peak], f"{i+1}", color='blue', fontsize=12)
            plt.axvline(peak, color='blue', linestyle='--', alpha=0.3)
    
    plt.title('Example Waveform with Reflection Numbers')
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(rf'PINN\outputs\reflection_error_analysis_{channel_name}.pdf')
    plt.close()