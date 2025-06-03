import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from network import simple_pinn
from dataloader import RIRDataset
from torch.utils.data import DataLoader 
from parser import get_parser
from auxfunctions import targeted_collocation_points, targeted_collocation_with_reflections, create_compatible_params

import scienceplots
plt.style.use('science')

"""This script loads a trained model and evaluates it on a test dataset, plotting a few predicted RIRs against the ground truth.
In the end, a summary of the average physics loss and MSE on the test set is printed, if flag 'DATASET_STATISTICS' is set to True. This takes a while to run.
For plotted statistics on the whole dataset, refer to statistics_pred.py."""

DATASET_STATISTICS = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args = get_parser().parse_args()
rir_length = args.rir_length


if args.model=='simple_pinn':
    print("simple_pinn")
    model = simple_pinn(
    hidden_dim=64,
    feature_dim=128,
    pinn_hidden_dim=64,
    pinn_num_hidden=4,
    rir_length=rir_length,
    mono=args.mono,
    room_dims=args.room_dims,
    laptop_pos=args.laptop_pos
    ).to(device)


model_base_path = args.models
# example path: needs to be changed to match your setup.
model_path = r"examples\more-layers-alpha03simple_pinn_step_27000.pth"
model_name= "alpha03simple_pinn_step_27k"


rir_path = args.rir_path
if args.debugging:
    max_samples = 5
else:
    max_samples = None

model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

test_dataset = RIRDataset(
    rir_path=args.rir_path,
    speech_path=None, # No speech data needed for prediction. bit of a leftover.
    split='test',
    max_samples = max_samples, # only for debugging.
)  

test_dataset.room_dims = args.room_dims
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

import random
seed = 42

max_samples = 5
random_indices = random.sample(range(len(test_dataset)), max_samples)
print(f"Selected random indices: {random_indices}")


plt.figure()

with torch.no_grad():
    for i, sample_idx in enumerate(random_indices):
        # Get specific sample directly without DataLoader
        sample_data = test_dataset[sample_idx]
        
        # Convert to batched tensors - what DataLoader would do
        convolved_audio = sample_data[0].unsqueeze(0).to(device)  # Add batch dimension
        head_params = sample_data[1].unsqueeze(0).to(device)
        rir_true = sample_data[2].unsqueeze(0).to(device)
        
        # Get room/position information (safely)
        try:
            room_id = sample_data[3]
            position_idx = sample_data[4]
        except IndexError:
            # Dataset doesn't include room_id and position_idx, use index instead
            room_id = f"Unknown"
            position_idx = f"{sample_idx}"
        
        print(f"Processing sample {i+1}/5: Sample {sample_idx}")


    # Create room info string for the plot title
    # debugging:
        print(f"Audio shape: {convolved_audio.shape}")
        print(f"RIR shape: {rir_true.shape}")
        print(f"Head params shape: {head_params.shape}")

        model_input = create_compatible_params(head_params)

        if args.model == 'simple_pinn':
            print("model:simple_pinn")
            # rir_pred = model(head_params)
            rir_pred = model(model_input)
        elif args.model =='simple_lstm_pinn':
            rir_pred = model(model_input)
        else:
            rir_pred = model(convolved_audio, model_input)      

        mse = torch.nn.functional.mse_loss(rir_pred, rir_true).item()


        plt.figure(figsize=(6.5, 3.2))

        if len(rir_true.shape) >= 3:  # Batch, channels, samples
            # Plot left channel
            plt.subplot(2, 1, 1)
            plt.plot(rir_true[0, 0].cpu().numpy(), label='Ground Truth L', color='r')

            plt.plot(rir_pred[0, 0].cpu().numpy(), label='Predicted L', color='k')
            plt.title(f"Left Channel - Room: {room_id}, Position: {position_idx}")
            plt.legend()
            plt.grid(True)
            
            # Plot right channel
            plt.subplot(2, 1, 2)
            plt.plot(rir_true[0, 1].cpu().numpy(), label='Ground Truth R', color='r')

            plt.plot(rir_pred[0, 1].cpu().numpy(), label='Predicted R', color='k')
            plt.title("Right Channel")
            plt.legend()
            plt.grid(True)
        else:
            # Original mono plotting code
            plt.subplot(2, 1, 1)
            plt.plot(rir_pred.cpu().numpy().squeeze(), label='Predicted', color='k')
            plt.title(f"Room: {room_id}, Position: {position_idx}")
            plt.legend()
            plt.grid(True)
            
            plt.subplot(2, 1, 2)
            plt.plot(rir_true.cpu().numpy().squeeze(), label='Ground Truth', color='r')
            plt.legend()
            plt.grid(True)

        plt.xlabel("Sample")
        plt.ylabel("Amplitude")

        plt.tight_layout()

# debugs: 
        print(f"Position: {head_params[0, 0:3]}, Az: {head_params[0, 3]}, El: {head_params[0, 4]}")
        print(f"Transformed input: {model_input}")
        print(f"RIR prediction mean: {rir_pred.mean().item()}, std: {rir_pred.std().item()}")
        print(f"Ground truth mean: {rir_true.mean().item()}, std: {rir_true.std().item()}")


        plt.savefig(rf"PINN\outputs\{model_name}-{i}errors.pdf")
        print("saved")


if DATASET_STATISTICS:
    total_mse = 0
    total_samples = 0
    total_physics_loss = 0

    # this goes through the test set and computes the average physics loss and MSE.
    # be careful, this can take some time
    with torch.no_grad():
        for batch_data in tqdm(test_loader, desc="Evaluating test set", total=len(test_loader)):
            convolved_audio = batch_data[0].to(device)
            head_params = batch_data[1].to(device)
            rir_true = batch_data[2].to(device)

            if args.model == 'simple_pinn':
                # rir_pred = model(head_params)
                rir_pred = model(model_input)
            else:
                rir_pred = model(convolved_audio, model_input)

            mse = torch.nn.functional.mse_loss(rir_pred, rir_true).item()

            with torch.enable_grad():
                try:
                    head_param = head_params[0]  
                    if args.collocation_w_reflections:
                        test_points = targeted_collocation_with_reflections(head_param, 0.0875,  device=device)
                    else:
                        test_points = targeted_collocation_points(head_param, 0.0875, device=device)
                    test_points.requires_grad_(True)

                    physics_loss = model.loss_PDE(test_points, head_param).item()
                except Exception as e:
                    print(f"physics computation failed: {e}")
                    print(f"Head params shape: {head_params.shape}, values: {head_params}")
                    physics_loss = 0.0
            total_mse += mse
            total_physics_loss += physics_loss
            total_samples += 1

    avg_mse = total_mse / total_samples
    avg_physics_loss = total_physics_loss / total_samples
    print(f"Average physics loss on test set: {avg_physics_loss}")
    print(f"Average MSE on test set: {avg_mse}")
