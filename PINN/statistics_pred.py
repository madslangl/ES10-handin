import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm

from dataloader import RIRDataset
from auxfunctions import create_compatible_params, analyse_reflection_accuracy
from network import simple_pinn
from parser import get_parser, get_name


"""This script loads a trained model and evaluates it on a test dataset, computing statistics on the predicted RIRs.
For single RIR predictions, use the predict_rir_local.py script instead."""

# for local runs, run in terminal:
#python statistics_pred.py" --models="examples\more-layers-alpha05simple_pinn_step_27000.pth" --rir_path="<RIR_PATH>" --split="test" --model="simple_pinn" --rir_length=500 --room_dims --laptop_pos
#

parser = get_parser()
args = parser.parse_args()

model_name = os.path.basename(args.model_path).split('.')[0]
output = r"PINN\outputs"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

print(f"Loading test dataset from {args.rir_path}")
test_dataset = RIRDataset(
    rir_path=args.rir_path,
    speech_path=args.speech_path,
    split='test'
    # max_samples=args.max_samples if args.max_samples > 0 else None
)

test_dataset.room_dims = args.room_dims
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
print(f"Test dataset loaded with {len(test_dataset)} samples")

print(f"Loading model from {args.model_path}")
if args.model == 'simple_pinn':
    model = simple_pinn(
        hidden_dim=64, 
        feature_dim=128,
        pinn_hidden_dim=64,
        pinn_num_hidden=4,
        rir_length=args.rir_length,
        mono=args.mono,
        room_dims=args.room_dims,
        laptop_pos=args.laptop_pos
    )
    
else:
    raise ValueError(f"Unsupported model type: {args.model}")



model.load_state_dict(torch.load(args.model_path, map_location=device))
model.to(device)
model.eval()
print("Model loaded successfully")

print("Processing test set...")
all_preds = []
all_targets = []

with torch.no_grad():
    for batch_idx, batch_data in tqdm(enumerate(test_loader), total=len(test_loader), desc="Processing batches"):
        # if batch_idx % 10 == 0:
            # print(f"Processing batch {batch_idx}/{len(test_loader)}")
        
        convolved_audio = batch_data[0].to(device)
        head_params = batch_data[1].to(device)
        rir_true = batch_data[2].to(device)
        
        model_input = create_compatible_params(head_params)
        if args.model == 'simple_pinn' or args.model == 'simple_lstm_pinn':
            rir_pred = model(model_input)
        else:
            rir_pred = model(convolved_audio, model_input)
            
        all_preds.append(rir_pred.cpu())
        all_targets.append(rir_true.cpu())

# Concatenate all batches
all_preds = torch.cat(all_preds, dim=0)
all_targets = torch.cat(all_targets, dim=0)
print(f"Total samples processed: {all_preds.shape[0]}")

print("Computing statistics...")

print(f"Total samples to analyze: {all_preds.shape[0]}") 

# Now run analysis on the full dataset
timing_results = analyse_reflection_accuracy(
    all_preds,
    all_targets,
    model_name=model_name
)


mse = torch.mean((all_preds - all_targets)**2).item()

early_energy_true = torch.sum(all_targets[:, :100]**2, dim=1)
early_energy_pred = torch.sum(all_preds[:, :100]**2, dim=1)
early_energy_diff = torch.mean(torch.abs(early_energy_pred - early_energy_true) / early_energy_true).item()

late_energy_true = torch.sum(all_targets[:, 300:]**2, dim=1)
late_energy_pred = torch.sum(all_preds[:, 300:]**2, dim=1)
late_energy_diff = torch.mean(torch.abs(late_energy_pred - late_energy_true) / late_energy_true).item()

#saving:

results_file = os.path.join(output, f"stats_{model_name}.txt")

with open(results_file, 'w') as f:
    f.write(f"Statistics for model: {model_name}\n")
    f.write(f"Dataset: {args.rir_path}\n")
    f.write(f"Number of samples: {all_preds.shape[0]}\n\n")
    
    f.write("-- Direct Sound Timing Analysis --\n")
    f.write(f"Mean sample error: {timing_results['mean_sample_error']} samples\n")
    f.write(f"Variance: {timing_results['var_sample_error']} samples\n")
    f.write(f"Mean time error: {timing_results['mean_time_error_ms']} ms\n\n")

    f.write(f"Mean Squared Error: {mse}\n\n")
    
    f.write("-- Energy Analysis --\n")
    f.write(f"Early reflection energy difference: {early_energy_diff}\n")
    f.write(f"Late reflection energy difference: {late_energy_diff}\n")

print(f"Results saved to {results_file}")
print("\nSummary of Results:")
print(f"- Direct sound timing error: {timing_results['mean_time_error_ms']} ms")
print(f"- Early reflection energy difference: {early_energy_diff}")
print(f"- Late reflection energy difference: {late_energy_diff}")

# Additional visualization: Sample waveform comparison
print("Generating example waveform plots...")
num_examples = min(5, all_preds.shape[0])

plt.figure(figsize=(15, 10))
for i in range(num_examples):
    plt.subplot(num_examples, 1, i+1)
    plt.plot(all_targets[i].numpy(), 'r-', label='Ground Truth')
    plt.plot(all_preds[i].numpy(), 'b-', label='Predicted', alpha=0.7)
    
    if i == 0:
        plt.legend()
    
    plt.title(f"Sample {i}")
    plt.grid(True)

plt.tight_layout()
waveform_plot = os.path.join(output, f"waveform_comparison_{model_name}.pdf")
plt.savefig(waveform_plot)
print(f"Waveform comparison saved to {waveform_plot}")
