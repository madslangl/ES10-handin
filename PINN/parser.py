import argparse
from pathlib import Path

def get_parser():
    """Parser for network training and evaluation"""
    parser = argparse.ArgumentParser("PINN", description="Train and evaluate PINN model for RIR prediction")

    parser.add_argument('--model', type=str, default='simple_pinn', choices=[ 'simple_pinn','feeb_pinn','simple_lstm_pinn'])
    parser.add_argument('--collocation_w_reflections', action='store_true', default=False, help='Use collocation points with reflections for training')
    parser.add_argument('--rir_length', type=int, default=500, help="Portion of rir to be loaded in samples. SR is 16e3 unless otherwise specified.")
    parser.add_argument("--debugging", action="store_true", default=False, help="Debugging mode. Use a small subset of data and limited epochs.")
    parser.add_argument("--room_dims", action="store_true", default=False, help="pass room dimensions as input to the model.")
    parser.add_argument("--laptop_pos", action="store_true", default=False, help="Use laptop position as input to the model.")
    parser.add_argument("--dummy", help="Dummy param for creating a new checkpoint file.")
    

    parser.add_argument('--model_path', type=Path, help="Used when testing a model.")
    parser.add_argument('--sample_rate', type=float, default=16e3)

    #paths:
    parser.add_argument("--rir_path", type=str, required=True, help="Path to the RIR dataset (.h5 file)")
    parser.add_argument("--speech_path", type=str, required=False, help="Path to the speech dataset")

    parser.add_argument("--split", type=str, choices=["train", "valid", "test"], required=True, help="Data split to use") 

    # loss related:
    parser.add_argument("--alpha", type=float, default=0.5, help="Weighting factor for the physics loss")
    parser.add_argument("--loss", type=str, default="mse", choices=["mse", "weighted_mse","reflection_aware"], help="Data loss function to use")
    parser.add_argument("--weight_decay", type=float, default=0.7, help="Decay rate for weighted loss function")

    #saving:
    parser.add_argument("--models", type=Path, default=Path("models"),help="Folder where to store trained models")
    parser.add_argument("--checkpoints",type=Path,default=Path("checkpoints"),help="Folder where to store checkpoints etc")
    parser.add_argument("--logs", type=Path, default=Path("logs"), help="Folder where to store logs")
    parser.add_argument("--save_often", action="store_true", default=False, help="Save model and checkpoint often during training.")
    
    #training related:
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to train")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for the optimizer")
    parser.add_argument("--wandb", action="store_true", help="Log the loss to Weights and Biases")
    parser.add_argument("--mono", action="store_true", default=False, help="Use mono audio instead of stereo. Error in implementation made this relevant.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility. Default = 42.")
    return parser


def get_name(parser, args):
    """Naming scheme based on parser parameters with better path handling."""
    import os
    ignore_args = set(["split","checkpoints", "models", "wandb", "logs", "rir_path", "speech_path"])

    parts = []
    name_args = dict(args.__dict__)
    for name, value in name_args.items():
        if name in ignore_args:
            continue
        if value != parser.get_default(name):
            if isinstance(value, Path):
                parts.append(f"{name}={value.name}")
            elif isinstance(value, str) and ('/' in value or '\\' in value):
                basename = os.path.basename(value)
                parts.append(f"{name}={basename}")
            else:
                parts.append(f"{name}={value}")
    
    if parts:
        name = "_".join(parts)
    else:
        name = "default"
        
    if len(name) > 90:
        name = name[:90]
    
    return name