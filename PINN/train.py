print("=== Script starting ===")
import torch
from torch.utils.data import DataLoader
from network import FEEB_PINN, simple_pinn, simple_lstm_pinn 
from dataloader import RIRDataset
# import os
from pathlib import Path
import glob
import matplotlib.pyplot as plt
from auxfunctions import targeted_collocation_points, validate, pad_collate, load_checkpoint, save_checkpoint, save_training_metrics, weigthed_mse_loss, reflection_aware_loss, targeted_collocation_with_reflections
from parser import get_parser, get_name

print("imports working")

# TODO: RIRPINN uses curriculum training learning (review: https://arxiv.org/abs/2101.10382). would that be relevant here?

parser = get_parser()
args = parser.parse_args()
model_name = get_name(parser,args)

checkpoint_dir = Path(args.checkpoints) / model_name
checkpoint_dir.mkdir(parents=True, exist_ok=True)


print(f"using model name: {model_name}")
print(f"checkpoints will be saved in: {checkpoint_dir}")

model_path = Path(args.models) / model_name # model_path and models in parser can be quite confusing. but models is to a folder,
# and model_path is used for loading specific models when testing 
model_path.mkdir(parents=True, exist_ok=True)

# anechoic speech path (using the EARS dataset. See dataloader for more.)
speech_path = args.speech_path

model_type = args.model # to be used for validation

# synthesised data:
rir_path = args.rir_path
rir_length = args.rir_length
batch_size = args.batch_size


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

# cuda debugging:
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"CUDA device name: {torch.cuda.get_device_name(0)}")


# models: Simple_pinn is the only one showing promising results, the others are still work in progress.
if args.model == 'simple_pinn':
    print("model  : simple_pinn")
    model = simple_pinn(
        hidden_dim=64,
        feature_dim=128,
        pinn_hidden_dim=64,
        pinn_num_hidden=4,
        rir_length=rir_length,
        mono=args.mono,
        room_dims=args.room_dims,
        laptop_pos = args.laptop_pos
    )

elif args.model == 'simple_lstm_pinn':
    print("model  : simple_lstm_pinn")
    model = simple_lstm_pinn(
        hidden_dim=64,
        feature_dim=128,
        pinn_hidden_dim=64,
        lstm_hidden=128,
        pinn_num_hidden=4,
        rir_length=rir_length,
        mono=args.mono,
        room_dims=args.room_dims,
        laptop_pos= args.laptop_pos
    )

else: 
    model = FEEB_PINN(
        audio_dim=2,
        hidden_dim=32,
        feature_dim=32,
        pinn_hidden_dim=32,
        pinn_num_hidden=2,
        rir_length=rir_length,
        mono=args.mono
    )

if args.wandb:
    import wandb
    print("init wandb")
    wandb.init(
        project = "es10",
        name = model_name,
        config={
            "rir_path": args.rir_path,
            "speech_path": args.speech_path,
            "alpha": args.alpha,
            "lr": args.learning_rate,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "model_type": type(model).__name__,
            "hidden_dim": model.hidden_dim,
            "feature_dim": model.feature_dim,
        })

# TODO: check why multiple gpus are not available from hpc side.
print("checking if multiple GPUs are available")
if torch.cuda.device_count() > 1:
    print(f"gpu count: {torch.cuda.device_count()}, parallel processing activated")
    model = torch.nn.DataParallel(model)
else: 
    print("...they were not..")

model = model.to(device)
torch.manual_seed(42)

global_step = 0
num_val_samples = 20

if args.debugging==True:
    print("debugging mode activated")
    max_samples = 100
else:
    max_samples = None

train_dataset = RIRDataset(
    rir_path=rir_path, 
    speech_path=None,
    split='train',
    max_samples = max_samples, # only for testing.
    )

valid_dataset = RIRDataset(
    rir_path=rir_path, 
    speech_path=None,
    split='valid',
)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_collate)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=pad_collate)


start_epoch = 0
train_loss = []

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
criterion = torch.nn.MSELoss()

# learning rate scheduler to avoid local minima
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='min', 
    factor=0.5, 
    patience=3, 
    verbose=True
)

# list checkpoint:
print("loading checkpoint if available")

checkpoint_pattern = Path(checkpoint_dir) / f"checkpoint_*.pt"
checkpoints = sorted(glob.glob(checkpoint_pattern))

if checkpoints: 
    latest_checkpoint = checkpoints[-1]
    start_epoch, train_loss = load_checkpoint(latest_checkpoint, model, optimizer)
    print(f"Checkpoint loaded, resuming from epoch {start_epoch+1}")
    print(f"Latest checkpoint: {latest_checkpoint}")
else:
    print("No checkpoints found, starting from epoch 1")


if args.debugging==True:
    num_epochs = 5
    print("debugging mode, only 5 epochs")
else:
    num_epochs = args.epochs
     
# hard coded head radius # TODO: consider changing this in future implementations, so the head size can be varied.
head_radius = 0.0875

data_losses = []
physics_losses = []
val_losses = []
val_data_losses = []
val_physics_losses = []

print(f"starting training for {num_epochs} epochs")
for epoch in range(start_epoch, num_epochs):
    model.train()
    running_loss = 0.0
    running_data_loss = 0.0
    running_physics_loss = 0.0

    for reverbed_speech, head_params, rir_laptop in train_loader:
        head_params = head_params.to(device)          # [batch, 3]
        rir_laptop = rir_laptop.to(device)            # [batch, N]
        
        # only use head params if its not using the reverberant speech
        if args.model == 'simple_pinn':
            pred = model(head_params)
        
        elif args.model == 'simple_lstm_pinn':
            pred = model(head_params)
        
        else:     
            reverbed_speech = reverbed_speech.to(device)  # [batch, 2, N]
            pred = model(reverbed_speech, head_params)    # [batch, N]
        
        # attempts at fixing the model by using different loss functions
        if args.loss == "weighted_mse":
            data_loss = weigthed_mse_loss(pred, rir_laptop)

        elif args.loss == "reflection_aware":
            data_loss = reflection_aware_loss(pred,rir_laptop,
                                              peak_height=0.1,
                                              peak_distance=3,
                                              peak_weight=10
                                              )
        
        elif args.loss == "mse":
            data_loss = criterion(pred, rir_laptop)
        
        # physics loss: collocation points on sphere for each sample in batch
        physics_loss = 0.0

        # last implementation of model. Sets collocation points at where reflections are.
        if args.collocation_w_reflections:
            for b in range(head_params.shape[0]):
                head_pos = head_params[b].detach()
                room_dims = head_pos[5:8] if not args.room_dims else None  # room dimensions
                
                # For direct sound only - high physics weight
                direct_collocation = targeted_collocation_points(head_pos, head_radius, device=device) 
                direct_physics = model.loss_PDE(direct_collocation, head_params[b])
                
                # Apply lower weight for reflections physics
                reflection_weight = 0.7 # maybe it shouldnt be that much lower. the amplitude of the direct sound will be relatively larger compared to the reflections. 
                physics_loss += direct_physics
                
                # only add reflection physics if room dimensions are provided
                if room_dims is not None:
                    reflection_collocation = targeted_collocation_with_reflections(head_pos, room_dims, head_radius, device=device)
                    reflection_physics = model.loss_PDE(reflection_collocation, head_params[b])
                    physics_loss += reflection_weight * reflection_physics

                else:
                    print("collocation_w_reflections is True, but no room dimensions provided. Skipping reflection physics loss.")

        else: # just using regular collocation points placed at ears.
            for b in range(head_params.shape[0]):
                head_pos = head_params[b].detach()
                
                collocation_points = targeted_collocation_points(head_pos, head_radius, mono=args.mono, device=device)
                
                physics_loss += model.loss_PDE(collocation_points, head_params[b])
            
        # average the physics loss over the batch:
        physics_loss = physics_loss / head_params.shape[0]

        loss = (1 - args.alpha) * data_loss + args.alpha * physics_loss 

        if args.wandb:
            wandb.log({
            "batch_loss": loss.item(),
            "batch_data_loss": data_loss.item(),
            "batch_physics_loss": physics_loss.item(),
            # "batch_direct_loss": direct_loss.item(),
            # "batch_late_decay_loss": late_decay_loss.item(),
            })

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if args.save_often==True:
            # saving both model and checkpoint pr. 1000 steps. 
            global_step += 1
            if global_step % 1000 == 0:
                checkpoint_path = checkpoint_dir / f"checkpoint_step_{global_step}.pt"
                save_checkpoint(epoch, model, optimizer, train_loss, str(checkpoint_path))
            
            # save model every 2000 steps
            elif global_step % 2000 == 0:
                model_save_path = model_path / f"{args.model}_step_{global_step}.pth"
                torch.save(model.state_dict(), str(model_save_path))

            # its a shame if the hpc time limit is reached and its been a while since the model saved, so i'd rather save a lot and delete unused after.

        running_loss += loss.item()
        running_data_loss += data_loss.item()
        running_physics_loss += physics_loss.item()

        # save checkpoints. should maybe be 5.
        if (epoch + 1) % 2 == 0:
            checkpoint_path = checkpoint_dir / f"checkpoint_{epoch+1}.pt"
            save_checkpoint(epoch, model, optimizer, train_loss, str(checkpoint_path))
            
            metrics_path = checkpoint_dir / f"metrics_{epoch+1}.csv"
            save_training_metrics(str(metrics_path), epoch+1, train_loss, data_losses, physics_losses)
                
            
    avg_loss = running_loss / len(train_loader)
    avg_data_loss = running_data_loss / len(train_loader)
    avg_physics_loss = running_physics_loss / len(train_loader)
    train_loss.append(avg_loss)
    print(f"Epoch [{epoch+1}/{num_epochs}], Avg Loss: {avg_loss:.6f}, "
          f"Data: {avg_data_loss:.6f}, Physics: {avg_physics_loss:.6f}")
    if args.wandb:
        wandb.log({ # type: ignore
            "epoch": epoch + 1,
            "avg_loss": avg_loss,
            "avg_data_loss": avg_data_loss, 
            "avg_physics_loss": avg_physics_loss,
        })
    
    data_losses.append(avg_data_loss)
    physics_losses.append(avg_physics_loss)


    model.eval()
    val_loss, val_data_loss, val_physics_loss = validate(
        model, valid_loader, criterion, device, head_radius, args.alpha, num_val_samples
    )
    
    val_losses.append(val_loss)
    val_data_losses.append(val_data_loss)
    val_physics_losses.append(val_physics_loss)
    
    # update the learning rate scheduler based on validation loss
    scheduler.step(val_loss)
    print(f"Validation - Loss: {val_loss:.6f}, Data Loss: {val_data_loss:.6f}, Physics Loss: {val_physics_loss:.6f}")
    
    if args.wandb:
        wandb.log({
            "val_loss": val_loss,
            "val_data_loss": val_data_loss,
            "val_physics_loss": val_physics_loss,
            "learning_rate": optimizer.param_groups[0]['lr'],
            "epoch": epoch + 1
        })
    
    model.train()



torch.save(model.state_dict(), model_path / f"{model_name}-final.pth")
print("done training, saving model")
metrics_path = model_path / f"{model_name}_metrics.csv"
save_training_metrics(str(metrics_path), range(1, num_epochs+1), train_loss, data_losses, physics_losses)


plt.figure(figsize=(10, 5))
plt.plot(range(1, len(train_loss) + 1), train_loss, label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss vs Epochs')
plt.legend()
plt.grid()
plt.savefig(model_path / f"{model_name}-loss.pdf")

if len(val_losses) > 0:
    val_metrics_path = model_path / f"{model_name}_val_metrics.csv"
    save_training_metrics(str(val_metrics_path), range(1, num_epochs+1), val_losses, val_data_losses, val_physics_losses)

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Validation Loss vs Epochs')
    plt.legend()
    plt.grid()
    plt.savefig(model_path / f"{model_name}-val_loss.pdf")

    if args.wandb:
        wandb.log({"final_val_loss_plot": wandb.Image(plt)})

if args.wandb:
    wandb.log({"final_loss_plot": wandb.Image(plt)})
    wandb.finish()