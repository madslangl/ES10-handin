import torch
from torch import nn
import torch.nn.functional as F
import torch.autograd as autograd

from auxfunctions import scale_t

class simple_pinn(nn.Module):
    """ CORRESPONDS TO MODEL 1 IN REPORT.
    A simple network which takes RIRs instead of convolved speech in order to focus on physics propagation.
    Used to test whether the PINN can learn the physics of wave propagation. """
    
    def __init__(self, rir_length, hidden_dim=64, feature_dim=64, pinn_hidden_dim=64, 
                pinn_num_hidden=3, c=343.0, mono=False, room_dims=True, laptop_pos=False):
        
        super(simple_pinn,self).__init__()
        self.rir_length = int(rir_length)
        self.c = c
        self.mono = mono
        self.room_dims = room_dims
        self.laptop_pos = laptop_pos

        self.collocation_encoder = nn.Linear(4, feature_dim)  # x,y,z,t
        # self.head_encoder = nn.Sequential(
        #     # if room_dims is false, the model only takes x,y,z, azimuth, elevation as input, so no prior knowledge of the room is given. 
        #     # useful for testing the physics of the model, and whether it can learn the direct sound path propagation based only on relative distance.
        #     nn.Linear(5 if not self.room_dims else (8 if not self.laptop_pos else 11), 64),
        #     # nn.Linear(5 if not self.room_dims else 8, 64), # [x,y,z, azimuth, elevation, room_x, room_y, room_z] 
        #     nn.Tanh(),
        #     nn.Linear(64, feature_dim)
        #     )
        
        
        self.audio_weight = nn.Parameter(torch.ones(1))
        self.head_weight = nn.Parameter(torch.ones(1))

        self.spatial_encoder = nn.Sequential(
            nn.Linear(5 if not self.room_dims else (8 if not self.laptop_pos else 11), 64),# [x,y,z, azimuth, elevation, room_x, room_y, room_z, laptop_x, laptop_y, laptop_z]
            nn.Tanh(),
            nn.Linear(64, feature_dim)
        )
        
        self.pinn_input = nn.Linear(feature_dim, pinn_hidden_dim)
        self.pinn_hidden = nn.ModuleList()
        


        self.pinn_hidden.append(nn.Linear(pinn_hidden_dim, pinn_hidden_dim))
        for i in range(1, pinn_num_hidden):
            self.pinn_hidden.append(nn.Linear(pinn_hidden_dim, pinn_hidden_dim))
        
        if mono:
            self.pinn_output = nn.Linear(pinn_hidden_dim, self.rir_length)
        else:
            self.pinn_output = nn.Sequential(
                nn.Linear(pinn_hidden_dim, pinn_hidden_dim*2),
                nn.Tanh(),
                nn.Linear(pinn_hidden_dim*2, pinn_hidden_dim*4),
                nn.Tanh(), # consider changing to leakyrelu. only problem - messes with physics loss due to non differentiability.
                nn.Linear(pinn_hidden_dim*4, self.rir_length*2)
            )
        self.activation = nn.Tanh()
        # self.activation = nn.LeakyReLU()  # #TODO leaky relu could probably catch spikes better. is it possible to implement despite issues with differentiabilyiy? 



    def forward(self, spatial_params):
        x = self.spatial_encoder(spatial_params)
        
        # process through PINN with skip connections
        x = self.activation(self.pinn_input(x))
        for layer in self.pinn_hidden:
            residual = x
            x = self.activation(layer(x)) + residual
        
        rir = self.pinn_output(x)
        
        if not self.mono:
            # reshape the batch × 1000 output to [batch, 2, rir_length]
            batch_size = spatial_params.shape[0]
            rir = rir.reshape(batch_size, 2, -1)
            
        return rir
    
    def forward_spatial(self, features):
        """Process spatial features through PINN layers for physics calculations."""
        x = self.activation(self.pinn_input(features))
        
        for layer in self.pinn_hidden:
            residual = x
            x = self.activation(layer(x)) + residual
            
        return self.pinn_output(x)

    def loss_PDE(self, collocation_points, head_params):
        """Physics-informed loss based on the wave equation.
        
        Args:
            collocation_points: Points in space-time to evaluate PDE
            head_params: 5D tensor with [x, y, z, azimuth, elevation]
        """
        device = collocation_points.device
        
        # Scale time dimension for wave equation
        points = collocation_points.clone().detach()
        points = scale_t(points, dim=4, c=self.c)
        points.requires_grad_(True)
        
        # Forward pass through spatial part of network
        spatial_features = self.collocation_encoder(points)
        # p_pred = self.forward_spatial(spatial_features)
        x = self.activation(self.pinn_input(spatial_features))
        
        for layer in self.pinn_hidden:
            residual = x
            x = self.activation(layer(x)) + residual
        
        # Handle the sequential output layer for physics calculation
        if isinstance(self.pinn_output, nn.Sequential):
            for layer in self.pinn_output:
                x = layer(x)
            p_pred = x
        else:
            p_pred = self.pinn_output(x)
        
        if not self.mono:
            p_pred = p_pred.reshape(p_pred.shape[0], 2, -1)
            # Use just one channel for physics (or average both)
            p_pred = p_pred[:, 0, :]


        
        # derivatives for wave equation
        grad_x = torch.autograd.grad(p_pred.sum(), points, create_graph=True)[0][:, 0:1]
        grad_y = torch.autograd.grad(p_pred.sum(), points, create_graph=True)[0][:, 1:2]
        grad_z = torch.autograd.grad(p_pred.sum(), points, create_graph=True)[0][:, 2:3]
        grad_t = torch.autograd.grad(p_pred.sum(), points, create_graph=True)[0][:, 3:4]
        
        # Second derivatives
        grad_xx = torch.autograd.grad(grad_x.sum(), points, create_graph=True)[0][:, 0:1]
        grad_yy = torch.autograd.grad(grad_y.sum(), points, create_graph=True)[0][:, 1:2]
        grad_zz = torch.autograd.grad(grad_z.sum(), points, create_graph=True)[0][:, 2:3]
        grad_tt = torch.autograd.grad(grad_t.sum(), points, create_graph=True)[0][:, 3:4]
        
        #  3D wave equation residual
        residual = grad_xx + grad_yy + grad_zz - grad_tt
        
        return torch.mean(residual ** 2)
    
# -------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------
# The following two models are not used in the report, but are kept here for reference and potential future use.
# -------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------

class FEEB_PINN(nn.Module):
    ## TODO: consideration: look into what other info i can extract - does it make sense to use tdoa to estimate head rotation?
    """ Feature-Extraction Based PINN for room impulse response estimation. Extracts features from audio input using an LSTM and then uses a PINN (FCN with skip connections) to estimate the RIR.
    Arguments:
    audio_dim (int): Dimension of audio input (default: 2, stereo)
    hidden_dim (int): Dimension of hidden states in LSTM (default: 64)
    feature_dim (int): Dimension of extracted features (default: 128)
    rir_length (int): Length of the RIR (default: 48e3, 1 second at 48kHz)
    pinn_hidden_dim (int): Dimension of hidden layers in the PINN (default: 50)
    pinn_num_hidden (int): Number of hidden layers in the PINN (default: 3)
    chunk_size (int): Chunk size for processing audio (default: 64)
    activation (torch.nn.Module): Activation function to use (default: Tanh)
    """
    def __init__(self, rir_length, audio_dim=2, hidden_dim=64, feature_dim=128,
                  pinn_hidden_dim=64, pinn_num_hidden=3,
                  c=343.0, activation=None, mono=False):
        # TODO: add parser for audio_dim, so it can take in both mono and multichannel as well
        super(FEEB_PINN, self).__init__()
        self.rir_length = rir_length 
        self.c = c
        self.mono = mono
        
        self.collocation_encoder = nn.Linear(4, feature_dim)  # x,y,z,t

        self.feature_extractor = nn.LSTM(
            input_size=audio_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )

        self.projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, feature_dim), 
            nn.LeakyReLU()
        )

        self.head_encoder = nn.Sequential(
            nn.Linear(3, 64),  # 6 parameters: x,y,z, azimuth, elevation, distance
            nn.Tanh(),
            nn.Linear(64, feature_dim)
        )
     
        self.audio_weight = nn.Parameter(torch.ones(1))
        self.head_weight = nn.Parameter(torch.ones(1))

        self.pinn_input = nn.Linear(feature_dim, pinn_hidden_dim)
        self.pinn_hidden = nn.ModuleList()
        
        # create pinnum_hidden-1 layers
        self.pinn_hidden.append(nn.Linear(pinn_hidden_dim, pinn_hidden_dim))
        
        for i in range(1,pinn_num_hidden):
            self.pinn_hidden.append(nn.Linear(pinn_hidden_dim,pinn_hidden_dim))
        
        if mono:
            self.pinn_output = nn.Linear(pinn_hidden_dim,int(rir_length))
        else: 
            self.pinn_output = nn.Linear(pinn_hidden_dim, int(rir_length) * 2)

        self.activation = nn.Tanh() if activation is None else activation
        
    def forward(self, audio, head_params):
        """ Forward pass with skip conncetions (takes spatial params as input together with audio features)
            """
        batch_size = audio.shape[0]
        
        if audio.dim() == 2:
            audio = audio.unsqueeze(1) # mapping to [batch, seq, features] if input is [batch, features]
        n_channels = audio.shape[1]

        # if self.feature_extractor.input_size != n_channels:
        #     raise ValueError(f"Expected {self.feature_extractor.input_size} channels, got {n_channels}")
        
        #TODO: explain the lstm error that occurs when the input is not in the right format (not contiguous)
        # This is a bottleneck. Investigate possible solutions so its not necessary to move the tensor to the cpu.

        # lstm_out, _ = self.feature_extractor(audio.transpose(1, 2).contiguous())  # [batch, seq, features]
        # lstm_out, _ = self.feature_extractor(audio.permute(0, 2, 1).contiguous())
        
        # the above gave non-contiguous tensor error, so i used the following instead, where i moved to cpu:
        batch_size, channels, seq_len = audio.shape

        cpu_lstm = self.feature_extractor.cpu()
        audio_cpu = audio.cpu()

        audio_reshaped = audio_cpu.permute(0, 2, 1).contiguous()
        lstm_out, _ = cpu_lstm(audio_reshaped)

        lstm_out = lstm_out.to(audio.device).detach()
        self.feature_extractor = self.feature_extractor.to(audio.device)

        # Continue processing on GPU
        audio_features = self.projection(lstm_out[:, -1])   

        head_features = self.head_encoder(head_params)  # [batch, features]

        features = self.audio_weight * audio_features + self.head_weight * head_features
        x = features
        x = self.activation(self.pinn_input(x))

        # w. skip connections :
        for layer in self.pinn_hidden:
            residual = x
            x = self.activation(layer(x)) + residual

        rir = self.pinn_output(x)

        # TODO: unsure about this implementation
        if self.mono:
            return rir
        else:
            rir = rir.reshape(batch_size, 2, -1)
            return rir

    def loss_PDE(self, collocation_points, head_params):
        device = collocation_points.device
        
        points = collocation_points.clone()
        points = scale_t(points, dim=4, c=self.c)
        points.requires_grad_(True)

        point_features = self.collocation_encoder(points)
        
        head_features = self.head_encoder(head_params.unsqueeze(0))  # [1, features]
    
        # combines features via learned weights
        features = self.audio_weight * point_features + self.head_weight * head_features
        
        x = self.activation(self.pinn_input(features))
        for layer in self.pinn_hidden:
            residual = x
            x = self.activation(layer(x)) + residual
        
        p_pred = self.pinn_output(x).squeeze(0)

        try:
            # had some issues here, so i added try except to catch specific physics erros 
            # TODO: clean code, unused functions and classes could be deleted if experiments show good results. 
            grad_p = torch.autograd.grad(
                p_pred.sum(), points, create_graph=True, allow_unused=True
            )[0]
            
            if grad_p is None:
                print("No gradients found between prediction and spatial coordinates")
                return torch.tensor(0.0, device=device)
            
            # second derivs, x, y, z, t
            grad_x = grad_p[:, 0:1]
            grad_xx = torch.autograd.grad(
                grad_x.sum(), points, create_graph=True, allow_unused=True
            )[0][:, 0:1]
            
            grad_y = grad_p[:, 1:2]
            grad_yy = torch.autograd.grad(
                grad_y.sum(), points, create_graph=True, allow_unused=True
            )[0][:, 1:2]
            
            grad_z = grad_p[:, 2:3]
            grad_zz = torch.autograd.grad(
                grad_z.sum(), points, create_graph=True, allow_unused=True
            )[0][:, 2:3]
            
            grad_t = grad_p[:, 3:4]
            grad_tt = torch.autograd.grad(
                grad_t.sum(), points, create_graph=True, allow_unused=True
            )[0][:, 3:4]
            
            # wave eq residual: ∇²p - (1/c²)∂²p/∂t²
            # scaled t, so (see scale_t func for description): ∇²p - ∂²p/∂t²
            residual = grad_xx + grad_yy + grad_zz - grad_tt
            
            return torch.mean(residual**2)
            
        except Exception as e:
            print(f"Error in physics computation: {e}")
            return torch.tensor(0.0, device=device)


    
class simple_lstm_pinn(nn.Module):
    """ Simple network which takes RIRs instead of convolved speech in order to focus on physics propagation.
    There are plenty of established frameworks for dereverberation (which also works the other way around), so the focus on rir-extraction from reverbed voice is less relevant
    As opposed to simple_pinn, this network uses an LSTM to extract features from the audio input so it hopefully can use the temporal information in the RIRs to learn better.
    """


    def __init__(self, rir_length, hidden_dim=64, feature_dim=64, pinn_hidden_dim=64, lstm_hidden=64,
                pinn_num_hidden=3, c=343.0, mono=False, room_dims=True,laptop_pos=False):
        
        super(simple_lstm_pinn,self).__init__()
        self.rir_length = int(rir_length)
        self.c = c
        self.mono = mono
        self.room_dims = room_dims
        self.laptop_pos = laptop_pos

        self.projection = nn.Linear(128, 64) # layer for matching the dim of residual connection  
        
        self.collocation_encoder = nn.Linear(4, feature_dim)  # x,y,z,t
        # self.head_encoder = nn.Sequential(
        #     # nn.Linear(3, 64), 
        #     # nn.Linear(5 if not self.room_dims else 8, 64), # [x,y,z, azimuth, elevation, room_x, room_y, room_z] 
        #     nn.Linear(5 if not self.room_dims else (8 if not self.laptop_pos else 11), 64),# [x,y,z, azimuth, elevation, room_x, room_y, room_z, laptop_x, laptop_y, laptop_z]
        #     nn.Tanh(),
        #     nn.Linear(64, feature_dim)
        #     )
        
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=lstm_hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )

        self.lstm_projection = nn.Linear(lstm_hidden * 2, feature_dim)

        self.audio_weight = nn.Parameter(torch.ones(1))
        # self.head_weight = nn.Parameter(torch.ones(1))

        self.spatial_encoder = nn.Sequential(
            # nn.Linear(3, 64), # distance, azimuth, elevation.
            nn.Linear(5 if not self.room_dims else (8 if not self.laptop_pos else 11), 64),# [x,y,z, azimuth, elevation, room_x, room_y, room_z, laptop_x, laptop_y, laptop_z]
            nn.Tanh(),
            nn.Linear(64, feature_dim)
        )

        self.pinn_input = nn.Linear(feature_dim, pinn_hidden_dim)
        self.pinn_hidden = nn.ModuleList()
        
        self.pinn_hidden.append(nn.Linear(pinn_hidden_dim, pinn_hidden_dim*2))
        self.pinn_hidden.append(nn.Linear(pinn_hidden_dim*2, pinn_hidden_dim*2))
        self.pinn_hidden.append(nn.Linear(pinn_hidden_dim*2, pinn_hidden_dim))
        self.pinn_hidden.append(nn.Linear(pinn_hidden_dim, pinn_hidden_dim))
        
        if mono:
            self.pinn_output = nn.Linear(pinn_hidden_dim, self.rir_length)
        else:
            self.pinn_output = nn.Linear(pinn_hidden_dim, self.rir_length * 2)
            
        self.activation = nn.Tanh()



    def forward(self, spatial_params):
        """ Forward pass with skip conncetions (takes spatial params as input together with audio features)
        NTS: pay attention to the input format of the LSTM. It needs to be [batch, seq_len, features]
            """        
        x = self.spatial_encoder(spatial_params)  # [batch, feature_dim]
        
        # Reshape for LSTM - create a sequence by repeating
        # We need a sequence for LSTM to process
        batch_size = spatial_params.shape[0]
        seq_len = 10  # Pick a reasonable sequence length
        x = x.unsqueeze(1).repeat(1, seq_len, 1)  # [batch, seq_len, feature_dim]
        
        # Process through LSTM
        lstm_out, _ = self.lstm(x)  # [batch, seq_len, lstm_hidden*2] (bidirectional)
        
        # Use last output of LSTM 
        lstm_features = self.lstm_projection(lstm_out[:, -1, :])        
        
        # Process through PINN with skip connections
        x = self.activation(self.pinn_input(lstm_features))
        
        for layer in self.pinn_hidden:
            residual = x
            # x = self.activation(layer(x)) + residual
            # x = self.projection(self.activation(layer(x))) + residual
            activated = self.activation(layer(x))
    
            # Project only if dimensions don't match
            if activated.shape[1] != residual.shape[1]:
                x = self.projection(activated) + residual
            else:
                x = activated + residual
                
        # Decode to RIR
        rir = self.pinn_output(x)
        
        if not self.mono:
            rir = rir.reshape(batch_size, 2, -1)
            
        return rir
    

    def forward_spatial(self, features):
        """Process spatial features through PINN layers for physics calculations."""
        x = self.activation(self.pinn_input(features))
        
        for layer in self.pinn_hidden:
            residual = x
            x = self.activation(layer(x)) + residual
            
        return self.pinn_output(x)

    def loss_PDE(self, collocation_points, head_params):
        """Physics-informed loss based on the wave equation.
        
        Args:
            collocation_points: Points in space-time to evaluate PDE
            head_params: 5D tensor with [x, y, z, azimuth, elevation]
        """
        device = collocation_points.device
        
        # Scale time dimension for wave equation
        points = collocation_points.clone().detach()
        points = scale_t(points, dim=4, c=self.c)
        points.requires_grad_(True)
        
        # Forward pass through spatial part of network
        spatial_features = self.collocation_encoder(points)
        p_pred = self.forward_spatial(spatial_features)
        
        # Calculate derivatives for wave equation
        grad_x = torch.autograd.grad(p_pred.sum(), points, create_graph=True)[0][:, 0:1]
        grad_y = torch.autograd.grad(p_pred.sum(), points, create_graph=True)[0][:, 1:2]
        grad_z = torch.autograd.grad(p_pred.sum(), points, create_graph=True)[0][:, 2:3]
        grad_t = torch.autograd.grad(p_pred.sum(), points, create_graph=True)[0][:, 3:4]
        
        # Second derivatives
        grad_xx = torch.autograd.grad(grad_x.sum(), points, create_graph=True)[0][:, 0:1]
        grad_yy = torch.autograd.grad(grad_y.sum(), points, create_graph=True)[0][:, 1:2]
        grad_zz = torch.autograd.grad(grad_z.sum(), points, create_graph=True)[0][:, 2:3]
        grad_tt = torch.autograd.grad(grad_t.sum(), points, create_graph=True)[0][:, 3:4]
        
        # Complete 3D wave equation residual
        residual = grad_xx + grad_yy + grad_zz - grad_tt
        
        return torch.mean(residual ** 2)