import torch
import torch.nn as nn
import torch.optim as optim

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        # Define the convolutional layers of AlexNet
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),  # Conv1
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                  # Pool1
            
            nn.Conv2d(64, 192, kernel_size=5, padding=2),          # Conv2
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                  # Pool2
            
            nn.Conv2d(192, 384, kernel_size=3, padding=1),         # Conv3
            nn.ReLU(inplace=True),
            
            nn.Conv2d(384, 256, kernel_size=3, padding=1),         # Conv4
            nn.ReLU(inplace=True),
            
            nn.Conv2d(256, 256, kernel_size=3, padding=1),         # Conv5
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)                  # Pool5
        )
        
        # Fully connected layer with 512 neurons
        self.fc = nn.Linear(256 * 6 * 6, 512)
        
    def forward(self, x):
        x = self.features(x)  # Extract features
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)  # Pass through the fully connected layer
        return x  # Output shape: [B, 512]

class AlexNetLSTM(nn.Module):
    def __init__(self, num_classes=25, hidden_size=256, num_layers=2):
        super(AlexNetLSTM, self).__init__()
     
        self.alexnet = AlexNet()
        
        self.lstm = nn.LSTM(
            input_size=512,  # Output size of the FC layer in AlexNet
            hidden_size=hidden_size,  # Hidden size of LSTM
            num_layers=num_layers,   # Number of LSTM layers
            batch_first=True         # Batch comes first in input
        )
        
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        """
        x: Input video data with shape [B, C, N, H, W]
        B: Batch size
        C: Number of channels
        N: Number of frames
        H, W: Height and width of each frame
        """
        batch_size, channels, num_frames, height, width = x.size()
        
        features = []
        for t in range(num_frames):
            frame = x[:, :, t, :, :]  # Extract the t-th frame, shape [B, C, H, W]
            feature = self.alexnet(frame)  # Extract features using AlexNet
            features.append(feature)  # Shape [B, 512]
        
        features = torch.stack(features, dim=1)
        
        lstm_out, _ = self.lstm(features)  # Output shape [B, N, hidden_size]
        
        final_out = lstm_out[:, -1, :]  # Shape [B, hidden_size]
        
        logits = self.fc(final_out)  # Shape [B, num_classes]
        
        probs = torch.softmax(logits, dim=1)
        return probs

if __name__ == "__main__":
    num_classes = 25

    
    # Create the model
    model = AlexNetLSTM(num_classes=num_classes)
    print(model)
