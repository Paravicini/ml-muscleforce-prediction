from torch import nn


class EMGEncoder(nn.Module):
    def __init__(self):
        super(EMGEncoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=1, kernel_size=33, stride=2, padding=16),
            nn.LeakyReLU(),
            nn.BatchNorm1d(1),
            nn.Conv1d(in_channels=1, out_channels=1, kernel_size=7, stride=2, padding=6),
            nn.LeakyReLU(),
            nn.BatchNorm1d(1),
            nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm1d(1),

        )
        self.latent_size = None

    def forward(self, x):
        encoded = self.encoder(x)
        self.latent_size = encoded.size(1) * encoded.size(2)
        return encoded
