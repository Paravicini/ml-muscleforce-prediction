from torch import nn


class EMGDecoder(nn.Module):
    def __init__(self):
        super(EMGDecoder, self).__init__()

        # Encoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(in_channels=1, out_channels=1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm1d(1),
            nn.ConvTranspose1d(in_channels=1, out_channels=1, kernel_size=7, stride=2, padding=6, output_padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm1d(1),
            nn.ConvTranspose1d(in_channels=1, out_channels=1, kernel_size=33, stride=2, padding=16),


        )

    def forward(self, x):
        x = self.decoder(x)
        return x