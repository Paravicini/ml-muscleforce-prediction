import torch
import torch.nn as nn


class MCEFNN(nn.Module):  # Multimodal Convolutional Encoded Features Neural Network (MCEFNN)
    def __init__(self, image_encoder, emg_encoder):
        super(MCEFNN, self).__init__()
        self.signal_encoder = emg_encoder
        self.image_encoder = image_encoder
        self.flatt = nn.Flatten()
        self.out_layer = nn.Sequential(
            nn.Linear(image_encoder.latent_size + emg_encoder.latent_size, 1),
            #nn.LeakyReLU(),
            #nn.Linear(8, 1)
        )

    def forward(self, emg, image):
        latent_emg = self.signal_encoder(emg)
        latent_emg = self.flatt(latent_emg)
        latent_img = self.image_encoder(image)
        latent_img = self.flatt(latent_img)
        combined_representation = torch.cat((latent_img, latent_emg), dim=1)
        output = self.out_layer(combined_representation)
        return output
