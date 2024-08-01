from torch import nn


class ImageDecoder(nn.Module):
    def __init__(self, stripes=False):  # mode = 'full_img' or 'stripes'
        super(ImageDecoder, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(in_features=100, out_features=24 * 7 * 7),
            nn.ReLU(),
        )
        if not stripes:
            self.Tconv_frame_1 = nn.Sequential(
                nn.ConvTranspose2d(in_channels=2, out_channels=2, kernel_size=(12, 12), stride=2, padding=1,
                                   output_padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(2),
                nn.ConvTranspose2d(in_channels=2, out_channels=2, kernel_size=(16, 16), stride=2, padding=1,
                                   output_padding=0),
                nn.ReLU(),
                nn.BatchNorm2d(2),
                nn.ConvTranspose2d(in_channels=2, out_channels=2, kernel_size=(24, 24), stride=2, padding=1,
                                   output_padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(2),
                nn.ConvTranspose2d(in_channels=2, out_channels=1, kernel_size=(33, 33), stride=2, padding=17,
                                   output_padding=1),
            )
        elif stripes:
            self.Tconv_frame_1 = nn.Sequential(
                nn.ConvTranspose2d(in_channels=2, out_channels=2, kernel_size=(5, 2), stride=2, padding=1,
                                   output_padding=1),
                nn.LeakyReLU(),
                nn.BatchNorm2d(2),
                nn.ConvTranspose2d(in_channels=2, out_channels=2, kernel_size=(8, 3), stride=2, padding=1,
                                   output_padding=1),
                nn.LeakyReLU(),
                nn.BatchNorm2d(2),
                nn.ConvTranspose2d(in_channels=2, out_channels=2, kernel_size=(11, 5), stride=2, padding=1,
                                   output_padding=1),
                nn.LeakyReLU(),
                nn.BatchNorm2d(2),
                nn.ConvTranspose2d(in_channels=2, out_channels=1, kernel_size=(15, 7), stride=2, padding=1,
                                   output_padding=1),
                nn.LeakyReLU(),
                nn.BatchNorm2d(1),

            )

    def forward(self, x):
        x = self.Tconv_frame_1(x)
        return x


