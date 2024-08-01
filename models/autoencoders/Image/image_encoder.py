from torch import nn


class ImageEncoder(nn.Module):
    def __init__(self, stripes=False):
        super(ImageEncoder, self).__init__()
        if stripes == False:
            self.conv_frame_1 = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(33, 33), stride=2, padding=17),
                nn.ReLU(),
                # nn.BatchNorm2d(1),
                nn.Conv2d(in_channels=1, out_channels=2, kernel_size=(24, 24), stride=2, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(2),
                nn.Conv2d(in_channels=2, out_channels=2, kernel_size=(16, 16), stride=2, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(2),
                nn.Conv2d(in_channels=2, out_channels=2, kernel_size=(12, 12), stride=2, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(2), )
        elif stripes == True:
            self.conv_frame_1 = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=2, kernel_size=(15, 7), stride=2, padding=1),
                nn.LeakyReLU(),
                # nn.MaxPool2d(kernel_size=2, stride=2),
                nn.BatchNorm2d(2),
                nn.Conv2d(in_channels=2, out_channels=2, kernel_size=(11, 5), stride=2, padding=1),
                nn.LeakyReLU(),
                # nn.MaxPool2d(kernel_size=2, stride=2),
                nn.BatchNorm2d(2),
                nn.Conv2d(in_channels=2, out_channels=2, kernel_size=(9, 3), stride=2, padding=1),
                nn.LeakyReLU(),
                # nn.MaxPool2d(kernel_size=2, stride=2),
                nn.BatchNorm2d(2),
                nn.Conv2d(in_channels=2, out_channels=2, kernel_size=(5, 3), stride=2, padding=1),
                nn.LeakyReLU(),
                # nn.MaxPool2d(kernel_size=2, stride=2),
                nn.BatchNorm2d(2),

            )
            self.latent_size = None

    def forward(self, x):
        encoded = self.conv_frame_1(x)
        self.latent_size = encoded.size(1) * encoded.size(2) * encoded.size(3)
        return encoded


