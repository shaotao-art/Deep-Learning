import torch
import torch.nn as nn

class Disc(nn.Module):
    def __init__(self, img_channel):
        super(Disc, self).__init__()
        self.init_layer = nn.Sequential(
            nn.Conv2d(img_channel, 64, 3, 1, 1),
            nn.LeakyReLU()
        )
        self.conv_layers_config = [[64, 64, 3, 2, 1],
                                   [64, 128, 3, 1, 1],
                                   [128, 128, 3, 2, 1],
                                   [128, 256, 3, 1, 1],
                                   [256, 256, 3, 2, 1],
                                   [256, 512, 3, 1, 1],
                                   [512, 512, 3, 2, 1]]
        self.fcs = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 6 * 6, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )
        self.conv_layers = self._create_conv_layers()

    def _create_conv_layers(self):
        conv_layers = []
        for i in range(len(self.conv_layers_config)):
            conv_layers.append(self.conv_bn_leaky_relu(self.conv_layers_config[i][0],
                                                       self.conv_layers_config[i][1],
                                                       self.conv_layers_config[i][2],
                                                       self.conv_layers_config[i][3],
                                                       self.conv_layers_config[i][4],))
        return nn.Sequential(*conv_layers)

    def conv_bn_leaky_relu(self, inchannel, outchannel, k_s, s, p):
        return nn.Sequential(
            nn.Conv2d(inchannel, outchannel, k_s, s, p),
            nn.BatchNorm2d(outchannel),
            nn.LeakyReLU()
        )

    def forward(self, x):
        print(x.shape)
        x = self.init_layer(x)
        for layer in self.conv_layers:
            x = layer(x)
        x = self.fcs(x)
        print(x.shape)
        return x


class Gen(nn.Module):
    """
    input: (3, 24, 24)
    output: (3, 96, 96)

    network arch:
    https://production-media.paperswithcode.com/methods/Screen_Shot_2020-07-19_at_11.13.45_AM_zsF2pa7.png
    (3, 24, 24)
    (256, 24, 24) -> (64, 48, 48)
    (256， 48， 48) -> (64, 96, 96)
    (3, 96, 96)
    """
    def __init__(self, channel_img):
        super(Gen, self).__init__()
        # init block
        self.conv1 = nn.Conv2d(channel_img, 64, 9, 1, 4)
        self.Prelu = nn.PReLU()
        # conv block after residual block
        self.conv2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(64)
        # pixelshuffle to upscale img
        self.pixelshuffle_block = nn.Sequential(
            nn.Conv2d(64, 256, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.PReLU()
        )
        # residual block
        self.inter_block = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64)
        )
        # final conv block to make img_channel == 3
        self.final_conv = nn.Conv2d(64, 3, 9, 1, 4)

    def _res_block(self, x):
        return x + self.inter_block(x)

    def _pixel_shuffle_block(self, x):
        return self.pixelshuffle_block(x)

    def forward(self, x):
        init = self.Prelu(self.conv1(x))

        # num residual block == 5
        for i in range(5):
            x = self._res_block(init)
        x = self.bn2(self.conv2(x))
        x += init
        # num pixelshuffle layer == 2
        for i in range(2):
            x = self.pixelshuffle_block(x)
        x = self.final_conv(x)
        return x


if __name__ == "__main__":
    x = torch.randn(16, 3, 96, 96)
    gen = Disc(3)
    gen(x)
