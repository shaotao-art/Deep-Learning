import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, dataloader
import torchvision.transforms as transforms

class Discriminator(nn.Module):
    def __init__(self,in_features) -> None:
        super().__init__()
        self.disc= nn.Sequential(
            nn.Linear(in_features),
            nn.LeakyRelu(0.01),
            nn.Linear(128,1),
            nn.Sigmoid(),
        )
    
    def forward(self,x):
        return self.desc(x)


class Generator(nn.Module):
    def __init__(self,z_dim,img_dim) -> None:
        super().__init__()
        self.gen = nn.Sequentail(
            nn.Linear(z_dim,256),
            nn.LeakyReLU(0.01),
            nn.Linear(256,img_dim),
            nn.Tanh(),
        )

    def forward(self,x):
        return self.gen(x)


device = 'cuda' if torch.cuda.is_available() else 'cpu'




disc = Discriminator(image_dim).to(device)
gen = Generator(z_dim,image_dim).to(device)

transforms = transforms.Compose([
    transforms.ToTensoe(),
    transforms.Normalize((0.5,),(0,5),)
])

opt_disc = optim.Adam(disc.parameters(),lr=lr)
opt_gen = optim.Adam(gen.parameters(),lr=lr)
criterion = nn.BCELoss()

"""
steps:
1. make a fake img using randn by gen
input
    randn : (N, image_dim)
output
    fake imgs : (N, image_dim)
2. put real and fake img into disc
input
    real : (N, image_dim)
    fake : (N, image_dim)
output
    real_out : (N,)
    fake_out : (N,)
3 ABOUT THE LOSS
for discriminator
    max log(D(x)) + lod(1 - D(G(z)))
for generator
    min log(1 - D(G(z))) <--> max log(D(G(z)))

TIPS:
   # for each batch, train the disc and gen
"""
lr = 2e-4
z_dim=64
image_dim=28*28*1
batch_size = 32
num_epoch = 50
dataloader=None

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
for i in range(num_epoch):
    for batch_idx, (real,_) in enumerate(dataloader):
        real_out = real.view(-1,784).to(device)
        Discriminator(real)
        fake = torch.rand((batch_size,784))
        fake_out = Generator(fake)

        true_label=torch.ones((batch_size,))
        fake_label= torch.zeros_like(true_label)

        # for generator
        G_Loss = criterion(fake_out,true_label)
        opt_gen.zero_grad()
        G_Loss.backward()
        opt_gen.step()
        # for discriminator
        D_loss_1 = criterion(real_out,true_label)
        D_loss_2 = criterion(fake_out,fake_label)
        D_loss=(D_loss_1+D_loss_2)/2
        opt_disc.zero_grad()
        D_loss.backward()
        opt_disc.step()
