import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

class Discriminator(nn.Module):
    def __init__(self, in_features) -> None:
        super().__init__()
        self.disc = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.LeakyReLU(0.01),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, x):
        return self.disc(x)
# input: (N,1*28*28)
# output: (N,)

class Generator(nn.Module):
    def __init__(self,z_dim,img_dim) -> None:
        super().__init__()
        self.gen = nn.Sequential(
            nn.Linear(z_dim,256),
            nn.LeakyReLU(0.01),
            nn.Linear(256,img_dim),
            # make all value in range 0-1
            nn.Tanh(),
        )

    def forward(self,x):
        return self.gen(x)
# input: (N, z_dim)
# output: (N, 1*28*28)

device = 'cuda' if torch.cuda.is_available() else 'cpu'




######################################################
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
################################################################
lr = 2e-4
z_dim=64
image_dim=28*28*1
batch_size = 32
num_epoch = 5

transforms = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)),]
)

dataset = datasets.MNIST(root="dataset/", transform=transforms, download=True)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.5,),(0,5),)
# ])



# dataset = datasets.MNIST(root="dataset/", transform=transform, download=True)
# dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

from torch.utils.tensorboard import SummaryWriter

real_writer = SummaryWriter(log_dir="simple_GAN/real")
fake_writer = SummaryWriter(log_dir="simple_GAN/fake")


disc = Discriminator(image_dim).to(device)

gen = Generator(z_dim, image_dim).to(device)

opt_disc = optim.Adam(disc.parameters(), lr=lr)
opt_gen = optim.Adam(gen.parameters(), lr=lr)

criterion = nn.BCELoss()

true_label = torch.ones((batch_size,1)).to(device)
fake_label = torch.zeros((batch_size,1)).to(device)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             
for i in range(num_epoch):
    for batch_idx, (real, _) in enumerate(loader):

        
        real = real.view(-1, 784).to(device)
        real_disc = disc(real)
        noise = torch.rand((batch_size, z_dim)).to(device)
        fake_out = gen(noise)
        fake_disc = disc(fake_out)

        D_loss_1 = criterion(real_disc,torch.ones_like(real_disc))
        D_loss_2 = criterion(fake_disc,torch.zeros_like(real_disc))      
        D_loss=(D_loss_1+D_loss_2)/2

        disc.zero_grad()
        D_loss.backward(retain_graph=True)
        opt_disc.step()

        ##########################################################
        ########## 计算梯度时要用到  所以要重新生成一遍  ############
        ##########################################################
        fake_disc = disc(fake_out)
        G_Loss = criterion(fake_disc, torch.ones_like(fake_disc))
        gen.zero_grad()
        G_Loss.backward()
        opt_gen.step()

        
        if batch_idx==0:
            with torch.no_grad():
                torch.manual_seed(7)
                noise = torch.rand((batch_size,z_dim)).to(device)
                fake_out = gen(noise)
                fake_out = fake_out.view(-1,1,28,28)
                real = real.view(-1,1,28,28)
                img_grid_fake = torchvision.utils.make_grid(fake_out, normalize=True)
                img_grid_true = torchvision.utils.make_grid(real, normalize=True)

                real_writer.add_image(tag="real", img_tensor=img_grid_true, global_step=i)
                fake_writer.add_image(tag="fake", img_tensor=img_grid_fake, global_step=i)
                print("tensorboard")