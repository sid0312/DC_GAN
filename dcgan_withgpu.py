from __future__ import print_function
import random
import torch
import torchvision
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable

seed = 42
random.seed(seed)
torch.manual_seed(seed)
ngpu = 1
num_epochs = 30
batchSize = 64
imageSize = 64
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

transform = transforms.Compose([transforms.Scale(imageSize), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),])
dataset = torchvision.datasets.CIFAR10(root='./data',download=True,transform = transform)
dataloader = torch.utils.data.DataLoader(dataset,batch_size=batchSize,shuffle =True,num_workers=2)

def initialize_weights(net):
    classname = net.__class__.__name__
    if classname.find('Conv') != -1:
        net.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        net.weight.data.normal_(1.0, 0.02)
        net.bias.data.fill_(0)
        
class Generator(nn.Module):
    
    def __init__(self,ngpu):
        super(Generator,self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.ConvTranspose2d(100, 512, 4, 1, 0, bias = False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias = False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias = False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias = False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias = False),
            nn.Tanh()
            )
    def forward(self, input):
        output = self.main(input)
        return output

generator_net = Generator(ngpu).to(device)
generator_net.apply(initialize_weights)

class Discriminator(nn.Module):
    
    def __init__(self,ngpu):
        super(Discriminator,self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.Conv2d(3,64,4,2,1,bias=False),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(64,128,4,2,1,bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(128,256,4,2,1,bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(256,512,4,2,1,bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(512,1,4,1,0,bias=False),
            nn.Sigmoid()
            )  
    def forward(self,input):
        output = self.main(input)
        return output.view(-1)

discriminator_net = Discriminator(ngpu).to(device)
discriminator_net.apply(initialize_weights)


criterion = nn.BCELoss()
opt_D = optim.Adam(discriminator_net.parameters(),lr = 0.0002,betas = (0.5,0.99))
opt_G = optim.Adam(generator_net.parameters(),lr = 0.0002,betas = (0.5,0.99))



for epoch in range(num_epochs):
    for i, data in enumerate(dataloader,0):
        
        discriminator_net.zero_grad()
        
        real,_ = data
        real = real.to(device)
        input = Variable(real)
        target = Variable(torch.ones(input.size()[0],device=device))
        output = discriminator_net(input)
        errorD_real = criterion(output,target)
        
        noise = Variable(torch.randn(input.size()[0],100,1,1,device=device))
        fake = generator_net(noise)
        target = Variable(torch.zeros(input.size()[0],device=device))
        output = discriminator_net(fake.detach())
        errorD_fake = criterion(output,target)
        
        errorD_total = errorD_real + errorD_fake
        errorD_total.backward()
        opt_D.step()
        
        generator_net.zero_grad()
        target = Variable(torch.ones(input.size()[0],device=device))
        output = discriminator_net(fake)
        errorG = criterion(output,target)
        errorG.backward()
        opt_G.step()
        
        print('Epoch->(%d/%d)[%d/%d] Discriminator\'s loss: %.4f Generator\'s loss: %.4f'%(epoch,num_epochs,i,len(dataloader),errorD_total.data,errorG.data))        
        if i % 100 == 0:
            vutils.save_image(real,'%s/real_images.png' % "./results",normalize = True)
            fake = generator_net(noise)
            vutils.save_image(fake.data,'%s/generated_images_epoch%03d.png' %("./results",epoch),normalize = True)
        
