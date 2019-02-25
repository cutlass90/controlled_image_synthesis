import torch
import torch.nn as nn
import numpy as np
import torchvision


def conv_block(in_dim,out_dim):
    return nn.Sequential(nn.Conv2d(in_dim,in_dim,kernel_size=3,stride=1,padding=1),
                         nn.ELU(True),
                         nn.Conv2d(in_dim,in_dim,kernel_size=3,stride=1,padding=1),
                         nn.ELU(True),
                         nn.Conv2d(in_dim,out_dim,kernel_size=1,stride=1,padding=0),
                         nn.AvgPool2d(kernel_size=2,stride=2))


def deconv_block(in_dim,out_dim):
    return nn.Sequential(nn.Conv2d(in_dim,out_dim,kernel_size=3,stride=1,padding=1),
                         nn.ELU(True),
                         nn.Conv2d(out_dim,out_dim,kernel_size=3,stride=1,padding=1),
                         nn.ELU(True),
                         nn.UpsamplingNearest2d(scale_factor=2))


class Discriminator(nn.Module):
    def __init__(self, nc, ndf, hidden_size, imageSize):
        super(Discriminator,self).__init__()
        # 64 x 64
        self.conv1 = nn.Sequential(nn.Conv2d(nc,ndf,kernel_size=3,stride=1,padding=1),
                                    nn.ELU(True),
                                    conv_block(ndf,ndf))
        # 32 x 32
        self.conv2 = conv_block(ndf, ndf*2)
        # 16 x 16
        if imageSize == 64:
            self.conv3 = conv_block(ndf*2, ndf*3)
            self.conv4 = nn.Sequential(nn.Conv2d(ndf*3,ndf*3,kernel_size=3,stride=1,padding=1),
                                       nn.ELU(True),
                                       nn.Conv2d(ndf*3,ndf*3,kernel_size=3,stride=1,padding=1),
                                       nn.ELU(True))
            self.embed1 = nn.Linear(ndf * 3 * 8 * 8, hidden_size)
        elif imageSize == 128:
            self.conv3 = nn.Sequential(
                conv_block(ndf * 2, ndf * 3),
                conv_block(ndf * 3, ndf * 4))
            self.conv4 = nn.Sequential(nn.Conv2d(ndf*4,ndf*4,kernel_size=3,stride=1,padding=1),
                                       nn.ELU(True),
                                       nn.Conv2d(ndf*4,ndf*4,kernel_size=3,stride=1,padding=1),
                                       nn.ELU(True))
            self.embed1 = nn.Linear(ndf*4*8*8, hidden_size)
        elif imageSize == 256:
            self.conv3 = nn.Sequential(
                conv_block(ndf * 2, ndf * 3),
                conv_block(ndf * 3, ndf * 4),
                conv_block(ndf * 4, ndf * 5))
            self.conv4 = nn.Sequential(nn.Conv2d(ndf * 5, ndf * 5, kernel_size=3, stride=1, padding=1),
                                       nn.ELU(True),
                                       nn.Conv2d(ndf * 5, ndf * 5, kernel_size=3, stride=1, padding=1),
                                       nn.ELU(True))
            self.embed1 = nn.Linear(ndf * 5 * 8 * 8, hidden_size)
        else:
            raise ValueError
        self.embed2 = nn.Linear(hidden_size, ndf*8*8)

        self.deconv1 = deconv_block(ndf, ndf)
        self.deconv2 = deconv_block(ndf, ndf)
        if imageSize == 64:
            self.deconv3 = deconv_block(ndf, ndf)
        elif imageSize == 128:
            self.deconv3 = nn.Sequential(deconv_block(ndf, ndf),
                                         deconv_block(ndf, ndf))
        elif imageSize == 256:
            self.deconv3 = nn.Sequential(deconv_block(ndf, ndf),
                                         deconv_block(ndf, ndf),
                                         deconv_block(ndf, ndf))
        else:
            raise ValueError
        self.deconv4 = nn.Sequential(nn.Conv2d(ndf,ndf,kernel_size=3,stride=1,padding=1),
                         nn.ELU(True),
                         nn.Conv2d(ndf,ndf,kernel_size=3,stride=1,padding=1),
                         nn.ELU(True),
                         nn.Conv2d(ndf, nc, kernel_size=3, stride=1, padding=1),
                         nn.Tanh())

        self.ndf = ndf
        self.imageSize = imageSize

    def forward(self,x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        if self.imageSize == 64:
            out = out.view(out.size(0), self.ndf*3 * 8 * 8)
        elif self.imageSize == 128:
            out = out.view(out.size(0), self.ndf*4 * 8 * 8)
        elif self.imageSize == 256:
            out = out.view(out.size(0), self.ndf * 5 * 8 * 8)
        else:
            raise ValueError
        out = self.embed1(out)
        out = self.embed2(out)
        out = out.view(out.size(0), self.ndf, 8, 8)
        out = self.deconv1(out)
        out = self.deconv2(out)
        out = self.deconv3(out)
        out = self.deconv4(out)
        return out


class Generator(nn.Module):
    def __init__(self, nc, ngf, nz, imageSize):
        super(Generator,self).__init__()
        self.embed1 = nn.Linear(nz, ngf*8*8)
        self.deconv1 = deconv_block(ngf, ngf)
        self.deconv2 = deconv_block(ngf, ngf)
        if imageSize == 64:
            self.deconv3 = deconv_block(ngf, ngf)
        elif imageSize == 128:
            self.deconv3 = nn.Sequential(deconv_block(ngf, ngf),
                                         deconv_block(ngf, ngf))
        elif imageSize == 256:
            self.deconv3 = nn.Sequential(deconv_block(ngf, ngf),
                                         deconv_block(ngf, ngf),
                                         deconv_block(ngf, ngf))
        else:
            raise ValueError
        self.deconv4 = nn.Sequential(nn.Conv2d(ngf,ngf,kernel_size=3,stride=1,padding=1),
                         nn.ELU(True),
                         nn.Conv2d(ngf, ngf, kernel_size=3,stride=1,padding=1),
                         nn.ELU(True),
                         nn.Conv2d(ngf, nc, kernel_size=3, stride=1, padding=1),
                         nn.Tanh())

        self.ngf = ngf
        self.imageSize = imageSize

    def forward(self, x):
        out = self.embed1(x)
        out = out.view(out.size(0), self.ngf, 8, 8)
        out = self.deconv1(out)
        out = self.deconv2(out)
        out = self.deconv3(out)
        out = self.deconv4(out)
        return out

    def sample(self, batch_size, noise=None):
        if noise is None:
            noise = torch.rand([batch_size, 64]).to(self.embed1.weight.device)*2 - 1
        with torch.no_grad():
            out = self.forward(noise)
        return out


if __name__ == "__main__":
    generator = Generator(3, 128, 64, 128)
    generator.load_state_dict(torch.load('/home/nazar/faces/workshop_controlled_image_synthesis/weights/BEGAN.pth'))
    images = generator.sample(16)
    grid = torchvision.utils.make_grid(images)
    from scipy.misc import imsave
    imsave('samples.png', np.transpose(grid.cpu().numpy(), [1, 2, 0]))







