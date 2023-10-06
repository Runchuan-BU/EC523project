import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19

class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0):
        super().__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.loss_fn = nn.MSELoss() if use_lsgan else nn.BCEWithLogitsLoss()

    def __call__(self, input, target_is_real):
        target = self.real_label if target_is_real else self.fake_label
        target = target.expand_as(input)
        loss = self.loss_fn(input, target)
        return loss

class VGGLoss(nn.Module):
    def __init__(self, weights=[1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0], device='cuda'):
        super().__init__()
        self.weights = weights
        self.vgg = vgg19(pretrained=True).features.to(device)
        self.criterion = nn.L1Loss()

    def forward(self, x, y):              
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())        
        return loss

class DiscriminatorGradientPenaltyLoss(nn.Module):
    def __init__(self, device='cuda', penalty_weight=10.0):
        super().__init__()
        self.device = device
        self.penalty_weight = penalty_weight

    def __call__(self, discriminator, real_images, fake_images):
        batch_size = real_images.size(0)
        alpha = torch.rand(batch_size, 1, 1, 1, device=self.device)
        interpolated_images = (alpha * real_images) + ((1 - alpha) * fake_images)
        interpolated_images.requires_grad = True
        outputs = discriminator(interpolated_images)
        gradients = torch.autograd.grad(outputs=outputs, inputs=interpolated_images,
                                        grad_outputs=torch.ones_like(outputs),
                                        create_graph=True, retain_graph=True)[0]
        gradients = gradients.view(batch_size, -1)
        penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.penalty_weight
        return penalty
    
from torchvision import models
class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)        
        h_relu3 = self.slice3(h_relu2)        
        h_relu4 = self.slice4(h_relu3)        
        h_relu5 = self.slice5(h_relu4)                
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out