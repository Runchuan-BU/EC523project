import torch
from torch import nn
import torch.nn.functional as F
from function import weights_init, get_norm_layer


def define_G(n_downsampling=4, n_resnet_blocks=4, gpu_ids=[]):    
    netG = Generator(n_downsampling, n_resnet_blocks)       
    print(netG)
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())   
        netG.cuda(gpu_ids[0])
    netG.apply(weights_init)
    return netG

class SFT(nn.Module):
    def __init__(self):
        super(SFT, self).__init__()
        self.conv1 = nn.Conv2d(64, 64, 1)
        self.conv2 = nn.Conv2d(64, 64, 1) 
        self.conv3 = nn.Conv2d(64, 64, 1)
        self.conv4 = nn.Conv2d(64, 64, 1)

    def forward(self, x, y):
        '''
        x is the feture map
        y is the conditions
        '''
        gamma = self.conv2(F.leaky_relu(self.conv1(y), 0.1, inplace=True))
        beta = self.conv4(F.leaky_relu(self.conv3(y), 0.1, inplace=True))
        return x * gamma + beta
    
class AdaptiveInstanceNorm2d(nn.Module):
    '''
    Cited from the "MaskGAN: Towards Diverse and Interactive Facial Image Manipulation"
    '''
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(AdaptiveInstanceNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        # weight and bias are dynamically assigned
        self.weight = None
        self.bias = None
        # just dummy buffers, not used
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        assert self.weight is not None and self.bias is not None, "Please assign weight and bias before calling AdaIN!"
        b, c = x.size(0), x.size(1)
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)

        # Apply instance norm
        x_reshaped = x.contiguous().view(1, b * c, *x.size()[2:])

        out = F.batch_norm(
            x_reshaped, running_mean, running_var, self.weight, self.bias,
            True, self.momentum, self.eps)

        return out.view(b, c, *x.size()[2:])

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_features) + ')'
    

class ConvBlock(nn.Module):
    def __init__(self, input_dim ,output_dim, kernel_size, stride, padding=0, norm='none', activation='relu', padtype='zero'):
        super(ConvBlock, self).__init__()
        """
        Initialize the padding operation
        """
        if padtype == 'reflection':
            self.pad = nn.ReflectionPad2d(padding)
        elif padtype == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Wrong choice of padding type!"

        """
        Initialize the Normalization Layer
        """
        if norm == 'batch':
            self.norm = nn.BatchNorm2d(output_dim)
        elif norm == 'instance':
            self.norm = nn.InstanceNorm2d(output_dim)
        elif norm == 'adain':
            self.norm = AdaptiveInstanceNorm2d(output_dim)
        elif norm == 'none' or norm == 'sn':
            self.norm = None
        else:
            assert 0, "Wrong choice of Normalization Layer!"

        """
        Initialize the Activation Layer
        """
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'Sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'Softmax':
            self.activation = nn.Softmax(dim=1)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Wrong choice of Activation Layer!"

        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=True)        

    def forward(self, x):
        x = self.conv(self.pad(x))
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x
    
class ResnetBlock(nn.Module):
    def __init__(self, dim=1024):
        super(ResnetBlock, self).__init__()
        model = []
        model += [ConvBlock(dim ,dim, 3, 1, 1, norm='adain', activation='relu', padtype='reflection')]
        model += [ConvBlock(dim ,dim, 3, 1, 1, norm='adain', activation='none', padtype='reflection')]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        out = x + self.model(x)
        return out
    

class SpatialStyleEncoder(nn.Module):
    def __init__(self, num_adain_para):
        super(SpatialStyleEncoder, self).__init__()
        self.label_conv_1 = ConvBlock(3, 16, 7, 1, 3, activation='relu', padtype='reflection')
        self.label_conv_2 = ConvBlock(16, 32, 4, 2, 1, activation='relu', padtype='reflection')
        self.label_conv_3 = ConvBlock(32, 64, 4, 2, 1, activation='none', padtype='reflection')
        self.label_conv_4 = ConvBlock(64, 64, 4, 2, 1, activation='relu', padtype='reflection')
        self.label_conv_5 = ConvBlock(64, 64, 4, 2, 1, activation='relu', padtype='reflection')
        self.label_conv_6 = ConvBlock(64, 64, 4, 2, 1, activation='none', padtype='reflection')
        self.style_conv_1 = ConvBlock(3, 16, 7, 1, 3, activation='relu', padtype='reflection')
        self.style_conv_2 = ConvBlock(16, 32, 4, 2, 1, activation='relu', padtype='reflection')
        self.style_conv_3 = ConvBlock(32, 64, 4, 2, 1, activation='relu', padtype='reflection')
        self.sft_layer_1 = SFT()
        self.style_conv_4 = ConvBlock(64, 64, 4, 2, 1, activation='relu', padtype='reflection')
        self.style_conv_5 = ConvBlock(64, 64, 4, 2, 1, activation='relu', padtype='reflection')
        self.style_conv_6 = ConvBlock(64, 64, 4, 2, 1, activation='relu', padtype='reflection')
        self.sft_layer_2 = SFT()
        self.average_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_last = nn.Conv2d(64, num_adain_para, 1, 1, 0)
    
    def forward(self, x, y):
        '''
        x is the labed mask picture
        y is the origin picture
        '''
        y_out_1 = self.label_conv_1(y)
        y_out_1 = self.label_conv_2(y_out_1)
        y_out_1 = self.label_conv_3(y_out_1)
        y_out_2 = self.label_conv_4(y_out_1)
        y_out_2 = self.label_conv_5(y_out_2) 
        y_out_2 = self.label_conv_6(y_out_2) 
        x = self.style_conv_1(x)
        x = self.style_conv_2(x)  
        x = self.style_conv_3(x)
        x = self.sft_layer_1(x, y_out_1)
        x = self.style_conv_4(x)
        x = self.style_conv_5(x)  
        x = self.style_conv_6(x)
        x = self.sft_layer_2(x, y_out_2)
        x = self.average_pool(x)
        out = self.conv_last(x)

        return out
    

class Generator(nn.Module):
    def __init__(self, n_downsampling=4, n_resnet_blocks=4):
        super(Generator, self).__init__()
        model = []
        # Transform 19 channel input into a 64-channel feature map
        model += [ConvBlock(3, 64, 7, 1, 3, norm='instance', activation='relu', padtype='reflection')]

        # Downsampling part
        for i in range (n_downsampling):
            dim = 64 * (2**i)
            model += [ConvBlock(dim, dim*2, 3, 2, 1, norm='instance', activation='relu', padtype='zero')]
        
        # Residual block part
        for i in range(n_resnet_blocks):
            model += [ResnetBlock(64 * (2**n_downsampling))]

        # Upsampling part
        for i in range(n_downsampling):
            dim = int (64 * (2**n_downsampling) / (2**i))
            model += [nn.ConvTranspose2d(dim, int(dim/2), 3, 2, 1, 1)]
            model += [nn.InstanceNorm2d(int(dim/2))]
            model += [nn.ReLU(inplace=True)]

        model += [ConvBlock(64, 3, 7, 1, 3, activation='tanh', padtype='reflection')]

        self.model = nn.Sequential(*model)

        self.style_encoder = SpatialStyleEncoder(self.num_adain_parameter(self.model))

    def assign_adain_params(self, adain_params, model):
        # assign the adain_params to the AdaIN layers in model
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                mean = adain_params[:, :m.num_features]
                std = adain_params[:, m.num_features:2*m.num_features]
                m.bias = mean.contiguous().view(-1)
                m.weight = std.contiguous().view(-1)
                if adain_params.size(1) > 2*m.num_features:
                    adain_params = adain_params[:, 2*m.num_features:]

    def num_adain_parameter(self, model):
        # return the number of AdaIN parameters needed by the model
        num_adain_params = 0
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                num_adain_params += 2*m.num_features
        return num_adain_params
    
    def forward(self, input, input_ref, image_ref):
        adain_params = self.style_encoder(image_ref, input_ref)
        self.assign_adain_params(adain_params, self.model)
        return self.model(input)
