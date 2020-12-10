#参考 https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py 
import torch
import torch.nn as nn
import torch.nn.functional as F
import  numpy as np
__all__ = [
    'FastNet', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]
# python模块中的__all__属性，可用于模块导入时限制，如：
# from module import *
# 此时被导入模块若定义了__all__属性，则只有__all__内指定的属性、方法、类可被导入。
N_CAHR=10 # number of char in font library

filter=np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
kernel=np.zeros([10,10,3,3])
for i in range(10):
    kernel[i,i,:,:]=filter
def channel_maxmization(v):
    """
    :param v: shape of (B, C, H, W)
    :return: get the max value of every channel (B, C, 1, 1)
    """
    nb_channel=v.shape[1]
    b = torch.max(v, 0)[0]
    c = torch.max(b, 1)[0]
    d = torch.max(c, 1)[0]
    # c=torch.max(v,-1)[0]
    # d = torch.max(c, -1)[0]
    return d.reshape(shape=(1, nb_channel, 1, 1))

class conv2d_with_relu_norm(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=3):
        super(conv2d_with_relu_norm,self).__init__()
        self.conv=nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2)
        self.batch_norm=nn.BatchNorm2d(out_channels)
        self.relu=nn.ReLU(inplace=True)
        self.layer=nn.Sequential(*[self.conv,self.batch_norm,self.relu])

    def forward(self, x):
        return self.layer(x)

class wavelet_conv(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=3):
        super(wavelet_conv,self).__init__()
        self.conv1=nn.Conv2d(in_channels,in_channels,kernel_size=(1,kernel_size),padding=kernel_size//2)
        self.pool1 = nn.MaxPool2d(kernel_size=(1, 2), stride=2)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=(kernel_size,1), padding=kernel_size // 2)
        self.pool2 = nn.MaxPool2d(kernel_size=(2,1), stride=2)
        self.batch_norm = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.layer = nn.Sequential(
            *[self.conv1, self.batch_norm, self.relu, self.pool1, self.conv2, self.batch_norm, self.relu, self.pool2])
        #layer can have different structure, which need to be optimized
        #self.layer = nn.Sequential(
            #*[self.conv1, self.pool1, self.conv2, self.batch_norm, self.relu, self.pool2])
    def forward(self, x):
        return self.layer(x)

class conv2d_with_relu(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=3):
        super(conv2d_with_relu,self).__init__()
        self.conv=nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2)
        self.relu=nn.ReLU(inplace=True)
    def forward(self, x):
        return self.relu(self.conv(x))

class FastNet(nn.Module):
    def __init__(self, layers, in_channels,num_class,init_weights=True):
        """
        :param layers: 提取最高级特征的模块
        :param in_channels: 卷积得到的最高级特征的channel数目
        :param num_class: 字库中字符数目
        :param init_weights:
        """
        super(FastNet, self).__init__()
        self.pool2 = nn.Sequential(*layers[:-4])
        self.pool3 = nn.Sequential(*layers[-4:])
        self.heatmap = nn.Conv2d(in_channels, num_class, kernel_size=1, padding=False)
        # self.heatmap_filter=nn.Conv2d(10,10,kernel_size=3,padding=1)
        # self.heatmap_filter.weight=nn.Parameter(data=torch.Tensor(kernel),requires_grad=False)
        self.BN=nn.BatchNorm2d(10)
        if init_weights:
            self._initialize_weights()

    def forward(self, x,phase="Train"):
        #参考 https://zhuanlan.zhihu.com/p/32506912

        x= self.pool2(x)
        #print(x.shape)
        s1=x #1/4
        #print(s1.shape)
        x = self.pool3(x)
        #print(x.shape)
        s2=x #1/8
        s2=F.interpolate(input=s2,scale_factor=2, mode='bilinear',align_corners=True)
        #print(s2.shape)
        #s=torch.cat([s1,s2],axis=0)
        s=s1+s2
        #print(s.shape)
        ht =self.heatmap(s) #shape BS*C*h/4*w/4
        if phase=="Test":
            M=torch.max(ht,dim=1)[0]
            ht=torch.div(ht,M)
            ht=torch.where(ht==1.,ht,torch.Tensor([0]).cuda())
            ht=torch.mul(ht,M)
            #filter the values in Channel dimension
        #print(x[0][0][1])
        x=torch.where(ht > 0.6, ht, torch.Tensor([0]).cuda())
        max_v = channel_maxmization(x)+1e-9 #channel maxmization
        #print(max_v)
        #max_v = torch.max(x) #overrall maxmization
        #print(max_v.shape)
        x =torch.div(x ,max_v)
        if phase == "Test":
            x = torch.where(x >0.85, torch.Tensor([1.]).cuda(), torch.Tensor([0.]).cuda())
            # we use in test phase
            tag=torch.arange(1,11).reshape((1,10,1,1)).float().cuda()
            x=torch.mul(tag,x)
            x=torch.sum(x,1)
            #x = self.heatmap_filter(x)
            return x
        return x
        # #x=F.relu(x-0.6)+0.6 #filter the values <0.6
        # #min_v = torch.min(x)
        # max_v = torch.max(x)
        # #range_v = max_v - min_v
        # normalised_x = x/max_v
        # #normalised_x=(x - min_v) / range_v
        # return normalised_x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 1
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            #conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                #layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                layers += [conv2d_with_relu_norm(in_channels,v,3)]
            else:
                layers += [conv2d_with_relu(in_channels, v, 3)]
                #layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return layers
    #return nn.Sequential(*layers)
# 在参数名之前使用一个星号，就是让函数接受任意多的位置参数。
# python在参数名之前使用2个星号来支持任意多的关键字参数。
# 参数前加一个星号，表明将所有的值放在同一个元组中，该参数的返回值是一个元组。
# 参数前加两个星号，表明将所有的值放在同一个字典中，该参数的返回值是一个字典。
cfgs = {
    'A': [64,64, 'M', 128, 128, 'M', 256, 256, 256, 'M'],
    'B': [128,128, 'M', 256,256, 'M',256,256, 256,'M'],
    'D': [128,128, 'M', 256,256, 'M',1024, 1024, 1024,'M'],
    'E': [256, 256, 256, 'M', 512, 512, 512, 'M', 1024, 1024, 1024, 'M'],
}


def _vgg(cfg, batch_norm,channels,**kwargs):
    model = FastNet(make_layers(cfgs[cfg], batch_norm=batch_norm), channels,N_CAHR, **kwargs)
    return model

def vgg11(**kwargs):
    r"""VGG 11-layer model (configuration "A") from
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg( 'A', False,256, **kwargs)


def vgg11_bn(**kwargs):
    r"""VGG 11-layer model (configuration "A") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('A', True,128+256, **kwargs)


def vgg13( **kwargs):
    r"""VGG 13-layer model (configuration "B")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg( 'B', False,256,**kwargs)


def vgg13_bn( **kwargs):
    r"""VGG 13-layer model (configuration "B") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('B', True, 256,**kwargs)


def vgg16( **kwargs):
    r"""VGG 16-layer model (configuration "D")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg( 'D', False, 256+2*512,**kwargs)


def vgg16_bn(**kwargs):
    r"""VGG 16-layer model (configuration "D") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg( 'D', True, 256+2*512, **kwargs)


def vgg19( **kwargs):
    r"""VGG 19-layer model (configuration "E")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg( 'E', False,512+1024, **kwargs)


def vgg19_bn( **kwargs):
    r"""VGG 19-layer model (configuration 'E') with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg( 'E', True, 512+1024, **kwargs)