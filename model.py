import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models
from torch.autograd import Variable
import pretrainedmodels
from torch.nn import functional as F
from IPython import embed
import cv2
import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from grad_cam import ShowGradCam


######################################################################
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv2d') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
        init.constant(m.bias.data, 0.0)
    elif classname.find('Linear') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_out')
    elif classname.find('BatchNorm1d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal(m.weight.data, std=0.001)


# Defines the new fc layer and classification layer
# |--Linear--|--bn--|--relu--|--Linear--|
class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, droprate, relu=False, bnorm=True, num_bottleneck=512, linear=True,
                 return_f=False):
        super(ClassBlock, self).__init__()
        self.return_f = return_f
        add_block = []
        if linear:
            add_block += [nn.Linear(input_dim, num_bottleneck)]
        else:
            num_bottleneck = input_dim
        if bnorm:
            add_block += [nn.BatchNorm1d(num_bottleneck)]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        if droprate > 0:
            add_block += [nn.Dropout(p=droprate)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier

    def forward(self, x):
        x = self.add_block(x)
        if self.return_f:
            f = x
            x = self.classifier(x)
            return [x, f]
        else:
            x = self.classifier(x)
            return x


class ClassBlock1(nn.Module):
    def __init__(self, input_dim, class_num, relu=True, num_bottleneck=512):
        super(ClassBlock1, self).__init__()
        add_block1 = []
        add_block2 = []
        add_block1 += [nn.BatchNorm1d(input_dim)]
        if relu:
            add_block1 += [nn.LeakyReLU(0.1)]
        add_block1 += [nn.Linear(input_dim, num_bottleneck, bias=False)]
        add_block2 += [nn.BatchNorm1d(num_bottleneck)]

        add_block1 = nn.Sequential(*add_block1)
        add_block1.apply(weights_init_kaiming)
        add_block2 = nn.Sequential(*add_block2)
        add_block2.apply(weights_init_kaiming)
        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num, bias=False)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block1 = add_block1  # linear层
        self.add_block2 = add_block2  # bn层
        self.classifier = classifier  # fc层

    def forward(self, x):
        x = self.add_block1(x)
        x1 = self.add_block2(x)
        x2 = self.classifier(x1)
        return x, x1, x2


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        w = self.sigmoid(out)
        return w


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x1 = torch.cat([avg_out, max_out], dim=1)
        x1 = self.conv1(x1)
        w = self.sigmoid(x1)
        return w



class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, k, s=1, p=0):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_c, out_c, k, stride=s, padding=p)
        self.bn = nn.BatchNorm2d(out_c)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


def conv1x1(in_planes, out_planes):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, padding=0, bias=False)


class TSA(nn.Module):
    def __init__(self):
        super(TSA, self).__init__()
        self.conv1 = ConvBlock(1, 1, 3, s=2, p=1)
        self.conv2 = ConvBlock(1, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        fea_map = x
        x = x.mean(1, keepdim=True)
        x = self.conv1(x)
        x = F.upsample(x, (x.size(2) * 2, x.size(3) * 2), mode='bilinear', align_corners=True)
        # w = F.sigmoid(self.conv2(x))
        w = self.sigmoid(self.conv2(x))
        return w * fea_map

# if __name__ == '__main__':
#     x = torch.Tensor(32,1024,16,8)
#     model = TSA()
#     y = model(x)
#     embed()




class Non_local(nn.Module):
    def __init__(self, in_channels, reduc_ratio=2):
        super(Non_local, self).__init__()

        self.in_channels = in_channels
        self.inter_channels = reduc_ratio // reduc_ratio

        self.g = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1,
                      padding=0),
        )

        self.W = nn.Sequential(
            nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.in_channels),
        )
        nn.init.constant_(self.W[1].weight, 0.0)
        nn.init.constant_(self.W[1].bias, 0.0)

        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                               kernel_size=1, stride=1, padding=0)

        self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        batch_size = x.size(0)
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        N = f.size(-1)
        # f_div_C = torch.nn.functional.softmax(f, dim=-1)
        f_div_C = f / N

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z


class Googlemodule(nn.Module):
    def __init__(self, in_channels):
        super(Googlemodule, self).__init__()
        self.branch1x1 = nn.Conv2d(in_channels, 1024, kernel_size=1)

        self.branch5x5_1 = nn.Conv2d(in_channels, 512, kernel_size=1)
        self.branch5x5_2 = nn.Conv2d(512, 1024, kernel_size=5, padding=2)

        self.branch3x3_1 = nn.Conv2d(in_channels, 512, kernel_size=1)
        self.branch3x3_2 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
        self.branch3x3_3 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1)

        self.branch_pool = nn.Conv2d(in_channels, 1024, kernel_size=1)


        self.TSA_1 = TSA()
        self.TSA_2 = TSA()
        self.TSA_3 = TSA()
        self.TSA_4 = TSA()

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch1x1 = self.TSA_1(branch1x1)


        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)
        branch5x5 = self.TSA_2(branch5x5)



        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)
        branch3x3 = self.branch3x3_3(branch3x3)
        branch3x3 = self.branch3x3_3(branch3x3)


        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)
        branch_pool = self.TSA_4(branch_pool)


        outputs = [branch1x1, branch5x5, branch3x3, branch_pool]

        return torch.cat(outputs, dim=1)


class PCB(nn.Module):
    def __init__(self, class_num, test):
        super(PCB, self).__init__()
        self.part = 4  # We cut the pool5 to 4 parts
        self.test = test
        model_ft = models.resnet50(pretrained=True)
        self.model = model_ft
        self.avgpool = nn.AdaptiveAvgPool2d((self.part, 1))
        self.dropout = nn.Dropout(p=0.5)

        # remove the final downsample
        self.model.layer4[0].downsample[0].stride = (1, 1)
        self.model.layer4[0].conv2.stride = (1, 1)

        # define 4 classifiers
        for i in range(self.part):
            name = 'classifier' + str(i)
            setattr(self, name, ClassBlock(2048, class_num, droprate=0.5, relu=False, bnorm=True, num_bottleneck=256))

        self.googlenet = Googlemodule(2048)
        self.conv_google = nn.Conv2d(4096, 2048, kernel_size=1)

        self.classifier_0 = ClassBlock1(2048, class_num, num_bottleneck=512)
        self.classifier_1 = ClassBlock1(1024, class_num, num_bottleneck=512)
        self.classifier_2 = ClassBlock1(2048, class_num, num_bottleneck=512)

        self.channel_atte = ChannelAttention(2048)
        self.spatial_atte = SpatialAttention()
        self.linear_atte = nn.Linear(2048, 751)

        self.l3_avepool = nn.AdaptiveAvgPool2d((1, 1))
        self.l3_maxpool = nn.AdaptiveMaxPool2d((1, 1))

        self.l4_maxpool = nn.AdaptiveMaxPool2d((1, 1))
        self.l4_avepool = nn.AdaptiveAvgPool2d((1, 1))


        self.non_local1 = Non_local(256)
        self.non_local2 = Non_local(512)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.non_local1(x)
        x = self.model.layer2(x)
        x = self.non_local2(x)


        x = self.model.layer3(x)
        x_l3 = x
        x_l3_a = self.l3_avepool(x_l3)
        # x_l3_a = torch.squeeze(x_l3_a)
        x_l3_a = x_l3_a.view(x_l3_a.size(0), -1)
        x0, x1, glo_l3idpre = self.classifier_1(x_l3_a)

        x = self.model.layer4(x)
        x_l4 = x
        x_l4_a = self.l4_avepool(x_l4)
        # x_l4_a = torch.squeeze(x_l4_a)
        x_l4_a = x_l4_a.view(x_l4_a.size(0), -1)
        x2, x3, glo_l4idpre = self.classifier_0(x_l4_a)

        # w_channle = self.channel_atte(x)
        # w_spatial = self.spatial_atte(x)
        # x_channel = w_channle * x
        # x_spatial = w_spatial * x
        # x_fusion = x_channel + x_spatial + x
        # x_fusion = F.avg_pool2d(x_fusion,(x_fusion.size(-2),x_fusion.size(-1))).view(x_fusion.size(0),-1)
        # att_pre = self.linear_atte(x_fusion)

        f = self.googlenet(x)
        f = self.conv_google(f)
        f = F.avg_pool2d(f, (f.size(-2), f.size(-1))).view(x.size(0), -1)
        f = F.relu(f)
        x4, x5, glo_y = self.classifier_2(f)

        x = self.avgpool(x)
        x = self.dropout(x)
        part = {}
        predict = {}
        # get 4 part feature batchsize*2048*6
        for i in range(self.part):
            part[i] = x[:, :, i]  # [32, 2048]

        rest_list = []
        for i in range(self.part):
            if i == 0:
                rest1 = part[1] + part[2] + part[3]
                rest_list.append(rest1)
            if i == 1:
                rest2 = part[0] + part[2] + part[3]
                rest_list.append(rest2)
            if i == 2:
                rest3 = part[0] + part[1] + part[3]
                rest_list.append(rest3)
            if i == 3:
                rest4 = part[0] + part[1] + part[2]
                rest_list.append(rest4)

        part_rest_list = []
        for i in range(self.part):
            part_rest = 0.8 * part[i] + 0.2 * rest_list[i]
            part_rest_list.append(part_rest)

        for i in range(self.part):
            part_rest_list[i] = part[i].view(x.size(0), x.size(1))  # [32, 2048]
            name = 'classifier' + str(i)
            c = getattr(self, name)
            predict[i] = c(part_rest_list[i])

        part_pre = []
        for i in range(self.part):
            part_pre.append(predict[i])

        part_1 = []
        for i in range(self.part):
            part_11 = torch.squeeze(part[i])
            part_1.append(part_11)

        if self.test == True:
            return x1, x3
        return part_pre, glo_y, glo_l4idpre, glo_l3idpre, x_l3_a, x_l4_a


if __name__ == '__main__':
    x = torch.Tensor(32, 3, 256, 128)
    model = PCB(751, test=True)
    y = model(x)
    print(model)

    # 打印模型参数量
    net = models.resnet50()
    total_params = sum(p.numel() for p in net.parameters())
    print('总参数个数：{}'.format(total_params))

    total_trainable_parameters = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print('需训练参数个数：{}'.format(total_trainable_parameters))

# def img_transform(img_in, transform):
#     """
#     将img进行预处理，并转换成模型输入所需的形式—— B*C*H*W
#     :param img_roi: np.array
#     :return:
#     """
#     img = img_in.copy()
#     img = Image.fromarray(np.uint8(img))
#     img = transform(img)
#     img = img.unsqueeze(0)
#     # C*H*W --> B*C*H*W
#     return img
#
#
# def img_preprocess(img_in):
#     """
#     读取图片，转为模型可读的形式
#     :param img_in: ndarray, [H, W, C]
#     :return: PIL.image
#     """
#     img = img_in.copy()
#     img = cv2.resize(img,(256, 128))
#     img = img[:, :, ::-1]   # BGR --> RGB
#     transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ])
#     img_input = img_transform(img, transform)
#     return img_input
#
# def comp_class_vec(ouput_vec, index=None):
#     """
#     计算类向量
#     :param ouput_vec: tensor
#     :param index: int，指定类别
#     :return: tensor
#     """
#     if not index:
#         index = np.argmax(ouput_vec.cpu().data.numpy())
#     else:
#         index = np.array(index)
#     index = index[np.newaxis, np.newaxis]
#     index = torch.from_numpy(index)
#     one_hot = torch.zeros(1, 10).scatter_(1, index, 1)
#     one_hot.requires_grad = True
#     class_vec = torch.sum(one_hot * output)  # one_hot = 11.8605
#
#     return class_vec
# if __name__ == '__main__':
#     BASE_DIR = os.path.dirname(os.path.abspath(__file__))
#     path_img = os.path.join("1.jpg")
#     path_net = os.path.join("E:\研究生ReID_项目实践\试验组1\实验18(目前组好结果)\model\our model/net_last.pth") # in this example not use
#
#     # classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
#     # 图片读取；网络加载
#     img = cv2.imread(path_img, 1)  # H*W*C
#     img_input = img_preprocess(img)
#     net = our model(751, True)
#
#     gradCam = ShowGradCam(net.model.layer3[0].conv1) #............................. def which layer to show
#
#     # forward
#     output = net(img_input)
#     idx = np.argmax(output.cpu().data.numpy())
#     # print("predict: {}".format(classes[idx]))
#
#     # backward
#     net.zero_grad()
#     class_loss = comp_class_vec(output)
#     class_loss.backward()
#
#     # save result
#     gradCam.show_on_img(img) #.......................... show gradcam on target pic
