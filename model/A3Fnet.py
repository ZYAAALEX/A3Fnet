import warnings
warnings.filterwarnings("ignore")
import torch
from torch import nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
import math
from torch.autograd import Variable

BN_EPS = 1e-4  

class Down(nn.Sequential):
    def __init__(self, in_channels):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=2, stride=2)

    def forward(self, x):
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        return x

class Image_Prediction_Generator(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1, stride=1)

    def forward(self, x):
        gt_pre = self.conv(x)
        return x, gt_pre



class Prediction_Generator(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1, stride=1)

    def forward(self, x):
        gt_pre = self.conv(x)
        return x, gt_pre

class MConv(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=3, dilation=1, pooling=True, bn=False, BatchNorm=False, num_groups=32):
        super(MConv, self).__init__()
        padding =(dilation*kernel_size-1)//2
        self.encode = nn.Sequential(
            nn.Conv2d(input_channels, output_channels,kernel_size=kernel_size, padding=1, stride=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        conv = self.encode(x)
        return conv


#
class Conv22d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, dilation=1, stride=1, groups=1, is_bn=False, BatchNorm=False, is_relu=True, num_groups=32):
        super(Conv22d, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups, bias=False)

        if BatchNorm:
            self.bn = nn.BatchNorm2d(out_channels, eps=BN_EPS)

        self.relu = nn.ReLU(inplace=True)

        if is_bn:
            if out_channels//num_groups==0:
                num_groups=1
            self.gn  =nn.GroupNorm(num_groups, out_channels, eps=BN_EPS)

        self.is_bn = is_bn
        self.is_BatchNorm=BatchNorm

        if is_relu is False: self.relu=None

    def forward(self,x):
        x = self.conv(x)
        if self.is_BatchNorm: x = self.bn(x)
        if self.is_bn: x = self.gn(x)
        if self.relu is not None: x = self.relu(x)
        return x

class A3Fnet(nn.Module):
    def __init__(self, num_classes=1, input_channels=3, c_list=[32, 64, 128, 256, 512], bn = True, BatchNorm=False):
        super().__init__()
        self.encoder1 = nn.Sequential(
            Conv22d(input_channels, c_list[0], 3, stride=1, padding=1,is_bn= bn),
            Conv22d(c_list[0], c_list[0], 3, stride=1, padding=1,is_bn= bn)
        )
        self.encoder2 = nn.Sequential(
            Conv22d(c_list[0], c_list[1], 3, stride=1, padding=1,is_bn= bn),
            Conv22d(c_list[1], c_list[1], 3, stride=1, padding=1,is_bn= bn)
        )
        self.encoder3 = nn.Sequential(
            Conv22d(c_list[1], c_list[2], 3, stride=1, padding=1,is_bn= bn),
            Conv22d(c_list[2], c_list[2], 3, stride=1, padding=1,is_bn= bn)
        )

        self.encoder4 = nn.Sequential(
            Conv22d(c_list[2], c_list[3], 3, stride=1, padding=1,is_bn= bn),
            Conv22d(c_list[3], c_list[3], 3, stride=1, padding=1,is_bn= bn)
        )
        self.encoder5 = nn.Sequential(
            Conv22d(c_list[3], c_list[4], 3, stride=1, padding=1,is_bn= bn),
        )
        self.conv2 = MConv(3, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.conv3 = MConv(3, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.conv4 = MConv(3, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        self.Down1 = Down(c_list[0])
        self.Down2 = Down(c_list[1])
        self.Down3 = Down(c_list[2])
        self.Down4 = Down(c_list[3])

        self.decoder1 = nn.Sequential(
            Conv22d(c_list[4], c_list[4], 3, stride=1, padding=1),
        )
        self.decoder2 = nn.Sequential(
            Conv22d(c_list[4], c_list[3], 3, stride=1, padding=1,is_bn= bn),
            Conv22d(c_list[3], c_list[3], 3, stride=1, padding=1,is_bn= bn),
            Conv22d(c_list[3], c_list[3], 3, stride=1, padding=1, is_bn=bn)
        )
        self.decoder3 = nn.Sequential(
            Conv22d(c_list[3], c_list[2], 3, stride=1, padding=1, is_bn=bn),
            Conv22d(c_list[2], c_list[2], 3, stride=1, padding=1, is_bn=bn),
            Conv22d(c_list[2], c_list[2], 3, stride=1, padding=1, is_bn=bn)
        )
        self.decoder4 = nn.Sequential(
            Conv22d(c_list[2], c_list[1], 3, stride=1, padding=1, is_bn=bn),
            Conv22d(c_list[1], c_list[1], 3, stride=1, padding=1, is_bn=bn),
            Conv22d(c_list[1], c_list[1], 3, stride=1, padding=1, is_bn=bn)
        )

        self.pred4 = Image_Prediction_Generator(c_list[4])
        self.pred3 = Image_Prediction_Generator(c_list[3])
        self.gate1 = Prediction_Generator(c_list[2])
        self.gate2 = Prediction_Generator(c_list[1])
        self.gate3 = Prediction_Generator(c_list[0])

        self.ebn1 = nn.GroupNorm(4, c_list[0])
        self.ebn2 = nn.GroupNorm(4, c_list[1])
        self.ebn3 = nn.GroupNorm(4, c_list[2])
        self.ebn4 = nn.GroupNorm(4, c_list[3])
        self.dbn0 = nn.GroupNorm(4, c_list[4])
        self.dbn1 = nn.GroupNorm(4, c_list[3])
        self.dbn2 = nn.GroupNorm(4, c_list[2])
        self.dbn3 = nn.GroupNorm(4, c_list[1])



        self.final = nn.Sequential(
            Conv22d(c_list[1], c_list[0], 3, stride=1, padding=1, is_bn=bn),
            nn.Conv2d(c_list[0], num_classes, kernel_size=1)
        )

        self.apply(self._init_weights)

        self.gf = GuidedFilter(r=2, eps=1e-2)

        self.attention4 = Attention(in_channels=512)
        self.attention3 = Attention(in_channels=256)
        self.attention2 = Attention(in_channels=128)
        self.attention1 = Attention(in_channels=64)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            n = m.kernel_size[0] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        _, _, img_shape, _ = x.size()
        x_2 = F.upsample(x, size=(int(img_shape / 2), int(img_shape / 2)), mode='bilinear')
        x_3 = F.upsample(x, size=(int(img_shape / 4), int(img_shape / 4)), mode='bilinear')
        x_4 = F.upsample(x, size=(int(img_shape / 8), int(img_shape / 8)), mode='bilinear')

        #############################
        out = self.encoder1(x)

        t1 = out
        out = self.Down1(out)
        #################
        out = self.encoder2(out)
        t2 = out
        out = self.Down2(out)

        ##################
        out = self.encoder3(out)
        t3 = out
        out = self.Down3(out)

        ###############
        out = self.encoder4(out)
        t4 = out
        out = self.Down4(out)


        ###############
        out = self.encoder5(out)


        ######################################解码#########################################################
        out = self.decoder1(out)

        out, gt_pre4 = self.pred4(out)

        gt_pre4 = F.interpolate(gt_pre4, scale_factor=16, mode='bilinear', align_corners=True)



        ################################
        N, C, H, W = t4.size()

        c4 = torch.cat([self.conv4(x_4), t4],dim=1)

        c4_small = F.upsample(c4, size=(int(H/2), int(W/2)),mode='bilinear')

        N, C, H, W = t3.size()

        t3 = torch.cat([self.conv3(x_3), t3],dim=1)

        t3_small = F.upsample(t3, size=(int(H/2), int(W/2)),mode='bilinear')

        N, C, H, W = t2.size()
 
        t2 = torch.cat([self.conv2(x_2), t2],dim=1)

        t2_small = F.upsample(t2, size=(int(H / 2), int(W / 2)), mode='bilinear')

        N, C, H, W = t1.size()

        t1 = torch.cat([t1, t1], dim=1)

        t1_small = F.upsample(t1, size=(int(H / 2), int(W / 2)), mode='bilinear')

        out = self.gf(c4_small,out , c4, self.attention(c4_small, t3_small,t2_small,t1_small, out))

        out = self.decoder2(out)
        out = F.gelu(self.dbn1(out))

        out, gt_pre3= self.pred3(out)

        gt_pre3 = F.interpolate(gt_pre3, scale_factor=8, mode='bilinear', align_corners=True)


        #############################


        out = self.gf(t3_small, out, t3, self.attention3(t3_small,t2_small,t1_small, out))

        out = self.decoder3(out)
        out = F.gelu(self.dbn2(out))

        out, gt_pre2 = self.gate1(out)

        gt_pre2 = F.interpolate(gt_pre2, scale_factor=4, mode='bilinear', align_corners=True)

        #################

        out = self.gf(t2_small, out, t2, self.attention2(t2_small,t1_small, out))
        out = self.decoder4(out)
        out = F.gelu(self.dbn3(out))

        out, gt_pre1, = self.gate2(out)

        gt_pre1 = F.interpolate(gt_pre1, scale_factor=2, mode='bilinear', align_corners=True)

        ################

        out = self.gf(t1_small, out, t1, self.attention1(t1_small, out))

        out = self.final(out)

        gt_pre1 = torch.sigmoid(gt_pre1)
        gt_pre2 = torch.sigmoid(gt_pre2)
        gt_pre3 = torch.sigmoid(gt_pre3)
        gt_pre4 = torch.sigmoid(gt_pre4)

        return (gt_pre4, gt_pre3, gt_pre2, gt_pre1), torch.sigmoid(out)



class Attention(nn.Module):
    def __init__(self, in_channels):
        super(Attention, self).__init__()

        self.inter_channels = in_channels
        self.in_channels = in_channels
        self.gating_channels = in_channels


        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,kernel_size=1)

        self.phi = nn.Conv2d(in_channels=self.gating_channels, out_channels=self.inter_channels,kernel_size=1, stride=1, padding=0, bias=True)
        self.psi = nn.Conv2d(in_channels=self.inter_channels, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, g):
        input_size = x.size()
        batch_size = input_size[0]
        assert batch_size == g.size(0)

        theta_x = self.theta(x)
        theta_x_size = theta_x.size()

        phi_g = F.upsample(self.phi(g), size=theta_x_size[2:], mode='bilinear')
        f = F.relu(theta_x + phi_g, inplace=True)

        sigm_psi_f = F.sigmoid(self.psi(f))

        return sigm_psi_f


class GuidedFilter(nn.Module):
    def __init__(self, r, eps=1e-8):
        super(GuidedFilter, self).__init__()
        self.r = r
        self.eps = eps
        self.box = Box(r)
        self.epss = 1e-12

    def forward(self, lr_x, lr_y, hr_x, l_a):
        n_lrx, c_lrx, h_lrx, w_lrx = lr_x.size()
        n_lry, c_lry, h_lry, w_lry = lr_y.size()
        n_hrx, c_hrx, h_hrx, w_hrx = hr_x.size()

        lr_x = lr_x.double()
        lr_y = lr_y.double()
        hr_x = hr_x.double()
        l_a = l_a.double()

        assert n_lrx == n_lry and n_lry == n_hrx
        assert c_lrx == c_hrx and (c_lrx == 1 or c_lrx == c_lry)
        assert h_lrx == h_lry and w_lrx == w_lry
        assert h_lrx > 2*self.r+1 and w_lrx > 2*self.r+1

        N = self.box(Variable(lr_x.data.new().resize_((1, 1, h_lrx, w_lrx)).fill_(1.0)))

        l_a = torch.abs(l_a) + self.epss

        t_all = torch.sum(l_a)
        l_t = l_a / t_all

        mean_a = self.box(l_a) / N
        mean_a2xy = self.box(l_a * l_a * lr_x * lr_y) / N
        mean_tax = self.box(l_t * l_a * lr_x) / N
        mean_ay = self.box(l_a * lr_y) / N
        mean_a2x2 = self.box(l_a * l_a * lr_x * lr_x) / N
        mean_ax = self.box(l_a * lr_x) / N

        temp = torch.abs(mean_a2x2 - N * mean_tax * mean_ax)
        A = (mean_a2xy - N * mean_tax * mean_ay) / (temp + self.eps)
        b = (mean_ay - A * mean_ax) / (mean_a)

        A = self.box(A) / N
        b = self.box(b) / N

        mean_A = F.upsample(A, (h_hrx, w_hrx), mode='bilinear')
        mean_b = F.upsample(b, (h_hrx, w_hrx), mode='bilinear')

        return (mean_A*hr_x+mean_b).float()

def diff_x(input, r):
    assert input.dim() == 4

    left   = input[:, :,         r:2 * r + 1]
    middle = input[:, :, 2 * r + 1:         ] - input[:, :,           :-2 * r - 1]
    right  = input[:, :,        -1:         ] - input[:, :, -2 * r - 1:    -r - 1]

    output = torch.cat([left, middle, right], dim=2)

    return output

def diff_y(input, r):
    assert input.dim() == 4

    left   = input[:, :, :,         r:2 * r + 1]
    middle = input[:, :, :, 2 * r + 1:         ] - input[:, :, :,           :-2 * r - 1]
    right  = input[:, :, :,        -1:         ] - input[:, :, :, -2 * r - 1:    -r - 1]

    output = torch.cat([left, middle, right], dim=3)

    return output

class Box(nn.Module):
    def __init__(self, r):
        super(Box, self).__init__()

        self.r = r

    def forward(self, x):
        assert x.dim() == 4

        return diff_y(diff_x(x.cumsum(dim=2), self.r).cumsum(dim=3), self.r)