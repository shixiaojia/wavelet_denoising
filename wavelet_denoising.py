import cv2
import copy
import torch
import torch.nn as nn
import numpy as np
import pywt
from torchsummary import summary

def get_db2_filters():
    wavelet = pywt.Wavelet('db2')
    dec_lo, dec_hi, rec_lo, rec_hi = wavelet.filter_bank
    return (
        torch.Tensor(dec_lo),  # 分解低通
        torch.Tensor(dec_hi),  # 分解高通
        torch.Tensor(rec_lo),  # 重构低通
        torch.Tensor(rec_hi)   # 重构高通
    )


dec_lo, dec_hi, rec_lo, rec_hi = get_db2_filters()


class ThresholdLayer(nn.Module):
    def __init__(self, mode='soft'):
        super().__init__()
        self.mode = mode

    def forward(self, coeffs):
        ll, lh, hl, hh = coeffs
        # 估计噪声标准差σ（使用HH子带的中值绝对偏差）:ml-citation{ref="5,8" data="citationList"}
        sigma = torch.median(torch.abs(hh)) / 0.6745
        T = sigma * torch.sqrt(2 * torch.log(torch.tensor(hh.numel())))
        # 应用软阈值
        lh_thr = torch.sign(lh) * torch.clamp(torch.abs(lh) - T, min=0)
        hl_thr = torch.sign(hl) * torch.clamp(torch.abs(hl) - T, min=0)
        hh_thr = torch.sign(hh) * torch.clamp(torch.abs(hh) - T, min=0)
        return ll, lh_thr, hl_thr, hh_thr


class DWT_DB2_Conv(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.dec_conv_ll = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=4, stride=1, padding='same', groups=channels, bias=False)
        self.dec_conv_lh = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=4, stride=1, padding='same', groups=channels, bias=False)
        self.dec_conv_hl = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=4, stride=1, padding='same', groups=channels, bias=False)
        self.dec_conv_hh = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=4, stride=1, padding='same', groups=channels, bias=False)

        self.threshold = ThresholdLayer()

        self.reset_parameters()

        self.channels = channels
        # self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def reset_parameters(self):
        # 分解过程的低通和高通滤波器权重
        self.dec_conv_ll.weight.data = torch.outer(dec_lo, dec_lo).view(1, 1, 4, 4).expand(self.channels, 1, 4, 4)
        self.dec_conv_lh.weight.data = torch.outer(dec_lo, dec_hi).view(1, 1, 4, 4).expand(self.channels, 1, 4, 4)
        self.dec_conv_hl.weight.data = torch.outer(dec_hi, dec_lo).view(1, 1, 4, 4).expand(self.channels, 1, 4, 4)
        self.dec_conv_hh.weight.data = torch.outer(dec_hi, dec_hi).view(1, 1, 4, 4).expand(self.channels, 1, 4, 4)

    def forward(self, x):
        # 分解：卷积
        ll = self.dec_conv_ll(x)
        lh = self.dec_conv_lh(x)
        hl = self.dec_conv_hl(x)
        hh = self.dec_conv_hh(x)

        # 阈值处理
        ll, lh, hl, hh = self.threshold((ll, lh, hl, hh))

        # 下采样
        # ll, lh, hl, hh = self.pool(ll), self.pool(lh), self.pool(hl), self.pool(hh)

        return ll, lh, hl, hh  # 返回四个子带（LL, LH, HL, HH）

class IDWT_DB2_Conv(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.upsample = nn.Upsample(scale_factor=2)
        self.rec_conv_ll = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=4, stride=1, padding='same', groups=channels, bias=False)
        self.rec_conv_lh = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=4, stride=1, padding='same', groups=channels, bias=False)
        self.rec_conv_hl = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=4, stride=1, padding='same', groups=channels, bias=False)
        self.rec_conv_hh = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=4, stride=1, padding='same', groups=channels, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        # 重构过程的低通和高通滤波器权重
        self.rec_conv_ll.weight.data = torch.outer(rec_lo, rec_lo).view(1, 1, 4, 4).expand(self.channels, 1, 4, 4)
        self.rec_conv_lh.weight.data = torch.outer(rec_lo, rec_hi).view(1, 1, 4, 4).expand(self.channels, 1, 4, 4)
        self.rec_conv_hl.weight.data = torch.outer(rec_hi, rec_lo).view(1, 1, 4, 4).expand(self.channels, 1, 4, 4)
        self.rec_conv_hh.weight.data = torch.outer(rec_hi, rec_hi).view(1, 1, 4, 4).expand(self.channels, 1, 4, 4)


    def forward(self, coeffs):
        ll, lh, hl, hh = coeffs

        ll = self.rec_conv_ll(ll)
        lh = self.rec_conv_lh(lh)
        hl = self.rec_conv_hl(hl)
        hh = self.rec_conv_hh(hh)


        return ll + lh + hl + hh

class Upsample(nn.Module):
    def __init__(self, scale_factor):
        super().__init__()
        self.up_sample = nn.Upsample(scale_factor=scale_factor)

    def forward(self, x):
        ll, lh, hl, hh = x
        ll = self.up_sample(ll)
        lh = self.up_sample(lh)
        hl = self.up_sample(hl)
        hh = self.up_sample(hh)

        return ll, lh, hl, hh


class Downsample(nn.Module):
    def __init__(self, kernel_size, stride):
        super().__init__()
        self.down_sample = nn.AvgPool2d(kernel_size=kernel_size, stride=stride)

    def forward(self, x):
        ll, lh, hl, hh = x
        ll = self.down_sample(ll)
        lh = self.down_sample(lh)
        hl = self.down_sample(hl)
        hh = self.down_sample(hh)

        return ll, lh, hl, hh


class Concat(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim= dim

    def forward(self, x, y):
        x1, x2, x3, x4 = x
        y1, y2, y3, y4 = y
        out1 = torch.cat((x1, y2), self.dim)
        out2 = torch.cat((x2, y2), self.dim)
        out3 = torch.cat((x3, y3), self.dim)
        out4 = torch.cat((x4, y4), self.dim)
        return out1, out2, out3, out4


class DWT_Net(nn.Module):
    def __init__(self, channels, levels):
        super().__init__()
        self.levels = levels
        self.enc_convs = nn.ModuleList()
        self.down_sample = Downsample(kernel_size=2, stride=2)
        self.dec_convs = nn.ModuleList()
        self.up_sample = Upsample(scale_factor=2)
        self.convs = nn.ModuleList()
        self.cat = Concat(dim=1)


        for _ in range(self.levels):

            self.enc_convs.append(nn.Sequential(DWT_DB2_Conv(channels=channels)))

            self.dec_convs.append(nn.Sequential(IDWT_DB2_Conv(channels=channels)))

            self.convs.append(nn.Conv2d(in_channels=2 * channels, out_channels=channels, kernel_size=3, stride=1, padding=1))


    def forward(self, x):
        coeff_list = []
        skip_list = []
        for enc_conv in self.enc_convs:
            ll, lh, hl, hh = enc_conv(x)
            skip_list.append((ll, lh, hl, hh))

            ll, lh, hl, hh = self.down_sample((ll, lh, hl, hh))

            coeff_list.append((lh, hl, hh))
            x = ll

        for conv, dec_conv in zip(self.convs, self.dec_convs):
            lh, hl, hh = coeff_list.pop()
            x, lh, hl, hh = self.up_sample((x, lh, hl, hh))

            skip_ll, skip_lh, skip_hl, skip_hh = skip_list.pop()
            x, lh, hl, hh = self.cat((x, lh, hl, hh), (skip_ll, skip_lh, skip_hl, skip_hh))

            x, lh, hl, hh = conv(x), conv(lh), conv(hl), conv(hh)

            x = dec_conv((x, lh, hl, hh))

        return x

    def forward_gt(self, x):
        coeff_list = []
        for enc_conv in self.enc_convs:
            # decomposition
            ll, lh, hl, hh = enc_conv(x)

            # down sample
            ll, lh, hl, hh = self.down_sample((ll, lh, hl, hh))

            coeff_list.append((lh, hl, hh))
            x = ll

        for dec_conv in self.dec_convs:
            lh, hl, hh = coeff_list.pop()

            # up sample
            x, lh, hl, hh = self.up_sample((x, lh, hl, hh))

            # reconstruction
            x = dec_conv((x, lh, hl, hh))

        return x


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

image = cv2.imread('noisy_image.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
ori_img = copy.deepcopy(image)

h, w = image.shape

image = torch.asarray(image, dtype=torch.float32).view(1, 1, h, w).to(device)

image = torch.asarray(image, dtype=torch.float32).to(device)

net = DWT_Net(channels=1, levels=2).to(device)
ans = net.forward_gt(image)

torch.save(net.state_dict(), "./model.pth")
# summary(model=net, input_size=(1, 256, 256))

ans = ans.squeeze(dim=(0, 1)).detach().cpu().numpy()

ans = np.uint8((ans - ans.min())/ (ans.max() - ans.min()) * 255)


cv2.namedWindow("noisy image")
cv2.imshow("noisy image", ori_img)

cv2.namedWindow("denoised image")
cv2.imshow("denoised image", ans)
cv2.waitKey(0)
