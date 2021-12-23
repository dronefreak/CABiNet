#!/usr/bin/python
# -*- encoding: utf-8 -*-

import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from cab import ContextAggregationBlock
from mobilenetv3 import mobilenetv3_small as MobileNetV3



class _DWConv(nn.Module):
	"""Depthwise Convolutions"""
	def __init__(self, dw_channels, out_channels, stride=1, **kwargs):
		super(_DWConv, self).__init__()
		self.conv = nn.Sequential(
			nn.Conv2d(dw_channels, out_channels, 3, stride, 1, groups=dw_channels, bias=False),
			nn.BatchNorm2d(out_channels),
			nn.ReLU(True)
		)

	def forward(self, x):
		return self.conv(x)


class _DSConv(nn.Module):
	"""Depthwise Separable Convolutions"""
	def __init__(self, dw_channels, out_channels, stride=1, **kwargs):
		super(_DSConv, self).__init__()
		self.conv = nn.Sequential(
			nn.Conv2d(dw_channels, dw_channels, 3, stride, 1, groups=dw_channels, bias=False),
			nn.BatchNorm2d(dw_channels),
			nn.ReLU(True),
			nn.Conv2d(dw_channels, out_channels, 1, bias=False),
			nn.BatchNorm2d(out_channels),
			nn.ReLU(True)
		)

	def forward(self, x):
		return self.conv(x)

class ConvBNReLU(nn.Module):
	def __init__(self, in_chan, out_chan, kernel_size=3, stride=1, padding=1, *args, **kwargs):
		super(ConvBNReLU, self).__init__()
		self.conv = nn.Conv2d(in_chan,
				out_chan,
				kernel_size = kernel_size,
				stride = stride,
				padding = padding,
				bias = False)
		self.bn = nn.BatchNorm2d(out_chan)
		self.relu = nn.ReLU()
		self.init_weight()

	def forward(self, x):
		x = self.conv(x)
		x = self.bn(x)
		x = self.relu(x)
		return x

	def init_weight(self):
		for ly in self.children():
			if isinstance(ly, nn.Conv2d):
				nn.init.kaiming_normal_(ly.weight, a=1)
				if not ly.bias is None: nn.init.constant_(ly.bias, 0)


	def get_params(self):
		wd_params, nowd_params = [], []
		for name, module in self.named_modules():
			if isinstance(module, (nn.Linear, nn.Conv2d)):
				wd_params.append(module.weight)
				if not module.bias is None:
					nowd_params.append(module.bias)
			elif isinstance(module, nn.BatchNorm2d):
				nowd_params += list(module.parameters())
		return wd_params, nowd_params


class AttentionBranch(nn.Module):
	def __init__(self, inplanes, interplanes, outplanes, num_classes):
		super(AttentionBranch, self).__init__()
		self.conva = nn.Sequential(nn.Conv2d(inplanes, interplanes, 3, padding=1, bias=False),
								   nn.BatchNorm2d(interplanes),
								   nn.ReLU(interplanes))
		self.a2block = ContextAggregationBlock(interplanes, interplanes//2)
		self.convb = nn.Conv2d(interplanes, outplanes, kernel_size=1, stride=1, padding=0, bias=True)

		self.b1 = nn.Conv2d(inplanes + interplanes, outplanes, kernel_size=3, padding=1, bias=False)
		self.b2 = nn.BatchNorm2d(interplanes)
		self.b3 = nn.ReLU(interplanes)
		self.b4 = nn.Conv2d(interplanes, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
		self.init_weight()

	def forward(self, x):
		output = self.conva(x)
		output = self.a2block(output)
		output = self.convb(output)
		output = torch.cat([x, output], 1)
		output = self.b1(output)
		output = self.b2(output)
		output = self.b3(output)
		output_final = self.b4(output)
		return output, output_final

	def init_weight(self):
		for ly in self.children():
			if isinstance(ly, nn.Conv2d):
				nn.init.kaiming_normal_(ly.weight, a=1)
				if not ly.bias is None: nn.init.constant_(ly.bias, 0)

	def get_params(self):
		wd_params, nowd_params = [], []
		for name, module in self.named_modules():
			if isinstance(module, (nn.Linear, nn.Conv2d)):
				wd_params.append(module.weight)
				if not module.bias is None:
					nowd_params.append(module.bias)
			elif isinstance(module, nn.BatchNorm2d):
				nowd_params += list(module.parameters())
		return wd_params, nowd_params


class SpatialBranch(nn.Module):
	def __init__(self, *args, **kwargs):
		super(SpatialBranch, self).__init__()
		self.conv1 = ConvBNReLU(3, 64, kernel_size=7, stride=2, padding=3)
		self.conv2 = _DWConv(64, 64, kernel_size=3, stride=2, padding=1)
		self.conv3 = _DWConv(64, 64, kernel_size=3, stride=2, padding=1)
		self.conv_out = ConvBNReLU(64, 128, kernel_size=1, stride=1, padding=0)
		self.init_weight()

	def forward(self, x):
		feat = self.conv1(x)
		feat = self.conv2(feat)
		feat = self.conv3(feat)
		feat = self.conv_out(feat)
		return feat

	def init_weight(self):
		for ly in self.children():
			if isinstance(ly, nn.Conv2d):
				nn.init.kaiming_normal_(ly.weight, a=1)
				if not ly.bias is None: nn.init.constant_(ly.bias, 0)

	def get_params(self):
		wd_params, nowd_params = [], []
		for name, module in self.named_modules():
			if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
				wd_params.append(module.weight)
				if not module.bias is None:
					nowd_params.append(module.bias)
			elif isinstance(module, nn.BatchNorm2d):
				nowd_params += list(module.parameters())
		return wd_params, nowd_params


class FeatureFusionModule(nn.Module):
	def __init__(self, in_chan, out_chan, *args, **kwargs):
		super(FeatureFusionModule, self).__init__()
		self.convblk = ConvBNReLU(in_chan, out_chan, kernel_size=1, stride=1, padding=0)
		self.conv1 = nn.Conv2d(out_chan,
				out_chan//4,
				kernel_size = 1,
				stride = 1,
				padding = 0,
				bias = False)
		self.conv2 = nn.Conv2d(out_chan//4,
				out_chan,
				kernel_size = 1,
				stride = 1,
				padding = 0,
				bias = False)
		self.relu = nn.ReLU(inplace=True)
		self.sigmoid = nn.Softmax(dim=-1)
		self.init_weight()

	def forward(self, fsp, fcp):
		fcat = torch.cat([fsp, fcp], dim=1)
		feat = self.convblk(fcat)
		atten = F.avg_pool2d(feat, feat.size()[2:])
		atten = self.conv1(atten)
		atten = self.relu(atten)
		atten = self.conv2(atten)
		atten = self.sigmoid(atten)
		feat_atten = torch.mul(feat, atten)
		feat_out = feat_atten + feat
		return feat_out

	def init_weight(self):
		for ly in self.children():
			if isinstance(ly, nn.Conv2d):
				nn.init.kaiming_normal_(ly.weight, a=1)
				if not ly.bias is None: nn.init.constant_(ly.bias, 0)

	def get_params(self):
		wd_params, nowd_params = [], []
		for name, module in self.named_modules():
			if isinstance(module, (nn.Linear, nn.Conv2d)):
				wd_params.append(module.weight)
				if not module.bias is None:
					nowd_params.append(module.bias)
			elif isinstance(module, nn.BatchNorm2d):
				nowd_params += list(module.parameters())
		return wd_params, nowd_params



class CABiNetOutput(nn.Module):
	def __init__(self, in_chan, mid_chan, n_classes, *args, **kwargs):
		super(CABiNetOutput, self).__init__()
		self.conv = ConvBNReLU(in_chan, mid_chan, kernel_size=3, stride=1, padding=1)
		self.conv_out = nn.Conv2d(mid_chan, n_classes, kernel_size=1, bias=False)
		self.init_weight()

	def forward(self, x):
		x = self.conv(x)
		x = self.conv_out(x)
		return x

	def init_weight(self):
		for ly in self.children():
			if isinstance(ly, nn.Conv2d):
				nn.init.kaiming_normal_(ly.weight, a=1)
				if not ly.bias is None: nn.init.constant_(ly.bias, 0)

	def get_params(self):
		wd_params, nowd_params = [], []
		for name, module in self.named_modules():
			if isinstance(module, (nn.Linear, nn.Conv2d)):
				wd_params.append(module.weight)
				if not module.bias is None:
					nowd_params.append(module.bias)
			elif isinstance(module, nn.BatchNorm2d):
				nowd_params += list(module.parameters())
		return wd_params, nowd_params


class AttentionFusion(nn.Module):
	def __init__(self, input_channels, output_channels, *args, **kwargs):
		super(AttentionFusion, self).__init__()
		self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=1)
		self.init_weight()

	def forward(self, features1, features2):
		feat_concat = torch.cat([features1, features2], dim=1)
		feat_new = self.conv(feat_concat)
		return feat_new

	def init_weight(self):
		for ly in self.children():
			if isinstance(ly, nn.Conv2d):
				nn.init.kaiming_normal_(ly.weight, a=1)
				if not ly.bias is None: nn.init.constant_(ly.bias, 0)

	def get_params(self):
		wd_params, nowd_params = [], []
		for name, module in self.named_modules():
			if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
				wd_params.append(module.weight)
				if not module.bias is None:
					nowd_params.append(module.bias)
			elif isinstance(module, nn.BatchNorm2d):
				nowd_params += list(module.parameters())
		return wd_params, nowd_params



class CABiNet(nn.Module):
	def __init__(self, n_classes, *args, **kwargs):
		super(CABiNet, self).__init__()
		self.mobile = MobileNetV3(pretrained=True, width_mult=1.)
		self.ab = AttentionBranch(576, 128, 128, n_classes)
		self.sb = SpatialBranch()
		self.ffm = FeatureFusionModule(256, 256)
		self.conv_out = CABiNetOutput(256, 256, n_classes)
		self.init_weight()

	def forward(self, x):
		H, W = x.size()[2:]
		feat_sb = self.sb(x)
		mobile_feat = self.mobile(x)

		feat_ab, feat_ab_final = self.ab(mobile_feat)
		
		feat_ab = F.interpolate(feat_ab, (feat_sb.size()[2:]), mode='bilinear', align_corners=True)
		feat_ab_final = F.interpolate(feat_ab_final, (feat_sb.size()[2:]), mode='bilinear', align_corners=True)

		feat_fuse = self.ffm(feat_sb, feat_ab)
		feat_out = self.conv_out(feat_fuse)

		feat_out = F.interpolate(feat_out, (H, W), mode='bilinear', align_corners=True)
		feat_con = F.interpolate(feat_ab_final, (H, W), mode='bilinear', align_corners=True)
		return feat_out, feat_con

	def init_weight(self):
		for ly in self.children():
			if isinstance(ly, nn.Conv2d):
				nn.init.kaiming_normal_(ly.weight, a=1)
				if not ly.bias is None: nn.init.constant_(ly.bias, 0)

	def get_params(self):
		wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params = [], [], [], []
		for name, child in self.named_children():
			child_wd_params, child_nowd_params = child.get_params()
			if isinstance(child, (FeatureFusionModule, CABiNetOutput)):
				lr_mul_wd_params += child_wd_params
				lr_mul_nowd_params += child_nowd_params
			else:
				wd_params += child_wd_params
				nowd_params += child_nowd_params
		return wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params


if __name__ == "__main__":
	net = CABiNet(19)
	net.cuda()
	net.eval()
	in_ten = torch.randn(1, 3, 2048, 1024).cuda()
	start = time.time()
	out, out1, out2 = net(in_ten)
	print(net.mobile.features[:4])
	end = time.time()
	print(out.shape)
	print('TIME in ms: ', (end - start)*1000)