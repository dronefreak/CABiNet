import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn import BatchNorm2d


class PSPModule(nn.Module):
	'''
	Implementation of pyramid spatial pyramid pooling block at 4 sclaes
	Input:
		N X C X H X W
	Parameters:
		sizes       : the scales of adaptive averagepooling
	Return:
		N X C X H X W
		center		: scale-aggregated features
	'''
	# (1, 2, 3, 6)
	def __init__(self, sizes=(1, 3, 6, 8), dimension=2):
		super(PSPModule, self).__init__()
		self.stages = nn.ModuleList([self._make_stage(size, dimension) for size in sizes])

	def _make_stage(self, size, dimension=2):
		if dimension == 1:
			prior = nn.AdaptiveAvgPool1d(output_size=size)
		elif dimension == 2:
			prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
		elif dimension == 3:
			prior = nn.AdaptiveAvgPool3d(output_size=(size, size, size))
		return prior

	def forward(self, feats):
		n, c, _, _ = feats.size()
		priors = [stage(feats).view(n, c, -1) for stage in self.stages]
		center = torch.cat(priors, -1)
		return center


class ReducedGlobalAttention(nn.Module):
	'''
	Implementation for compact global attention block
	Input:
		N X C X H X W
	Parameters:
		in_channels       : the dimension of the input feature map
		key_channels      : the dimension after the key/query transform
		value_channels    : the dimension after the value transform
		scale             : choose the scale to downsample the input feature maps (save memory cost)
	Return:
		N X C X H X W
		context			  : globally-aware contextual features.(w/o concate or add with the input)
	'''

	def __init__(self, in_channels, key_channels, value_channels, out_channels=None, scale=1, norm_type=None,psp_size=(1,3,6,8)):
		super(ReducedGlobalAttention, self).__init__()
		self.scale = scale
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.key_channels = key_channels
		self.value_channels = value_channels
		if out_channels == None:
			self.out_channels = in_channels
		self.pool = nn.MaxPool2d(kernel_size=(scale, scale))
		self.f_key = nn.Sequential(
			nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
					  kernel_size=1, stride=1, padding=0),
			nn.BatchNorm2d(self.key_channels),
			nn.ReLU(),
		)
		self.f_query = self.f_key
		self.f_value = nn.Conv2d(in_channels=self.in_channels, out_channels=self.value_channels,
								 kernel_size=1, stride=1, padding=0)
		self.W = nn.Conv2d(in_channels=self.value_channels, out_channels=self.out_channels,
						   kernel_size=1, stride=1, padding=0)

		self.psp = PSPModule(psp_size)
		nn.init.constant_(self.W.weight, 0)
		nn.init.constant_(self.W.bias, 0)

	def forward(self, x):
		batch_size, h, w = x.size(0), x.size(2), x.size(3)
		if self.scale > 1:
			x = self.pool(x)

		value = self.psp(self.f_value(x))
		query = self.f_query(x).view(batch_size, self.key_channels, -1)
		query = query.permute(0, 2, 1)
		key = self.f_key(x)
		value = value.permute(0, 2, 1)
		key = self.psp(key)
		sim_map = torch.bmm(query, key)
		sim_map = (self.key_channels ** -.5) * sim_map
		sim_map = F.softmax(sim_map, dim=-1)

		context = torch.bmm(sim_map, value)
		context = context.permute(0, 2, 1).contiguous()
		context = context.view(batch_size, self.value_channels, *x.size()[2:])
		context = self.W(context)
		return context

class ContextAggregationBlock(nn.Module):
	'''
	Implementation for the propsed CAB
	Input:
		N X C X H X W
	Parameters:
		inplane       : the dimension of the input feature map
		plane      	  : the dimension after the inplane transform
	Return:
		N X C X H X W
		globally and postion-aware contextual features.(w/o concate or add with the input)
	'''
	def __init__(self, inplane, plane):
		super(ContextAggregationBlock, self).__init__()

		self.long_relation = ReducedGlobalAttention(inplane, inplane, plane)
		self.local_attention = LocalAttention(inplane)

	def forward(self, x):
		size = x.size()[2:]
		# global attention
		x = self.long_relation(x)
		# local attention
		x = F.interpolate(x,size=size, mode="bilinear", align_corners=True)
		res = x
		x = self.local_attention(x)
		return x + res


class LocalAttention(nn.Module):
	'''
	Implementation for the propsed Local Attention block
	Input:
		N X C X H X W
	Parameters:
		plane      : channels for analysis
	Return:
		N X C X H X W
		globally and postion-aware contextual features.(w/o concate or add with the input)
	'''
	def __init__(self, inplane):
		super(LocalAttention, self).__init__()
		self.dconv1 = _DWConv(inplane, inplane, kernel_size=3, stride=2)
		self.dconv2 = _DWConv(inplane, inplane, kernel_size=3, stride=2)
		self.dconv3 = _DWConv(inplane, inplane, kernel_size=1, stride=1)
		self.sigmoid_spatial = nn.Sigmoid()

	def forward(self, x):
		b, c, h, w = x.size()
		res_1 = x
		res_2 = x
		x = self.dconv1(x)
		x = self.dconv2(x)
		x = self.dconv3(x)
		x = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=True)
		x_mask = self.sigmoid_spatial(x)

		res_1 = res_1 * x_mask

		return res_2 + res_1



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



if __name__ == '__main__':
	model = ContextAggregationBlock(512, 512//4)
	model.eval()
	model.cuda()
	img = torch.randn(1, 512, 32, 64).cuda()
	output = model(img)
	print(output.shape)