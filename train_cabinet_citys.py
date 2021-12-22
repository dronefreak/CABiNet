#!/usr/bin/python
# -*- encoding: utf-8 -*-


from logger import setup_logger

from cabinet import CABiNet
from cityscapes import CityScapes
from loss import OhemCELoss
from evaluate_cabinet_citys import evaluate
from optimizer import Optimizer

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.distributed as dist

from pathlib import Path
import logging
import time
import datetime
import argparse
import json


def train_and_evaluate(params, logger):
	""" Set directories """
	respth = Path(params["training_config"]["experiments_path"])
	Path.mkdir(respth, parents=True, exist_ok=True)

	torch.cuda.set_device(params["training_config"]["gpu_id"])
	dist.init_process_group(
				backend = 'nccl',
				init_method = 'tcp://127.0.0.1:33271',
				world_size = torch.cuda.device_count(),
				rank=params["training_config"]["gpu_id"]
				)
	setup_logger(respth)
	torch.cuda.synchronize()

	""" Set Dataset Params """
	n_classes = params["dataset_config"]["num_classes"]
	n_img_per_gpu = params["training_config"]["batch_size"]
	n_workers = params["training_config"]["num_workers"]
	cropsize = params["dataset_config"]["cropsize"]
	dataset_path = params["dataset_config"]["dataset_path"]
	mode = params["training_config"]["mode"]

	""" Prepare DataLoader """
	ds = CityScapes(params, mode=mode)
	sampler = torch.utils.data.distributed.DistributedSampler(ds)
	dl = DataLoader(ds,
					batch_size = n_img_per_gpu,
					shuffle = False,
					sampler = sampler,
					num_workers = n_workers,
					pin_memory = True,
					drop_last = True)

	""" Set Model of CABiNet """
	ignore_idx = params["dataset_config"]["ignore_idx"]
	net = CABiNet(n_classes=n_classes)
	net.cuda()
	net.train()
	net = nn.parallel.DistributedDataParallel(net,
			device_ids=[params["training_config"]["gpu_id"],],
			output_device=params["training_config"]["gpu_id"],
			find_unused_parameters=True
			)
	score_thres = 0.7
	n_min = n_img_per_gpu*cropsize[0]*cropsize[1]//16
	criteria_p = OhemCELoss(thresh=score_thres, n_min=n_min, ignore_lb=ignore_idx)
	criteria_16 = OhemCELoss(thresh=score_thres, n_min=n_min, ignore_lb=ignore_idx)
	criteria_32 = OhemCELoss(thresh=score_thres, n_min=n_min, ignore_lb=ignore_idx)

	""" Set Optimization Parameters """
	momentum = params["training_config"]["optimizer_momentum"]
	weight_decay = params["training_config"]["optimizer_weight_decay"]
	lr_start = params["training_config"]["optimizer_lr_start"]
	max_iter = params["training_config"]["max_iterations"]
	power = params["training_config"]["optimizer_power"]
	warmup_steps = params["training_config"]["warmup_stemps"]
	warmup_start_lr = params["training_config"]["warmup_start_lr"]
	optim = Optimizer(
			model = net.module,
			lr0 = lr_start,
			momentum = momentum,
			wd = weight_decay,
			warmup_steps = warmup_steps,
			warmup_start_lr = warmup_start_lr,
			max_iter = max_iter,
			power = power)

	""" Set Train Loop Params """
	msg_iter = params["training_config"]["msg_iterations"]
	save_steps = int(params["training_config"]["max_iterations"] / 10)
	best_score = 0.0
	loss_avg = []
	st = glob_st = time.time()
	diter = iter(dl)
	epoch = 0
	logger.info('\n')
	logger.info('===='*20)
	logger.info('[INFO]: Begining Training of Model ...\n')
	for it in range(max_iter):
		try:
			im, lb = next(diter)
			if not im.size()[0]==n_img_per_gpu: raise StopIteration
		except StopIteration:
			epoch += 1
			sampler.set_epoch(epoch)
			diter = iter(dl)
			im, lb = next(diter)
		im = im.cuda()
		lb = lb.cuda()
		H, W = im.size()[2:]
		lb = torch.squeeze(lb, 1)

		optim.zero_grad()
		out, out16, out32 = net(im)
		loss1 = criteria_p(out, lb)
		loss2 = criteria_16(out16, lb)
		loss3 = criteria_32(out32, lb)
		loss = loss1 + loss2 + loss3
		loss.backward()
		optim.step()
		torch.cuda.synchronize()
		loss_avg.append(loss.item())

		""" Log Values """
		if (it + 1) % msg_iter==0:
			loss_avg = sum(loss_avg) / len(loss_avg)
			lr = optim.lr
			ed = time.time()
			t_intv, glob_t_intv = ed - st, ed - glob_st
			eta = int((max_iter - it) * (glob_t_intv / it))
			eta = str(datetime.timedelta(seconds=eta))
			msg = ', '.join([
					'it: {it}/{max_it} || ',
					'lr: {lr:4f} || ',
					'loss: {loss:.4f} || ',
					'eta: {eta} || ',
					'time: {time:.4f}',
				]).format(
					it = it+1,
					max_it = max_iter,
					lr = lr,
					loss = loss_avg,
					time = t_intv,
					eta = eta
				)
			logger.info(msg)
			loss_avg = []
			st = ed
		
		if (it + 1) % save_steps == 0:
			save_name = params["training_config"]["model_save_name"].split(".pth")[0] + f"_iter_{it + 1}.pth"
			save_pth = respth / save_name
			net.cpu()
			state = net.module.state_dict() if hasattr(net, 'module') else net.state_dict()
			if dist.get_rank()==0: torch.save(state, str(save_pth))
			logger.info(f'{it + 1} iterations Finished!; Model Saved to: {save_pth}')
			current_score = evaluate(params=params, save_pth=save_pth)
			if current_score > best_score:
				save_name = params["training_config"]["model_save_name"].split(".pth")[0] + f"_iter_{it + 1}_best_mIOU_{current_score:.4f}.pth"
				save_pth = respth / save_name
				state = net.module.state_dict() if hasattr(net, 'module') else net.state_dict()
				if dist.get_rank()==0: torch.save(state, str(save_pth))
				best_score = current_score
			net.cuda()
			torch.cuda.synchronize()

	""" Dump and Save the Final Model """
	save_pth = respth / params["training_config"]["model_save_name"]
	net.cpu()
	state = net.module.state_dict() if hasattr(net, 'module') else net.state_dict()
	if dist.get_rank()==0: torch.save(state, str(save_pth))
	logger.info('Training Finished!; Model Saved to: {}'.format(save_pth))
	torch.cuda.empty_cache()


if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument('--config',
					type=str,
					default="training_validation_config.json",)
	args = parser.parse_args()
	with open(args.config, "r") as f:
		params = json.loads(f.read())
	logger = logging.getLogger()

	train_and_evaluate(params, logger)