# -*- coding: utf-8 -*-
# @author: caoyang
# @email: caoyang@163.sufe.edu.cn
# 主程序

import os
import json

from copy import deepcopy

from config import *
from setting import *

from src.simulation_tools import generate_algorithm_args
from src.evaluation_tools import evaluate, analysis
from src.utils import load_args

# 默认的模型配置在默认的算法配置上的情况：
# 1. 报价集无容量限制
# 2. 移除或移入次数的上限都为1
# 3. 不考虑块级调整
def main(n_samples=1000):
	summary = {}
	for model_name in MODEL_MAPPING.keys():
		summary[model_name] = {}
		model_args = load_args(eval(MODEL_MAPPING[model_name]['config']))
		for algorithm_name in ALGORITHM_NAMES:
			print(model_name, algorithm_name)
			algorithm_args = generate_algorithm_args(algorithm_name=algorithm_name)
			_, result = evaluate(model_name=model_name, 
								 model_args=model_args, 
								 algorithm_name=algorithm_name, 
								 algorithm_args=algorithm_args, 
								 n_sample=n_samples, 
								 do_export=True)
			summary[model_name][algorithm_name] = [result]
	with open(os.path.join(TEMP_FOLDER, 'summary.json'), 'w') as f:
		json.dump(summary, f, indent=4)

# 情形一：考虑报价集容量限制
def main_offerset_capacity(n_samples=1000):
	capacity_ratios = [0.25, 0.5, 0.75]
	summary = {}
	for model_name in MODEL_MAPPING.keys():
		summary[model_name] = {}
		model_args = load_args(eval(MODEL_MAPPING[model_name]['config']))
		for algorithm_name in ALGORITHM_NAMES:
			summary[model_name][algorithm_name] = []
			for capacity_ratio in capacity_ratios:
				offerset_capacity = int(model_args.num_product * capacity_ratio)
				model_args.offerset_capacity = offerset_capacity	# 修改报价集容量限制
				print(model_name, algorithm_name, offerset_capacity)
				algorithm_args = generate_algorithm_args(algorithm_name=algorithm_name)
				_, result = evaluate(model_name=model_name, 
									 model_args=model_args, 
									 algorithm_name=algorithm_name, 
									 algorithm_args=algorithm_args, 
									 n_sample=n_samples, 
									 do_export=True)
				result['offerset_capacity'] = offerset_capacity
				summary[model_name][algorithm_name].append(result)
	with open(os.path.join(TEMP_FOLDER, 'summary_offerset_capacity.json'), 'w') as f:
		json.dump(summary, f, indent=4)

# 情形二：增加移除或移入次数的上限
def main_max_addition_or_removal(n_samples=1000):
	max_addition_or_removals = [2, 4, 8, 16]
	summary = {}
	for model_name in MODEL_MAPPING.keys():
		summary[model_name] = {}
		model_args = load_args(eval(MODEL_MAPPING[model_name]['config']))
		for algorithm_name in ALGORITHM_NAMES:
			summary[model_name][algorithm_name] = []
			for max_addition_or_removal in max_addition_or_removals:
				print(model_name, algorithm_name, max_addition_or_removal)
				if algorithm_name.endswith('_forward'):
					kwargs = {'max_removal': max_addition_or_removal}
				elif algorithm_name.endswith('_backward'):
					kwargs = {'max_addition': max_addition_or_removal}
				else:
					raise NotImplementedError
				algorithm_args = generate_algorithm_args(algorithm_name=algorithm_name, **kwargs)
				_, result = evaluate(model_name=model_name, 
									 model_args=model_args, 
									 algorithm_name=algorithm_name, 
									 algorithm_args=algorithm_args, 
									 n_sample=n_samples, 
									 do_export=True)
				result['max_addition_or_removal'] = max_addition_or_removal
				summary[model_name][algorithm_name].append(result)
	with open(os.path.join(TEMP_FOLDER, 'summary_max_addition_or_removal.json'), 'w') as f:
		json.dump(summary, f, indent=4)

# 情形三：考虑块级调整策略
def main_block_size(n_samples=1000):
	block_sizes = [2, 3, 4, 5]
	summary = {}
	for model_name in MODEL_MAPPING.keys():
		summary[model_name] = {}
		model_args = load_args(eval(MODEL_MAPPING[model_name]['config']))
		for algorithm_name in ALGORITHM_NAMES:
			summary[model_name][algorithm_name] = []
			for block_size in block_sizes:
				print(model_name, algorithm_name, block_size)
				kwargs = {
					'addable_block_size'		: block_size,
					'deleteable_block_size'		: block_size,
					'exchangeable_block_size'	: block_size,
				}
				algorithm_args = generate_algorithm_args(algorithm_name=algorithm_name, **kwargs)
				_, result = evaluate(model_name=model_name, 
									 model_args=model_args, 
									 algorithm_name=algorithm_name, 
									 algorithm_args=algorithm_args, 
									 n_sample=n_samples, 
									 do_export=True)
				result['block_size'] = block_size
				summary[model_name][algorithm_name].append(result)
	with open(os.path.join(TEMP_FOLDER, 'summary_block_size.json'), 'w') as f:
		json.dump(summary, f, indent=4)
		
# 深度分析ADXOpt2014在多个嵌套逻辑模型上的效果，研究前后向算法与块级调整两个因素的影响
def main_adxopt2014_for_nl2_further_analysis(n_samples=1000):
	model_name = 'nl2'
	algorithm_name = 'adxopt2014_forward'
	model_args = load_args(eval(MODEL_MAPPING[model_name]['config']))
	algorithm_args = generate_algorithm_args(algorithm_name=algorithm_name)
	
	for algorithm_name in ['adxopt2014_forward', 'adxopt2014_backward']:
		for block_size in [1, 2]:
			kwargs = {
				'addable_block_size'		: block_size,
				'deleteable_block_size'		: block_size,
				'exchangeable_block_size'	: block_size,
			}
			algorithm_args = generate_algorithm_args(algorithm_name=algorithm_name, **kwargs)
			if algorithm_name == 'adxopt2014_forward' and block_size == 1:
				continue
			summary = []
			count = 0
			for num_nest in [2, 3, 4]:
				for min_dis_similarity, max_dis_similarity in zip([0., 1.], [1., 10.]):
					for exist_no_purchase_per_nest in [True, False]:
						for allow_nest_repetition in [True, False]:
							count += 1
							print(count, algorithm_name, block_size, num_nest, min_dis_similarity, max_dis_similarity, exist_no_purchase_per_nest, allow_nest_repetition)
							_model_args = deepcopy(model_args)
							_model_args.num_nest = num_nest
							_model_args.min_dis_similarity = min_dis_similarity
							_model_args.max_dis_similarity = max_dis_similarity
							_model_args.exist_no_purchase_per_nest = exist_no_purchase_per_nest
							_model_args.allow_nest_repetition = allow_nest_repetition
							
							_algorithm_args = deepcopy(algorithm_args)
							_, result = evaluate(model_name=model_name, 
												 model_args=_model_args, 
												 algorithm_name=algorithm_name, 
												 algorithm_args=_algorithm_args, 
												 n_sample=n_samples, 
												 do_export=True)
							result['num_nest'] = num_nest
							result['min_dis_similarity'] = min_dis_similarity
							result['max_dis_similarity'] = max_dis_similarity
							result['exist_no_purchase_per_nest'] = exist_no_purchase_per_nest
							result['allow_nest_repetition'] = allow_nest_repetition
							summary.append(result)
			
			with open(os.path.join(TEMP_FOLDER, f'summary_{algorithm_name}_for_nl2_further_analysis_{block_size}.json'), 'w') as f:
				json.dump(summary, f, indent=4)	

	


if __name__ == '__main__':
	# evaluate(model_name='mnl', 
			 # model_args=load_args(MNLConfig), 
			 # algorithm_name='naivegreedy_backward', 
			 # algorithm_args=generate_algorithm_args(algorithm_name='naivegreedy_forward'), 
			 # n_sample=100, 
			 # do_export=True)
			 
	# main(1000)
	# main_offerset_capacity(1000)
	# main_max_addition_or_removal(1000)
	# main_block_size(1000)

	main_adxopt2014_for_nl2_further_analysis(1000)
