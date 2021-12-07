# -*- coding: utf-8 -*-
# @author: caoyang
# @email: caoyang@163.sufe.edu.cn
# 用于算法评估的工具函数
if __name__ == '__main__':
	import sys
	sys.path.append('../')

import os
import time
import json
import pandas

from config import *
from setting import *

from src.algorithm import BaseAlgorithm, NaiveGreedy, GreedyOpt, ADXOpt2014, ADXOpt2016
from src.choice_model import MultiNomialLogit, NestedLogit2, MixedLogit
from src.simulation_tools import generate_params_for_MNL, generate_params_for_NL2, generate_params_for_ML, generate_model_instance_and_solve
from src.utils import load_args, save_args
	
# 仿真模拟
# :param model_name		: 模型名称，目前只考虑{'mnl', 'nl', 'ml'}；
# :param model_args		: 模型配置，可用于生成不同的模型参数；
# :param algorithm_name	: 算法名称，这个参数其实并不重要，只用于文件命名，因为目前所有算法目前已经可以在统一的BaseAlgorithm框架下实现，只需要修改算法配置即可实现不同的算法；
# :param algorithm_args	: 算法配置，关键参数，设置不同的配置可以实现很多不同的算法；
# :param n_sample		: 模型实例仿真次数；
# :param do_export		: 是否导出详细评估结果；
def evaluate(model_name, model_args, algorithm_name, algorithm_args, n_sample=1000, do_export=True):
	# 模型设定
	model_name = model_name.replace(' ', '').lower()
	assert model_name in MODEL_MAPPING
	Model = eval(MODEL_MAPPING[model_name]['class'])
	generate_params_function = eval(MODEL_MAPPING[model_name]['param'])
	
	# 算法设定
	algorithm_name = algorithm_name.replace(' ', '').lower()
	algorithm = BaseAlgorithm(load_args(BaseAlgorithmConfig), **algorithm_args)
	
	# 用于评估记录的字典
	evaluation_dict = {
		'optimal': [],	# 记录每一次仿真是否收敛到全局最优
		'max_revenue': [],	# 最大收益
		'gap': [],			# 记录每一次仿真算法输出与全局最优之间在收益上的差距
	}
	
	start_time = time.time()
	for _ in range(n_sample):
		# 随机生成模型参数与模型实例
		model_params = generate_params_function(model_args)
		model = Model(**model_params)
		
		# 穷举精确求解所有的最优解
		max_revenue, optimal_solutions = BaseAlgorithm.bruteforce(model=model, 
																  min_size=1, 
																  max_size=model.offerset_capacity)
		
		# 算法求解
		output_max_revenue, output_optimal_solution = algorithm.run(model)
		
		if set(output_optimal_solution) in list(map(set, optimal_solutions)):
			# 算法收敛到全局最优解
			assert abs(max_revenue - output_max_revenue) < EPSILON, f'Untolerable error between {max_revenue} and {output_max_revenue}'
			optimal = 1
			gap = 0
		else:
			# 算法未收敛到全局最优解
			assert max_revenue > output_max_revenue - EPSILON
			optimal = 0
			gap = max_revenue - output_max_revenue
		evaluation_dict['optimal'].append(optimal)
		evaluation_dict['max_revenue'].append(max_revenue)
		evaluation_dict['gap'].append(gap)	
	end_time = time.time()
	
	evaluation_dataframe = pandas.DataFrame(evaluation_dict, columns=list(evaluation_dict.keys()))
	result = analysis(evaluation_dataframe)
	result['time'] = end_time - start_time
	result['n_sample'] = n_sample
	if do_export:
		dirname = f'{model_name}-{algorithm_name}-{n_sample}-{int(time.time()) % 100000}'
		root = os.path.join(LOGGING_FOLDER, dirname)
		os.makedirs(root, exist_ok=True)
		
		# 导出模型配置
		save_args(args=model_args, save_path=os.path.join(root, 'model_args.json'))
		
		# 导出算法配置
		# save_args(args=algorithm.algorithm_args, save_path=os.path.join(root, 'algorithm_args.json'))
		with open(os.path.join(root, 'algorithm_args.json'), 'w') as f:
			json.dump(algorithm_args, f, indent=4)
		
		# 导出仿真结果
		evaluation_dataframe.to_csv(os.path.join(root, 'evaluation.csv'), header=True, index=False, sep='\t')
		
		# 导出分析结果
		result['model_name'] = model_name
		result['algorithm_name'] = algorithm_name
		result['dirname'] = dirname	# 便于检索
		with open(os.path.join(root, 'result.json'), 'w') as f:
			json.dump(result, f, indent=4)
		
	
	return evaluation_dataframe, result

# 20211206改进的仿真模拟函数，可以在同一模型配置下对多个算法同时进行演算，因为穷举最优解太费时间，可以用一个模型实例对多个算法同时进行评估，这样也更公平
# :param model_name				: 模型名称，目前只考虑{'mnl', 'nl', 'ml'}；
# :param model_args				: 模型配置，可用于生成不同的模型参数；
# :param algorithm_name_list	: 算法名称列表，这个参数其实并不重要，只用于文件命名，因为目前所有算法目前已经可以在统一的BaseAlgorithm框架下实现，只需要修改算法配置即可实现不同的算法；
# :param algorithm_args_list	: 算法配置列表，关键参数，设置不同的配置可以实现很多不同的算法；
# :param n_sample				: 模型实例仿真次数；
# :param do_export				: 是否导出详细评估结果；
def evaluate_new(model_name, model_args, algorithm_name_list, algorithm_args_list, n_sample=1000, do_export=True):
	# 模型设定
	model_name = model_name.replace(' ', '').lower()
	assert model_name in MODEL_MAPPING
	Model = eval(MODEL_MAPPING[model_name]['class'])
	generate_params_function = eval(MODEL_MAPPING[model_name]['param'])
	
	# 算法设定
	algorithm_dict = {}
	for algorithm_name, algorithm_args in zip(algorithm_name_list, algorithm_args_list):
		algorithm_name = algorithm_name.replace(' ', '').lower()
		algorithm = BaseAlgorithm(load_args(BaseAlgorithmConfig), **algorithm_args)
		algorithm_dict[algorithm_name] = algorithm
	
	# 用于评估记录的字典
	evaluation_dict = {}
	for algorithm_name in algorithm_name_list:
		evaluation_dict[algorithm_name] = {
			'optimal': [],		# 记录每一次仿真是否收敛到全局最优
			'max_revenue': [],	# 最大收益
			'gap': [],			# 记录每一次仿真算法输出与全局最优之间在收益上的差距
			'time': [],
		}
	
	for model, max_revenue, optimal_solutions in generate_model_instance_and_solve(model_name=model_name, 
																				   model_args=model_args, 
																				   n_sample=n_sample):
		# 对于一个模型实例使用所有算法进行求解
		for algorithm_name, algorithm in algorithm_dict.items():
			# 算法求解
			start_time = time.time()
			output_max_revenue, output_optimal_solution = algorithm.run(model)
			end_time = time.time()
			if set(output_optimal_solution) in list(map(set, optimal_solutions)):
				# 算法收敛到全局最优解
				assert abs(max_revenue - output_max_revenue) < EPSILON, f'Untolerable error between {max_revenue} and {output_max_revenue}'
				optimal = 1
				gap = 0
			else:
				# 算法未收敛到全局最优解
				assert max_revenue > output_max_revenue - EPSILON
				optimal = 0
				gap = max_revenue - output_max_revenue
			_time = end_time - start_time
			evaluation_dict[algorithm_name]['optimal'].append(optimal)
			evaluation_dict[algorithm_name]['max_revenue'].append(max_revenue)
			evaluation_dict[algorithm_name]['gap'].append(gap)	
			evaluation_dict[algorithm_name]['time'].append(_time)	
	
	results = {}
	for algorithm_name, algorithm_args in zip(algorithm_name_list, algorithm_args_list):
		evaluation_dataframe = pandas.DataFrame(evaluation_dict[algorithm_name], columns=list(evaluation_dict[algorithm_name].keys()))
		result = analysis(evaluation_dataframe)
		result['time'] = evaluation_dataframe['time'].sum()
		result['n_sample'] = n_sample
		if do_export:
			dirname = f'{model_name}-{algorithm_name}-{n_sample}-{int(time.time()) % 100000}'
			root = os.path.join(LOGGING_FOLDER, dirname)
			os.makedirs(root, exist_ok=True)
			
			# 导出模型配置
			save_args(args=model_args, save_path=os.path.join(root, 'model_args.json'))
			
			# 导出算法配置
			# save_args(args=algorithm.algorithm_args, save_path=os.path.join(root, 'algorithm_args.json'))
			with open(os.path.join(root, 'algorithm_args.json'), 'w') as f:
				json.dump(algorithm_args, f, indent=4)
			
			# 导出仿真结果
			evaluation_dataframe.to_csv(os.path.join(root, 'evaluation.csv'), header=True, index=False, sep='\t')
			
			# 导出分析结果
			result['model_name'] = model_name
			result['algorithm_name'] = algorithm_name
			result['dirname'] = dirname	# 便于检索
			with open(os.path.join(root, 'result.json'), 'w') as f:
				json.dump(result, f, indent=4)
		results[algorithm_name] = result

	return evaluation_dict, results


# 根据evaluate函数统计的evaluation_dataframe进行相关统计指数的计算
def analysis(evaluation_dataframe):
	evaluation_dataframe_optimal = evaluation_dataframe[evaluation_dataframe['optimal']==1]
	evaluation_dataframe_unoptimal = evaluation_dataframe[evaluation_dataframe['optimal']==0]
	gap_ratio_all = evaluation_dataframe['gap'] / evaluation_dataframe['max_revenue']
	gap_ratio_nonopt = evaluation_dataframe_unoptimal['gap'] / evaluation_dataframe_unoptimal['max_revenue']
	
	total_case = evaluation_dataframe.shape[0]
	num_optimal_case = evaluation_dataframe_optimal.shape[0]
	num_unoptimal_case = evaluation_dataframe_unoptimal.shape[0]
	percentage_of_optimal_instances = num_optimal_case / total_case
	average_gap_ratio_all = gap_ratio_all.mean()
	average_gap_ratio_nonopt = 0. if evaluation_dataframe_unoptimal.shape[0] == 0 else gap_ratio_nonopt.mean()

	return {
		'num_optimal_case': num_optimal_case,
		'num_unoptimal_case': num_unoptimal_case,
		'percentage_of_optimal_instances': percentage_of_optimal_instances,
		'average_gap_ratio_all': average_gap_ratio_all,
		'average_gap_ratio_nonopt': average_gap_ratio_nonopt,
	}
