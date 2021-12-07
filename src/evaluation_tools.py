# -*- coding: utf-8 -*-
# @author: caoyang
# @email: caoyang@163.sufe.edu.cn
# �����㷨�����Ĺ��ߺ���
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
	
# ����ģ��
# :param model_name		: ģ�����ƣ�Ŀǰֻ����{'mnl', 'nl', 'ml'}��
# :param model_args		: ģ�����ã����������ɲ�ͬ��ģ�Ͳ�����
# :param algorithm_name	: �㷨���ƣ����������ʵ������Ҫ��ֻ�����ļ���������ΪĿǰ�����㷨Ŀǰ�Ѿ�������ͳһ��BaseAlgorithm�����ʵ�֣�ֻ��Ҫ�޸��㷨���ü���ʵ�ֲ�ͬ���㷨��
# :param algorithm_args	: �㷨���ã��ؼ����������ò�ͬ�����ÿ���ʵ�ֺܶ಻ͬ���㷨��
# :param n_sample		: ģ��ʵ�����������
# :param do_export		: �Ƿ񵼳���ϸ���������
def evaluate(model_name, model_args, algorithm_name, algorithm_args, n_sample=1000, do_export=True):
	# ģ���趨
	model_name = model_name.replace(' ', '').lower()
	assert model_name in MODEL_MAPPING
	Model = eval(MODEL_MAPPING[model_name]['class'])
	generate_params_function = eval(MODEL_MAPPING[model_name]['param'])
	
	# �㷨�趨
	algorithm_name = algorithm_name.replace(' ', '').lower()
	algorithm = BaseAlgorithm(load_args(BaseAlgorithmConfig), **algorithm_args)
	
	# ����������¼���ֵ�
	evaluation_dict = {
		'optimal': [],	# ��¼ÿһ�η����Ƿ�������ȫ������
		'max_revenue': [],	# �������
		'gap': [],			# ��¼ÿһ�η����㷨�����ȫ������֮���������ϵĲ��
	}
	
	start_time = time.time()
	for _ in range(n_sample):
		# �������ģ�Ͳ�����ģ��ʵ��
		model_params = generate_params_function(model_args)
		model = Model(**model_params)
		
		# ��پ�ȷ������е����Ž�
		max_revenue, optimal_solutions = BaseAlgorithm.bruteforce(model=model, 
																  min_size=1, 
																  max_size=model.offerset_capacity)
		
		# �㷨���
		output_max_revenue, output_optimal_solution = algorithm.run(model)
		
		if set(output_optimal_solution) in list(map(set, optimal_solutions)):
			# �㷨������ȫ�����Ž�
			assert abs(max_revenue - output_max_revenue) < EPSILON, f'Untolerable error between {max_revenue} and {output_max_revenue}'
			optimal = 1
			gap = 0
		else:
			# �㷨δ������ȫ�����Ž�
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
		
		# ����ģ������
		save_args(args=model_args, save_path=os.path.join(root, 'model_args.json'))
		
		# �����㷨����
		# save_args(args=algorithm.algorithm_args, save_path=os.path.join(root, 'algorithm_args.json'))
		with open(os.path.join(root, 'algorithm_args.json'), 'w') as f:
			json.dump(algorithm_args, f, indent=4)
		
		# ����������
		evaluation_dataframe.to_csv(os.path.join(root, 'evaluation.csv'), header=True, index=False, sep='\t')
		
		# �����������
		result['model_name'] = model_name
		result['algorithm_name'] = algorithm_name
		result['dirname'] = dirname	# ���ڼ���
		with open(os.path.join(root, 'result.json'), 'w') as f:
			json.dump(result, f, indent=4)
		
	
	return evaluation_dataframe, result

# 20211206�Ľ��ķ���ģ�⺯����������ͬһģ�������¶Զ���㷨ͬʱ�������㣬��Ϊ������Ž�̫��ʱ�䣬������һ��ģ��ʵ���Զ���㷨ͬʱ��������������Ҳ����ƽ
# :param model_name				: ģ�����ƣ�Ŀǰֻ����{'mnl', 'nl', 'ml'}��
# :param model_args				: ģ�����ã����������ɲ�ͬ��ģ�Ͳ�����
# :param algorithm_name_list	: �㷨�����б����������ʵ������Ҫ��ֻ�����ļ���������ΪĿǰ�����㷨Ŀǰ�Ѿ�������ͳһ��BaseAlgorithm�����ʵ�֣�ֻ��Ҫ�޸��㷨���ü���ʵ�ֲ�ͬ���㷨��
# :param algorithm_args_list	: �㷨�����б��ؼ����������ò�ͬ�����ÿ���ʵ�ֺܶ಻ͬ���㷨��
# :param n_sample				: ģ��ʵ�����������
# :param do_export				: �Ƿ񵼳���ϸ���������
def evaluate_new(model_name, model_args, algorithm_name_list, algorithm_args_list, n_sample=1000, do_export=True):
	# ģ���趨
	model_name = model_name.replace(' ', '').lower()
	assert model_name in MODEL_MAPPING
	Model = eval(MODEL_MAPPING[model_name]['class'])
	generate_params_function = eval(MODEL_MAPPING[model_name]['param'])
	
	# �㷨�趨
	algorithm_dict = {}
	for algorithm_name, algorithm_args in zip(algorithm_name_list, algorithm_args_list):
		algorithm_name = algorithm_name.replace(' ', '').lower()
		algorithm = BaseAlgorithm(load_args(BaseAlgorithmConfig), **algorithm_args)
		algorithm_dict[algorithm_name] = algorithm
	
	# ����������¼���ֵ�
	evaluation_dict = {}
	for algorithm_name in algorithm_name_list:
		evaluation_dict[algorithm_name] = {
			'optimal': [],		# ��¼ÿһ�η����Ƿ�������ȫ������
			'max_revenue': [],	# �������
			'gap': [],			# ��¼ÿһ�η����㷨�����ȫ������֮���������ϵĲ��
			'time': [],
		}
	
	for model, max_revenue, optimal_solutions in generate_model_instance_and_solve(model_name=model_name, 
																				   model_args=model_args, 
																				   n_sample=n_sample):
		# ����һ��ģ��ʵ��ʹ�������㷨�������
		for algorithm_name, algorithm in algorithm_dict.items():
			# �㷨���
			start_time = time.time()
			output_max_revenue, output_optimal_solution = algorithm.run(model)
			end_time = time.time()
			if set(output_optimal_solution) in list(map(set, optimal_solutions)):
				# �㷨������ȫ�����Ž�
				assert abs(max_revenue - output_max_revenue) < EPSILON, f'Untolerable error between {max_revenue} and {output_max_revenue}'
				optimal = 1
				gap = 0
			else:
				# �㷨δ������ȫ�����Ž�
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
			
			# ����ģ������
			save_args(args=model_args, save_path=os.path.join(root, 'model_args.json'))
			
			# �����㷨����
			# save_args(args=algorithm.algorithm_args, save_path=os.path.join(root, 'algorithm_args.json'))
			with open(os.path.join(root, 'algorithm_args.json'), 'w') as f:
				json.dump(algorithm_args, f, indent=4)
			
			# ����������
			evaluation_dataframe.to_csv(os.path.join(root, 'evaluation.csv'), header=True, index=False, sep='\t')
			
			# �����������
			result['model_name'] = model_name
			result['algorithm_name'] = algorithm_name
			result['dirname'] = dirname	# ���ڼ���
			with open(os.path.join(root, 'result.json'), 'w') as f:
				json.dump(result, f, indent=4)
		results[algorithm_name] = result

	return evaluation_dict, results


# ����evaluate����ͳ�Ƶ�evaluation_dataframe�������ͳ��ָ���ļ���
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
