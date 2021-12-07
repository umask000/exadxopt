# -*- coding: utf-8 -*-
# @author: caoyang
# @email: caoyang@163.sufe.edu.cn
# 用于绘图的工具函数
if __name__ == '__main__':
	import sys
	sys.path.append('../')

import os
import json
import pandas

from matplotlib import pyplot

from setting import *

# 对manage.py中main, main_offerset_capacity, main_max_addition_or_removal, main_block_size四个函数的结果进行分析
def plot_main_1_to_4(do_export=True, root=TEMP_FOLDER):
	paths = [
		os.path.join(root, 'summary.json'),
		os.path.join(root, 'summary_offerset_capacity.json'),
		os.path.join(root, 'summary_max_addition_or_removal.json'),
		os.path.join(root, 'summary_block_size.json'),
	]	
	summary_dict = {
		'model_name': [],
		'algorithm_name': [],
		'block_size': [],
		'max_addition_or_removal': [],
		'offerset_capacity': [],
		'percentage_of_optimal_instances': [],
		'average_gap_ratio_all': [],
		'average_gap_ratio_nonopt': [],
		'time': [],
	}	
		
	for path in paths:
		summary = json.load(open(path, 'r'))
		for model_name in MODEL_MAPPING.keys():
			for algorithm_name in ALGORITHM_NAMES:
				results = summary[model_name][algorithm_name]
				for result in results:
					percentage_of_optimal_instances = round(result['percentage_of_optimal_instances'], 3)
					average_gap_ratio_nonopt = round(result['average_gap_ratio_nonopt'], 5)
					average_gap_ratio_all = round(result['average_gap_ratio_all'], 8)
					_time = round(result['time'], 1)
					
					offerset_capacity = result.get('offerset_capacity', 10)
					block_size = result.get('block_size', 1)
					max_addition_or_removal = result.get('max_addition_or_removal', 1.)
					
					summary_dict['model_name'].append(model_name)
					summary_dict['algorithm_name'].append(algorithm_name)
					summary_dict['block_size'].append(block_size)
					summary_dict['max_addition_or_removal'].append(max_addition_or_removal)
					summary_dict['offerset_capacity'].append(offerset_capacity)
					summary_dict['percentage_of_optimal_instances'].append(percentage_of_optimal_instances)
					summary_dict['average_gap_ratio_all'].append(average_gap_ratio_all)
					summary_dict['average_gap_ratio_nonopt'].append(average_gap_ratio_nonopt)
					summary_dict['time'].append(_time)
	
	summary_dataframe = pandas.DataFrame(summary_dict, columns=list(summary_dict.keys()))
	if do_export:
		summary_dataframe.to_csv('summary1.csv', header=True, index=False, sep=',')
	
	
	# 分析运行时间
	
	new_summary = {
		'model_name': [],
		'block_size': [],
		'max_addition_or_removal': [],
		'offerset_capacity': [],
		'total_time_backward': [],
		'total_time_forward': [],
		'average_percentage_of_optimal_instances_backward': [],
		'average_percentage_of_optimal_instances_forward': [],
	}
	
	for (model_name, 
		 block_size, 
		 max_addition_or_removal, 
		 offerset_capacity), _summary_dataframe in summary_dataframe.groupby(['model_name', 
																			  'block_size', 
																			  'max_addition_or_removal', 
																			  'offerset_capacity']):
		print(model_name, block_size, max_addition_or_removal, offerset_capacity)		
		_summary_dataframe = _summary_dataframe.reset_index(drop=True)
		_summary_dataframe['suffix'] = _summary_dataframe['algorithm_name'].map(lambda x: x.split('_')[-1])
		
		total_time_backward = _summary_dataframe[_summary_dataframe['suffix'] == 'backward']['time'].sum()
		total_time_forward = _summary_dataframe[_summary_dataframe['suffix'] == 'forward']['time'].sum()
		average_percentage_of_optimal_instances_backward = _summary_dataframe[_summary_dataframe['suffix'] == 'backward']['percentage_of_optimal_instances'].mean()
		average_percentage_of_optimal_instances_forward = _summary_dataframe[_summary_dataframe['suffix'] == 'forward']['percentage_of_optimal_instances'].mean()
		
		new_summary['model_name'].append(model_name)														  
		new_summary['block_size'].append(block_size)														  
		new_summary['max_addition_or_removal'].append(max_addition_or_removal)														  
		new_summary['offerset_capacity'].append(offerset_capacity)	
		new_summary['total_time_backward'].append(total_time_backward)	
		new_summary['total_time_forward'].append(total_time_forward)	
		new_summary['average_percentage_of_optimal_instances_backward'].append(average_percentage_of_optimal_instances_backward)	
		new_summary['average_percentage_of_optimal_instances_forward'].append(average_percentage_of_optimal_instances_forward)	
															  
		# print(_summary_dataframe[['algorithm_name', 'percentage_of_optimal_instances', 'average_gap_ratio_all', 'average_gap_ratio_nonopt', 'time']])
		# print('#' * 64)
	
	print(pandas.DataFrame(new_summary, columns=list(new_summary.keys())).to_csv('summary1.1.csv', sep=',', header=True, index=False))
	
	return summary_dataframe

# 对manage.py中main_adxopt2014_for_nl2_further_analysis函数的结果进行分析
def plot_main_adxopt2014_for_nl2_further_analysis(do_export=True, root=TEMP_FOLDER):
	paths = [
		os.path.join(root, 'summary_adxopt2014_forward_for_nl2_further_analysis_1.json'),
		os.path.join(root, 'summary_adxopt2014_forward_for_nl2_further_analysis_2.json'),
		os.path.join(root, 'summary_adxopt2014_backward_for_nl2_further_analysis_1.json'),
		os.path.join(root, 'summary_adxopt2014_backward_for_nl2_further_analysis_2.json'),
	]
	summary_dict = {
		'algorithm_name': [],
		'block_size': [],
		'num_nest': [],
		'min_dis_similarity': [],
		'max_dis_similarity': [],
		'exist_no_purchase_per_nest': [],
		'allow_nest_repetition': [],

		'percentage_of_optimal_instances': [],
		'average_gap_ratio_all': [],
		'average_gap_ratio_nonopt': [],
		'time': [],
	}		

	
	for path in paths:
		summary = json.load(open(path, 'r'))
		block_size = int(path[-6])
		algorithm_name = '_'.join(path.split('_')[1: 3])
		
		for result in summary:
			num_nest = result['num_nest']
			min_dis_similarity = result['min_dis_similarity']
			max_dis_similarity = result['max_dis_similarity']
			exist_no_purchase_per_nest = int(result['exist_no_purchase_per_nest'])
			allow_nest_repetition = int(result['allow_nest_repetition'])
			percentage_of_optimal_instances = round(result['percentage_of_optimal_instances'], 3)
			average_gap_ratio_nonopt = round(result['average_gap_ratio_nonopt'], 5)
			average_gap_ratio_all = round(result['average_gap_ratio_all'], 8)
			_time = round(result['time'], 1)
			
			summary_dict['algorithm_name'].append(algorithm_name)
			summary_dict['block_size'].append(block_size)
			summary_dict['num_nest'].append(num_nest)
			summary_dict['min_dis_similarity'].append(min_dis_similarity)
			summary_dict['max_dis_similarity'].append(max_dis_similarity)
			summary_dict['exist_no_purchase_per_nest'].append(exist_no_purchase_per_nest)
			summary_dict['allow_nest_repetition'].append(allow_nest_repetition)
			summary_dict['percentage_of_optimal_instances'].append(percentage_of_optimal_instances)
			summary_dict['average_gap_ratio_all'].append(average_gap_ratio_all)
			summary_dict['average_gap_ratio_nonopt'].append(average_gap_ratio_nonopt)
			summary_dict['time'].append(_time)
		
	summary_dataframe = pandas.DataFrame(summary_dict, columns=list(summary_dict.keys()))
	if do_export:
		summary_dataframe.to_csv('summary2.csv', header=True, index=False, sep=',')
	
	
	for (num_nest, 
		 min_dis_similarity, 
		 max_dis_similarity, 
		 exist_no_purchase_per_nest,
		 allow_nest_repetition), _summary_dataframe in summary_dataframe.groupby(['num_nest', 
																				  'min_dis_similarity', 
																				  'max_dis_similarity', 
																				  'exist_no_purchase_per_nest',
																				  'allow_nest_repetition']):
		print(num_nest, min_dis_similarity, max_dis_similarity, exist_no_purchase_per_nest, allow_nest_repetition)																  
		print(_summary_dataframe[['algorithm_name', 'block_size', 'percentage_of_optimal_instances', 'average_gap_ratio_all', 'average_gap_ratio_nonopt', 'time']])
		print('#' * 64)
	
	return summary_dataframe
	
def plot_main_adxopt2014_for_nl2_further_analysis_new(do_export=True, root=TEMP_FOLDER):
	
	path = os.path.join(root, 'summary_adxopt2014_for_nl2_further_analysis_all.json')
	summary = json.load(open(path, 'r'))

	summary_dict = {
		'algorithm_name': [],
		'block_size': [],
		'num_nest': [],
		'min_dis_similarity': [],
		'max_dis_similarity': [],
		'exist_no_purchase_per_nest': [],
		'allow_nest_repetition': [],
		'percentage_of_optimal_instances': [],
		'average_gap_ratio_all': [],
		'average_gap_ratio_nonopt': [],
		'time': [],
	}	
	
	for result in summary:
		num_nest = result['num_nest']
		min_dis_similarity = result['min_dis_similarity']
		max_dis_similarity = result['max_dis_similarity']
		exist_no_purchase_per_nest = result['exist_no_purchase_per_nest']
		allow_nest_repetition = result['allow_nest_repetition']

		
		for full_algorithm_name in result['results']:
			algorithm_name = '_'.join(full_algorithm_name.split('_')[: -1])
			block_size = int(full_algorithm_name.split('_')[-1])
			percentage_of_optimal_instances = round(result['results'][full_algorithm_name]['percentage_of_optimal_instances'], 3)
			average_gap_ratio_all = round(result['results'][full_algorithm_name]['average_gap_ratio_all'], 5)
			average_gap_ratio_nonopt = round(result['results'][full_algorithm_name]['average_gap_ratio_nonopt'], 8)
			_time = round(result['results'][full_algorithm_name]['time'], 1)
			
			summary_dict['algorithm_name'].append(algorithm_name)
			summary_dict['block_size'].append(block_size)
			summary_dict['num_nest'].append(num_nest)
			summary_dict['min_dis_similarity'].append(min_dis_similarity)
			summary_dict['max_dis_similarity'].append(max_dis_similarity)
			summary_dict['exist_no_purchase_per_nest'].append(exist_no_purchase_per_nest)
			summary_dict['allow_nest_repetition'].append(allow_nest_repetition)
			summary_dict['percentage_of_optimal_instances'].append(percentage_of_optimal_instances)
			summary_dict['average_gap_ratio_all'].append(average_gap_ratio_all)
			summary_dict['average_gap_ratio_nonopt'].append(average_gap_ratio_nonopt)
			summary_dict['time'].append(_time)

	summary_dataframe = pandas.DataFrame(summary_dict, columns=list(summary_dict.keys()))
	if do_export:
		summary_dataframe.to_csv('summary3.csv', header=True, index=False, sep=',')

	for (num_nest, 
		 min_dis_similarity, 
		 max_dis_similarity, 
		 exist_no_purchase_per_nest,
		 allow_nest_repetition), _summary_dataframe in summary_dataframe.groupby(['num_nest', 
																				  'min_dis_similarity', 
																				  'max_dis_similarity', 
																				  'exist_no_purchase_per_nest',
																				  'allow_nest_repetition']):
		print(num_nest, min_dis_similarity, max_dis_similarity, exist_no_purchase_per_nest, allow_nest_repetition)																  
		print(_summary_dataframe[['algorithm_name', 'block_size', 'percentage_of_optimal_instances', 'average_gap_ratio_all', 'average_gap_ratio_nonopt', 'time']])
		print('#' * 64)

	return summary_dataframe
	
