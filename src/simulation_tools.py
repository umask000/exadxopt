# -*- coding: utf-8 -*-
# @author: caoyang
# @email: caoyang@163.sufe.edu.cn
# 用于仿真的工具函数
if __name__ == '__main__':
    import sys
    sys.path.append('../')

import numpy
from copy import deepcopy

from setting import *
from src.algorithm import BaseAlgorithm, NaiveGreedy, GreedyOpt, ADXOpt2014, ADXOpt2016
from src.choice_model import MultiNomialLogit, NestedLogit2, MixedLogit
from src.utils import random_split

# 多项逻辑模型实例化
def generate_params_for_MNL(args):
	# 提取配置参数
	num_product = args.num_product
	offerset_capacity = args.offerset_capacity
	max_product_price = args.max_product_price
	min_product_price = args.min_product_price
	max_product_valuation = args.max_product_valuation
	min_product_valuation = args.min_product_valuation
	
	# 随机生成模型参数
	product_prices = (max_product_price - min_product_price) * numpy.random.rand(num_product) + min_product_price					# 产品价格
	product_valuations = (max_product_valuation - min_product_valuation) * numpy.random.rand(num_product) + min_product_valuation	# 产品估值
	no_purchase_valuation =  (max_product_valuation - min_product_valuation) * numpy.random.random() + min_product_valuation		# 不购买估值
	
	# 打包参数
	params = {
		'product_prices'		: product_prices,
		'product_valuations'	: product_valuations,
		'no_purchase_valuation'	: no_purchase_valuation,
		'offerset_capacity'		: offerset_capacity,
	}
	return params

# 嵌套逻辑模型实例化
def generate_params_for_NL2(args):
	# 提取配置参数
	num_product = args.num_product
	offerset_capacity = args.offerset_capacity
	max_product_price = args.max_product_price
	min_product_price = args.min_product_price
	max_product_valuation = args.max_product_valuation
	min_product_valuation = args.min_product_valuation
	
	num_nest = args.num_nest
	max_dis_similarity = args.max_dis_similarity
	min_dis_similarity = args.min_dis_similarity
	exist_no_purchase_per_nest = args.exist_no_purchase_per_nest
	allow_nest_repetition = args.allow_nest_repetition
	
	# 随机生成模型参数
	product_prices = (max_product_price - min_product_price) * numpy.random.rand(num_product) + min_product_price					# 产品价格
	product_valuations = (max_product_valuation - min_product_valuation) * numpy.random.rand(num_product) + min_product_valuation	# 产品估值			
	no_purchase_valuation = (max_product_valuation - min_product_valuation) * numpy.random.random() + min_product_valuation			# 不购买估值	
	nest_dis_similaritys = (max_dis_similarity - min_dis_similarity) * numpy.random.random(num_nest) + min_dis_similarity			# 嵌套相异度参数																			# 产品嵌套
	if exist_no_purchase_per_nest:																									# 每个嵌套的不购买估值
		nest_no_purchase_valuations = (max_product_valuation - min_product_valuation) * numpy.random.rand(num_nest) + min_product_valuation  
	else: 
		nest_no_purchase_valuations = numpy.zeros((num_nest, ))
	
	if allow_nest_repetition:											# 允许一个产品出现在多个嵌套内的随机分组
		nests = [numpy.random.choice(a=list(range(num_product)), 
								     size=numpy.random.randint(1, num_product + 1), 
								     replace=False) for _ in range(num_nest)]
	else:																# 不允许一个产品出现在多个嵌套内的随机分组
		nests = random_split(array=list(range(num_product)), 
							 n_splits=num_nest,
							 do_shuffle=True,
							 do_balance=False)	

	# 打包参数
	params = {
		'product_prices'				: product_prices,
		'product_valuations'			: product_valuations,	
		'no_purchase_valuation'			: no_purchase_valuation,
		'offerset_capacity'				: offerset_capacity,
		'nests'							: nests,
		'nest_dis_similaritys'			: nest_dis_similaritys,
		'nest_no_purchase_valuations'	: nest_no_purchase_valuations,
	}
	return params

# 混合逻辑模型实例化
def generate_params_for_ML(args):
	# 提取配置参数
	num_product = args.num_product
	offerset_capacity = args.offerset_capacity
	max_product_price = args.max_product_price
	min_product_price = args.min_product_price
	max_product_valuation = args.max_product_valuation
	min_product_valuation = args.min_product_valuation
	
	num_class = args.num_class
	
	# 随机生成模型参数
	product_prices = (max_product_price - min_product_price) * numpy.random.rand(num_product) + min_product_price								# 产品价格
	product_valuations = (max_product_valuation - min_product_valuation) * numpy.random.rand(num_class, num_product) + min_product_valuation	# 产品估值
	no_purchase_valuation = (max_product_valuation - min_product_valuation) * numpy.random.rand(num_class) + min_product_valuation				# 不购买估值
	sorted_random_array = numpy.sort(numpy.random.rand(num_class - 1))
	_class_weight = sorted_random_array[1: ] - sorted_random_array[: -1]
	_class_weight = numpy.append(_class_weight, sorted_random_array[0])
	class_weight = numpy.append(_class_weight, 1 - sorted_random_array[-1])
	
	# 打包参数
	params = {
		'product_prices'		: product_prices,
		'product_valuations'	: product_valuations,	
		'no_purchase_valuation'	: no_purchase_valuation,
		'offerset_capacity'		: offerset_capacity,
		'class_weight'			: class_weight,
	}
	return params

# 随机生成给定配置下的模型实例，并求解最优解的生成器
def generate_model_instance_and_solve(model_name, model_args, n_sample=1000):
	model_name = model_name.replace(' ', '').lower()
	assert model_name in MODEL_MAPPING
	Model = eval(MODEL_MAPPING[model_name]['class'])
	generate_params_function = eval(MODEL_MAPPING[model_name]['param'])
		
	for _ in range(n_sample):
		# 随机生成模型参数与模型实例
		model_params = generate_params_function(model_args)
		model = Model(**model_params)
		
		# 穷举精确求解所有的最优解
		max_revenue, optimal_solutions = BaseAlgorithm.bruteforce(model=model, 
																  min_size=1, 
																  max_size=model.offerset_capacity)
		
		yield model, max_revenue, optimal_solutions
		
		
# 算法实例化
def generate_algorithm_args(algorithm_name, **kwargs):
	if algorithm_name == 'naivegreedy_forward':
		# 平凡的正向贪心算法
		params = {
			'do_add'					: True,
			'do_add_first'				: True,
			'do_delete'					: False,
			'do_delete_first'			: False,
			'do_exchange'				: False,
			'max_removal'				: 0.,
			'max_addition'				: float('inf'),
			'initial_size'				: kwargs.get('initial_size', 0),
			'addable_block_size'		: kwargs.get('addable_block_size', 1),
			'deleteable_block_size'		: 1,
			'exchangeable_block_size'	: 1,
		}

	elif algorithm_name == 'naivegreedy_backward':
		# 平凡的反向贪心算法
		params = {
			'do_add'					: False,
			'do_add_first'				: False,
			'do_delete'					: True,
			'do_delete_first'			: True,
			'do_exchange'				: False,
			'max_removal'				: float('inf'),
			'max_addition'				: 0.,
			'initial_size'				: kwargs.get('initial_size', -1),	# -1表示默认从全集开始搜索，-2则表示从比全集少一个元素的子集开始搜索，以此类推
			'addable_block_size'		: 1,
			'deleteable_block_size'		: kwargs.get('deleteable_block_size', 1),
			'exchangeable_block_size'	: 1,
		}

	elif algorithm_name == 'greedyopt_forward':
		# 正向的2011年GreedyOpt算法
		params = {
			'do_add'					: True,
			'do_add_first'				: True,
			'do_delete'					: False,
			'do_delete_first'			: False,
			'do_exchange'				: True,
			'max_removal'				: kwargs.get('max_removal', 1),
			'max_addition'				: float('inf'),
			'initial_size'				: kwargs.get('initial_size', 0),
			'addable_block_size'		: kwargs.get('addable_block_size', 1),
			'deleteable_block_size'		: 1,
			'exchangeable_block_size'	: kwargs.get('exchangeable_block_size', 1),
		}

	elif algorithm_name == 'greedyopt_backward':
		# 反向的2011年GreedyOpt算法
		params = {
			'do_add'					: False,
			'do_add_first'				: False,
			'do_delete'					: True,
			'do_delete_first'			: True,
			'do_exchange'				: True,
			'max_removal'				: float('inf'),
			'max_addition'				: kwargs.get('max_addition', 1),
			'initial_size'				: kwargs.get('initial_size', -1),
			'addable_block_size'		: 1,
			'deleteable_block_size'		: kwargs.get('deleteable_block_size', 1),
			'exchangeable_block_size'	: kwargs.get('exchangeable_block_size', 1),
		}
	
	elif algorithm_name == 'adxopt2014_forward':
		# 正向的2014年ADXOpt算法
		params = {
			'do_add'					: True,
			'do_add_first'				: False,
			'do_delete'					: True,
			'do_delete_first'			: False,
			'do_exchange'				: True,
			'max_removal'				: kwargs.get('max_removal', 1),
			'max_addition'				: float('inf'),
			'initial_size'				: kwargs.get('initial_size', 0),
			'addable_block_size'		: kwargs.get('addable_block_size', 1),
			'deleteable_block_size'		: kwargs.get('deleteable_block_size', 1),
			'exchangeable_block_size'	: kwargs.get('exchangeable_block_size', 1),
		}
		
	elif algorithm_name == 'adxopt2014_backward':
		# 反向的2014年ADXOpt算法
		params = {
			'do_add'					: True,
			'do_add_first'				: False,
			'do_delete'					: True,
			'do_delete_first'			: False,
			'do_exchange'				: True,
			'max_removal'				: float('inf'),
			'max_addition'				: kwargs.get('max_addition', 1),
			'initial_size'				: kwargs.get('initial_size', -1),
			'addable_block_size'		: kwargs.get('addable_block_size', 1),
			'deleteable_block_size'		: kwargs.get('deleteable_block_size', 1),
			'exchangeable_block_size'	: kwargs.get('exchangeable_block_size', 1),
		}

	elif algorithm_name == 'adxopt2016_forward':
		# 正向的2016年ADXOpt算法
		params = {
			'do_add'					: True,
			'do_add_first'				: True,
			'do_delete'					: True,
			'do_delete_first'			: False,
			'do_exchange'				: True,
			'max_removal'				: kwargs.get('max_removal', 1),
			'max_addition'				: float('inf'),
			'initial_size'				: kwargs.get('initial_size', 0),
			'addable_block_size'		: kwargs.get('addable_block_size', 1),
			'deleteable_block_size'		: kwargs.get('deleteable_block_size', 1),
			'exchangeable_block_size'	: kwargs.get('exchangeable_block_size', 1),
		}
		
	elif algorithm_name == 'adxopt2016_backward':
		# 反向的2016年ADXOpt算法
		params = {
			'do_add'					: True,
			'do_add_first'				: False,
			'do_delete'					: True,
			'do_delete_first'			: True,
			'do_exchange'				: True,
			'max_removal'				: float('inf'),
			'max_addition'				: kwargs.get('max_addition', 1),
			'initial_size'				: kwargs.get('initial_size', -1),
			'addable_block_size'		: kwargs.get('addable_block_size', 1),
			'deleteable_block_size'		: kwargs.get('deleteable_block_size', 1),
			'exchangeable_block_size'	: kwargs.get('exchangeable_block_size', 1),
		}
	
	else:
		# 可以继续开发新的算法配置以获得新算法
		raise NotImplementedError
		
	return params


