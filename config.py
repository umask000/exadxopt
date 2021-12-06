# -*- coding: utf-8 -*-
# @author : caoyang
# @email: caoyang@163.sufe.edu.cn
# 存储可变参数的配置文件

import argparse

class BaseModelConfig:
	"""模型基本配置"""
	parser = argparse.ArgumentParser('--')
	parser.add_argument('--num_product', default=10, type=int, help='产品的数量')
	parser.add_argument('--offerset_capacity', default=None, type=int, help='报价集容量限制')
	parser.add_argument('--max_product_price', default=150., type=float, help='最大产品价格')
	parser.add_argument('--min_product_price', default=100., type=float, help='最大产品价格')
	parser.add_argument('--max_product_valuation', default=10., type=float, help='最大产品估值')
	parser.add_argument('--min_product_valuation', default=0., type=float, help='最小产品估值')
	

class MNLConfig(BaseModelConfig):
	"""多项逻辑模型的参数"""
	pass
	

class NL2Config(BaseModelConfig):
	"""嵌套逻辑模型的参数"""
	BaseModelConfig.parser.add_argument('--num_nest', default=1, type=int, help='嵌套数')
	BaseModelConfig.parser.add_argument('--max_dis_similarity', default=1., type=float, help='最大嵌套相异度')
	BaseModelConfig.parser.add_argument('--min_dis_similarity', default=0., type=float, help='最小嵌套相异度')
	BaseModelConfig.parser.add_argument('--exist_no_purchase_per_nest', default=False, type=bool, help='每个嵌套内是否都存在不购买选项，默认不存在')
	BaseModelConfig.parser.add_argument('--allow_nest_repetition', default=False, type=bool, help='是否允许同一个商品出现在不同嵌套内，默认不允许')


class MLConfig(BaseModelConfig):
	"""一般情况的混合逻辑模型的参数"""
	BaseModelConfig.parser.add_argument('--num_class', default=5, type=int, help='客户类别数')

# --------------------------------------------------------------------
# -*-*-*-*-*-*- 这-*-里-*-是-*-华-*-丽-*-的-*-分-*-割-*-线 -*-*-*-*-*-*-
# --------------------------------------------------------------------

class BaseAlgorithmConfig:
	"""算法基本配置"""
	parser = argparse.ArgumentParser('--')
	parser.add_argument('--do_add', default=True, type=bool, help='算法是否执行增加操作，后来我想了想是否可以将算法反向执行')
	parser.add_argument('--do_add_first', default=False, type=bool, help='算法是否优先执行增加操作，2011年的GreedyOpt优先执行增加操作，2014年的ADXOpt算法在某些情况下会优先执行增加操作，2016年的ADXOpt算法完全修正为优先执行增加操作')
	parser.add_argument('--do_delete', default=True, type=bool, help='算法是否执行删除操作，除了平凡的贪心算法外，所有算法都会考虑删除操作')
	parser.add_argument('--do_exchange', default=True, type=bool, help='算法是否执行交换操作，为了降低算法复杂度，可以不执行交换操作，这个思想从2014年的ADXOpt算法开始被提出')
	parser.add_argument('--max_removal', default=float('inf'), type=float, help='每个产品被移出的最多次数，该参数三版算法都有提及')
	parser.add_argument('--max_addition', default=float('inf'), type=float, help='每个产品被移入的最多次数，如果算法的起点是产品全集，就会有这个参数的引入')
	
	# 新加的几个参数用于改进算法
	parser.add_argument('--initial_size', default=0, type=int, help='2011年的GreedyOpt算法中提及，后来就不考虑了，指算法会遍历所有的大小为initial_size的子集，并分别以它们为起点进行迭代，最终取所有起点得到的最优解中的最优解')
	parser.add_argument('--addable_block_size', default=1, type=int, help='算法每次迭代可增加的产品块大小')
	parser.add_argument('--deleteable_block_size', default=1, type=int, help='算法每次迭代可删除的产品块大小')
	parser.add_argument('--exchangeable_block_size', default=1, type=int, help='算法每次迭代可交换的产品块大小')

class GreedyOptConfig(BaseAlgorithmConfig):
	"""GreedyOpt算法的基本配置"""
	pass

class ADXOpt2014Config(BaseAlgorithmConfig):
	"""2014年的ADXOpt算法的基本配置"""
	pass

class ADXOpt2016Config(BaseAlgorithmConfig):
	"""2016年的ADXOpt算法的基本配置"""
	pass

if __name__ == "__main__":
	import json
	from src.utils import load_args, save_args
	
	config = BaseAlgorithmConfig()
	parser = config.parser
	args = parser.parse_args()
	
	save_args(args, '1.json')
	
	# print(args.__getattribute__('num_product'))
	# args.__setattr__('num_product', 100)
	# print(args.num_product)

