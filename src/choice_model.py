# -*- coding: utf-8 -*-
# @author: caoyang
# @email: caoyang@163.sufe.edu.cn
# 选择模型

if __name__ == '__main__':
	import sys
	sys.path.append('../')

import numpy
from copy import deepcopy
from abc import abstractmethod

from setting import *

class BaseChoiceModel:
	"""选择模型基类"""
	def __init__(self, 
				 product_prices: numpy.ndarray, 
				 product_valuations: numpy.ndarray, 
				 no_purchase_valuation,
				 offerset_capacity: int=None, 
				 *args, **kwargs):
		"""
		:param product_prices		: 产品价格数组，形状为(n, )，目前不考虑价格歧视模型，因此只可能是一阶数组；
		:param product_valuations	: 产品估值数组，形状为(-1, n)，通常为一阶数组，在混合逻辑模型中每个客户类都会有一套产品估值数组，因此可能是二阶数组；
		:param no_purchase_valuation: 不购买的估值，形状为(-1, 1)，通常为标量，在混合逻辑模型中每个用户类都会有一个不购买的估值，因此可能是一阶数组；
		:param offerset_capacity	: 报价集容量，默认值None表示无容量限制，即等于产品总数；
		"""
		
		# 初始化构造参数
		self.product_prices = deepcopy(product_prices)
		self.product_valuations = deepcopy(product_valuations)
		self.no_purchase_valuation = deepcopy(no_purchase_valuation)
		
		assert product_prices.shape[0] == product_valuations.shape[-1]							
		self.num_product = product_prices.shape[0]													
		self.product_ids = list(range(self.num_product))											
		self.offerset_capacity = self.num_product if offerset_capacity is None else offerset_capacity
	
	def validate_offerset(self, offerset):
		"""验证报价集的合法性"""
		assert len(offerset) <= self.offerset_capacity
		assert set(offerset).issubset(set(self.product_ids))
	
	def calc_revenue_expectation(self, offerset):
		"""计算收益期望"""
		if len(offerset) == 0:
			return 0.
		self.validate_offerset(offerset)
		product_choice_probabilitys = self.calc_product_choice_probabilitys(offerset)
		revenue_expectation = numpy.sum(product_choice_probabilitys * self.product_prices[offerset])
		return revenue_expectation

	@abstractmethod
	def calc_product_choice_probabilitys(self, *args, **kwargs):
		"""计算所有产品选择概率"""
		raise NotImplementedError


class MultiNomialLogit(BaseChoiceModel):
	"""多项逻辑模型"""
	def __init__(self, 
				 product_prices: numpy.ndarray, 
				 product_valuations: numpy.ndarray, 
				 no_purchase_valuation: float,
				 offerset_capacity: int):
		"""
		:param product_prices		: 产品价格数组，形状为(n, )；
		:param product_valuations	: 产品估值数组，形状为(n, )；
		:param no_purchase_valuation: 不购买的估值，标量；
		:param offerset_capacity	: 报价集容量，默认值None表示无容量限制，即等于产品总数；
		"""
		super(MultiNomialLogit, self).__init__(product_prices=product_prices, 
											   product_valuations=product_valuations,
											   no_purchase_valuation=no_purchase_valuation,
											   offerset_capacity=offerset_capacity)
		
	def calc_product_choice_probabilitys(self, offerset):
		# 计算分母总估值
		total_valuation = self.no_purchase_valuation + numpy.sum(self.product_valuations[offerset])
		
		# 计算每个产品的选择概率
		product_choice_probabilitys = self.product_valuations[offerset] / total_valuation
		return product_choice_probabilitys


class NestedLogit2(BaseChoiceModel):
	"""二级嵌套逻辑模型：必然存在一个只包含不购买选项的空嵌套"""
	def __init__(self, 
				 product_prices: numpy.ndarray, 
				 product_valuations: numpy.ndarray, 
				 no_purchase_valuation: float,
				 offerset_capacity: int,
				 nests: list,
				 nest_dis_similaritys: numpy.ndarray,
				 nest_no_purchase_valuations: numpy.ndarray):
		"""
		:param product_prices				: 产品价格数组，形状为(n, )；
		:param product_valuations			: 产品估值数组，形状为(n, )；
		:param no_purchase_valuation		: 不购买的估值，标量；
		:param offerset_capacity			: 报价集容量，默认值None表示无容量限制，即等于产品总数；
		:param nests						: 产品嵌套，长度为m，每个元素为一个随机产品子集列表；
		:param nest_dis_similaritys			: 嵌套相异度参数，形状为(m, )；
		:param nest_no_purchase_valuations	: 每个嵌套内的不购买选项估值，形状为(m, )；
		"""
		super(NestedLogit2, self).__init__(product_prices=product_prices, 
										   product_valuations=product_valuations,
										   no_purchase_valuation=no_purchase_valuation,
										   offerset_capacity=offerset_capacity)
		self.nests = deepcopy(nests)
		self.nest_dis_similaritys = deepcopy(nest_dis_similaritys)
		self.nest_no_purchase_valuations = nest_no_purchase_valuations

	def classify_offerset_by_nests(self, offerset):
		"""根据嵌套情况对给定的报价集进行划分，即将S划分为{S1, S2, S3,..., Sm}"""
		offerset_nests = []
		for nest in self.nests:
			offerset_nest = []
			for product_id in offerset:
				if product_id in nest:
					offerset_nest.append(product_id)
			offerset_nests.append(offerset_nest)
		return offerset_nests

	def calc_product_choice_probabilitys(self, offerset):
		# 生成报价集在给定嵌套下划分得到的报价子集
		offerset_nests = self.classify_offerset_by_nests(offerset)
		
		# 计算每个嵌套的效用值V_i(S)
		nest_valuations = []
		for nest_id, offerset_nest in enumerate(offerset_nests):
			nest_valuation = self.nest_no_purchase_valuations[nest_id] + numpy.sum(self.product_valuations[offerset_nest])
			nest_valuations.append(numpy.power(nest_valuation, self.nest_dis_similaritys[nest_id]))
		
		# 计算每个嵌套的选择概率Q_i(S)
		nest_choice_probabilitys = []
		total_valuation = self.no_purchase_valuation + sum(nest_valuations)
		for nest_id, offerset_nest in enumerate(offerset_nests):
			nest_choice_probability = nest_valuations[nest_id] / total_valuation
			nest_choice_probabilitys.append(nest_choice_probability)
		
		# 计算每个报价子集的总效用值V(S_i)
		offerset_nest_total_valuations = []
		for offerset_nest, nest_no_purchase_valuation in zip(offerset_nests, self.nest_no_purchase_valuations):
			offerset_nest_total_valuation = nest_no_purchase_valuation + numpy.sum(self.product_valuations[offerset_nest])
			offerset_nest_total_valuations.append(offerset_nest_total_valuation)

		# 计算每个产品的选择概率Pr(j|S)
		product_choice_probabilitys = []
		for target_product_id in offerset:
			product_choice_probability = 0.
			for offerset_nest, offerset_nest_total_valuation, nest_choice_probability in zip(offerset_nests, offerset_nest_total_valuations, nest_choice_probabilitys):
				if target_product_id in offerset_nest:
					product_choice_probability += (nest_choice_probability * self.product_valuations[target_product_id] / offerset_nest_total_valuation)
			product_choice_probabilitys.append(product_choice_probability)
		return numpy.array(product_choice_probabilitys, dtype=numpy.float64)


class MixedLogit(BaseChoiceModel):
	"""混合逻辑模型"""
	def __init__(self, 
				 product_prices: numpy.ndarray, 
				 product_valuations: numpy.ndarray, 
				 no_purchase_valuation: numpy.ndarray,
				 offerset_capacity: int,
				 class_weight: numpy.ndarray):
		"""
		:param product_prices			: 产品价格数组，形状为(n, )；
		:param product_valuations		: 产品估值数组，形状为(k, n)；
		:param no_purchase_valuation	: 不购买的估值，形状为(k, )；
		:param offerset_capacity		: 报价集容量，默认值None表示无容量限制，即等于产品总数；
		:param class_weight				: 客户类别的权重，形状为(k, )；
		"""
		super(MixedLogit, self).__init__(product_prices=product_prices, 
										 product_valuations=product_valuations,
										 no_purchase_valuation=no_purchase_valuation,
										 offerset_capacity=offerset_capacity)
		self.class_weight = deepcopy(class_weight)

	def calc_product_choice_probabilitys(self, offerset):
		# 计算每个客户类的分母总估值，形状为(k, )
		total_valuation = self.no_purchase_valuation + numpy.sum(self.product_valuations[:, offerset], axis=-1)
		
		# 计算每个产品的选择概率
		product_choice_probabilitys = numpy.dot(self.class_weight, self.product_valuations[:, offerset] / numpy.vstack([total_valuation] * len(offerset)).T)
		return product_choice_probabilitys

