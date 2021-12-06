# -*- coding: utf-8 -*-
# @author: caoyang
# @email: caoyang@163.sufe.edu.cn
# 算法设计

if __name__ == '__main__':
	import sys
	sys.path.append('../')

import numpy

from copy import deepcopy

from setting import *
from src.utils import generate_subset
from src.choice_model import BaseChoiceModel

class BaseAlgorithm:
	"""算法基类"""
	def __init__(self, algorithm_args, *args, **kwargs):
		self.algorithm_args = deepcopy(algorithm_args)
		# 根据传入的构造参数修改算法配置
		for key, value in kwargs.items():
			self.algorithm_args.__setattr__(key, value)

	@classmethod
	def find_initial_offerset(cls, model, initial_size):
		"""
		我原本2011版本中的GreedyOpt算法是穷举子集尺寸不超过inital_size的最优报价集作为算法起点；
		事实上这是对initial_size的错误理解，应该是穷举所有大小的为initial_size的报价子集作为算法起点；
		前者是牺牲精度以减少算法复杂度，这可能也是一个对节约时间有意义的方向；
		后者是牺牲时间以提升算法精度，不过设置initial_size值为零就没有任何影响；
		这里我还是对这种错误的想法进行了实现。
		:param model				: 选择模型，类型为src.choice_model.BaseModel的子类；
		:param initial_size			: 初始报价集的大小；
		:return initial_offerset	: 初始报价集；
		"""
		max_revenue, optimal_solutions = BaseAlgorithm.bruteforce(model=model, min_size=1, max_size=initial_size)
		assert len(optimal_solutions) == 1
		initial_offerset = optimal_solutions[0]
		return initial_offerset

	@classmethod
	def bruteforce(cls, model, min_size=1, max_size=None):
		"""
		暴力穷举法
		:param model	: 选择模型，类型为src.choice_model.BaseModel的子类；
		:param min_size	: 穷举子集的最小尺寸，默认忽略空集；
		:param max_size	: 穷举子集的最大尺寸，默认值None表示穷举所有子集，也可以限制子集大小以少枚举一些情况；
		
		:return max_revenue			: 最大收益；
		:return optimal_solutions	: 所有最优解；
		"""
		
		if max_size is None:
			max_size = model.offerset_capacity
		else:
			# 理论上max_size不能超过报价集容量，但是可能会有一些特殊的用法，总之断言一般都能成立
			assert max_size <= model.offerset_capacity
		max_revenue = 0.
		optimal_solutions = []
		for offerset in generate_subset(universal_set=deepcopy(model.product_ids),
										min_size=min_size,
										max_size=max_size):
			# 遍历所有产品子集并更新最大收益与最优解集
			offerset = list(offerset)
			revenue = model.calc_revenue_expectation(offerset=offerset)
			if revenue > max_revenue:
				optimal_solutions = [offerset]
				max_revenue = revenue
			elif revenue == max_revenue:
				optimal_solutions.append(offerset)
			else:
				continue
		return max_revenue, optimal_solutions
	
	@classmethod
	def greedy_add(cls, 
				   model: BaseChoiceModel, 
				   current_offerset: list, 
				   current_revenue: float, 
				   addable_product_ids: list, 
				   max_addable_block_size: int):
		"""
		穷举搜索最优的贪心增加产品块
		:param model					: 选择模型，类型为src.choice_model.BaseChoiceModel的子类；
		:param current_offerset			: 当前报价集；
		:param current_revenue			: 当前收益；
		:param addable_product_ids		: 可增加的产品子集；
		:param max_addable_block_size	: 最多可增加产品块的大小；
		
		:return added_product_block		: 搜索得到的最优的贪心增加产品块，可能为None则表示没有能够使得收益提升的增加操作；
		:return max_improvement			: 贪心增加能够达到的最高收益提升，若added_product_block为None，则对应的max_improvement为0；
		"""
		# 初始化最优的贪心增加产品块
		added_product_block = None
		max_improvement = 0.
		
		# 遍历产品找到能使得收益提升最多的一个产品块
		for addable_product_block in generate_subset(universal_set=addable_product_ids, 
													 min_size=1,
													 max_size=max_addable_block_size):
			addable_product_block = list(addable_product_block)
			updated_offerset = current_offerset + addable_product_block
			updated_revenue = model.calc_revenue_expectation(offerset=updated_offerset)
			revenue_improvement = updated_revenue - current_revenue
			if revenue_improvement > max_improvement:
				added_product_block = addable_product_block
				max_improvement = revenue_improvement
				
		return added_product_block, max_improvement
	
	@classmethod
	def greedy_delete(cls, 
					  model: BaseChoiceModel,  
					  current_offerset: list, 
					  current_revenue: float, 
					  deleteable_product_ids: list, 
					  max_deleteable_block_size: int):
		"""
		穷举搜索最优的贪心删除产品块
		:param model					: 选择模型，类型为src.choice_model.BaseChoiceModel的子类；
		:param current_offerset			: 当前报价集；
		:param current_revenue			: 当前收益；
		:param deleteable_product_ids	: 可删除的产品子集；
		:param max_deleteable_block_size: 最多可删除产品块的大小；
		
		:return deleted_product_block	: 搜索得到的最优的贪心删除产品块，可能为None则表示没有能够使得收益提升的删除操作；
		:return max_improvement			: 贪心删除能够达到的最高收益提升，若added_product_block为None，则对应的max_improvement为0；
		"""		
		# 初始化最优的贪心删除产品块
		deleted_product_block = None
		max_improvement = 0.
		
		# 遍历产品找到能使得收益提升最多的一个产品块
		for deleteable_product_block in generate_subset(universal_set=deleteable_product_ids, 
														min_size=1,
														max_size=max_deleteable_block_size):
			deleteable_product_block = list(deleteable_product_block)
			updated_offerset = list(set(current_offerset) - set(deleteable_product_block))
			updated_revenue = model.calc_revenue_expectation(offerset=updated_offerset)
			revenue_improvement = updated_revenue - current_revenue
			if revenue_improvement > max_improvement:
				deleted_product_block = deleteable_product_block
				max_improvement = revenue_improvement
				
		return deleted_product_block, max_improvement
		
	@classmethod
	def greedy_exchange(cls, 
						model: BaseChoiceModel, 
						current_offerset: list, 
						current_revenue: float, 
						addable_product_ids: list, 
						deleteable_product_ids: list, 
						max_exchangeable_block_size: int):
		"""
		穷举搜索最优的贪心交换产品块
		:param model						: 选择模型，类型为src.choice_model.BaseChoiceModel的子类；
		:param current_offerset				: 当前报价集；
		:param current_revenue				: 当前收益；
		:param addable_product_ids			: 可增加的产品子集；
		:param deleteable_product_ids		: 可删除的产品子集；
		:param max_exchangeable_block_size	: 最多可交换产品块的大小；
		
		:return added_product_block			: [×]搜索得到的最优的用于交换进去的贪心交换产品块，可能为None则表示没有能够使得收益提升的交换操作；
		:return deleted_product_block		: [×]搜索得到的最优的用于交换出来的贪心交换产品块，可能为None则表示没有能够使得收益提升的交换操作；
		:return exchanged_product_block		: (added_product_block, deleted_product_block)；
		:return max_improvement				: 贪心交换能够达到的最高收益提升，若added_product_block为None，则对应的max_improvement为0；
		"""		
		# 初始化最优的贪心交换产品块
		added_product_block = None
		deleted_product_block = None
		max_improvement = 0.
		
		# 遍历产品找到能使得收益提升最多的一个产品块
		for addable_product_block in generate_subset(universal_set=addable_product_ids, 
													 min_size=1,
													 max_size=max_exchangeable_block_size):
			addable_product_block = list(addable_product_block)
			for deleteable_product_block in generate_subset(universal_set=deleteable_product_ids, 
															min_size=1,
															max_size=max_exchangeable_block_size):
				deleteable_product_block = list(deleteable_product_block)
				updated_offerset = list(set(current_offerset) - set(deleteable_product_block)) + addable_product_block
				updated_revenue = model.calc_revenue_expectation(offerset=updated_offerset)
				revenue_improvement = updated_revenue - current_revenue
				if revenue_improvement > max_improvement:
					added_product_block = addable_product_block
					deleted_product_block = deleteable_product_block
					max_improvement = revenue_improvement
		exchanged_product_block = (added_product_block, deleted_product_block)
		return exchanged_product_block, max_improvement		

	def run(self, model):
		"""算法逻辑编写：后来我发现所有版本的贪心算法都可以在同一框架下实现"""
		# 提取参数
		initial_size = self.algorithm_args.initial_size
		addable_block_size = self.algorithm_args.addable_block_size
		do_add = self.algorithm_args.do_add
		do_add_first = self.algorithm_args.do_add_first
		do_delete = self.algorithm_args.do_delete
		do_delete_first = self.algorithm_args.do_delete_first
		do_exchange = self.algorithm_args.do_exchange
		max_addition = self.algorithm_args.max_addition
		max_removal = self.algorithm_args.max_removal
		initial_size = self.algorithm_args.initial_size
		addable_block_size = self.algorithm_args.addable_block_size
		deleteable_block_size = self.algorithm_args.deleteable_block_size
		exchangeable_block_size = self.algorithm_args.exchangeable_block_size
		
		assert do_add >= do_add_first, 'If do add first, you must allow add operation.'
		assert do_delete >= do_delete_first, 'If do delete first, you must allow delete operation.'
		assert do_add_first * do_delete_first == 0, 'You cannot do add first and do delete first at the same time.'
		
		product_ids = model.product_ids[:]
		num_product = model.num_product
		offerset_capacity = model.offerset_capacity
		
		# 若initial_size为负，则认为是从反向搜索，如initial_size为-1表示从产品全集开始搜索
		if initial_size < 0:
			initial_size = offerset_capacity + initial_size + 1
		
		# 一些节约时间的标记
		limit_addition = not max_addition == float('inf')
		limit_removal = not max_removal == float('inf')
		
		# 用于统计算法运行状况的全局变量
		global_optimal_offerset = None	# 全局的最优报价集
		global_max_revenue = 0.			# 全局的最有报价集对应的收益
		global_initial_offerset = None	# 全局的最优报价集从哪一个初始报价集迭代得到的

		# 算法逻辑开始
		for initial_offerset in generate_subset(universal_set=product_ids,
												min_size=initial_size,
												max_size=initial_size):
			# 每个initial_offerset对应的局部最优报价集与局部最大收益
			local_optimal_offerset = list(initial_offerset)
			local_max_revenue = model.calc_revenue_expectation(offerset=local_optimal_offerset)
			
			# 统计局部的增加与删除次数，这里使用numpy数组存储是为了简化索引代码逻辑
			addition_count = numpy.zeros((num_product, ))
			removal_count = numpy.zeros((num_product, ))
			
			# 算法迭代逻辑
			while True:
				# 下面这些变量如果最终在一次迭代结束仍为None，就说明对应的操作是不可行的；
				# 可能的原因包括：无法带来收益提升、对应操作不被允许、对应的操作受报价集容量限制而无法执行等；
				# 注意exchanged_product_block是由增加和删除两个产品块构成的，其他两个都只有一个产品块；
				added_product_block = None	
				deleted_product_block = None
				exchanged_product_block = (None, None)
				max_revenue_add = max_revenue_delete = max_revenue_exchange = local_max_revenue
				
				# 增加操作逻辑
				if do_add:
					# 初始化用于记录的变量
					optimal_offerset_add = local_optimal_offerset[:]
					
					# 检验报价集可用的容量是否为正
					optimal_offerset_length = len(optimal_offerset_add)
					available_capacity = offerset_capacity - optimal_offerset_length
					if available_capacity > 0:	
						max_addable_block_size = addable_block_size if addable_block_size <= available_capacity else available_capacity	# 确定可用于增加的报价集容量
						
						# 确定可用于增加的产品子集
						addable_product_ids = set(product_ids) - set(optimal_offerset_add)													
						if limit_addition:
							# 将不在候选报价集中的产品构成的产品子集与增加次数还没有超过限制的产品构成的产品的子集取交集
							addable_product_ids = addable_product_ids.intersection(set(numpy.where(addition_count < max_addition)[0].tolist()))
						addable_product_ids = list(addable_product_ids)
						
						# 遍历产品找到能使得收益提升最多的一个产品块
						added_product_block, max_improvement = BaseAlgorithm.greedy_add(model=model,
																						current_offerset=optimal_offerset_add[:],
																						current_revenue=max_revenue_add,
																						addable_product_ids=addable_product_ids,
																						max_addable_block_size=max_addable_block_size)	
						# 更新最优报价集与最大收益														
						if added_product_block is not None:
							optimal_offerset_add.extend(added_product_block)
							max_revenue_add += max_improvement

				# 删除操作逻辑
				if do_delete:
					# 初始化删除操作的最优解与最大收益
					optimal_offerset_delete = local_optimal_offerset[:]
					
					# 检验候选报价集是否为空
					optimal_offerset_length = len(optimal_offerset_delete)
					if optimal_offerset_length > 0:	
						max_deleteable_block_size = deleteable_block_size if deleteable_block_size <= optimal_offerset_length else optimal_offerset_length	# 确定可用于删除的产品子集大小
						
						# 确定可用于删除的产品子集													
						if limit_removal:
							# 将在候选报价集中的产品构成的产品子集与删除次数还没有超过限制的产品构成的产品的子集取交集
							deleteable_product_ids = list(set(optimal_offerset_delete).intersection(set(numpy.where(removal_count < max_removal)[0].tolist())))
						else:
							# 否则所有在候选报价集中的产品都可以用于删除
							deleteable_product_ids = optimal_offerset_delete[:]
						
						# 遍历产品找到能使得收益提升最多的一个产品块
						deleted_product_block, max_improvement = BaseAlgorithm.greedy_delete(model=model,
																							 current_offerset=optimal_offerset_delete[:],
																							 current_revenue=max_revenue_delete,
																							 deleteable_product_ids=deleteable_product_ids,
																							 max_deleteable_block_size=max_deleteable_block_size)	
						# 更新最优报价集与最大收益														
						if deleted_product_block is not None:
							optimal_offerset_delete = list(set(optimal_offerset_delete) - set(deleted_product_block))
							max_revenue_delete += max_improvement
				
				# 交换操作逻辑
				if do_exchange:
					# 初始化交换操作的最优解与最大收益
					optimal_offerset_exchange = local_optimal_offerset[:]
					
					# 检验报价集可用的容量是否为正且候选集是否为空
					optimal_offerset_length = len(optimal_offerset_exchange)
					available_capacity = offerset_capacity - optimal_offerset_length
					if available_capacity > 0 and optimal_offerset_length > 0:	
						max_addable_block_size = exchangeable_block_size if exchangeable_block_size <= available_capacity else available_capacity				# 确定可用于增加的报价集容量
						max_deleteable_block_size = exchangeable_block_size if exchangeable_block_size <= optimal_offerset_length else optimal_offerset_length	# 确定可用于删除的报价集容量
						# 最多可以交换的块大小取增加块与删除块中的较小值
						max_exchangeable_block_size = min(max_addable_block_size, max_deleteable_block_size)
						
						# 确定可用于交换进来的产品子集
						addable_product_ids = set(product_ids) - set(optimal_offerset_exchange)													
						if limit_addition:
							# 将不在候选报价集中的产品构成的产品子集与增加次数还没有超过限制的产品构成的产品的子集取交集
							addable_product_ids = addable_product_ids.intersection(set(numpy.where(addition_count < max_addition)[0].tolist()))
						addable_product_ids = list(addable_product_ids)

						# 确定可用于交换出去的产品子集													
						if limit_removal:
							# 将在候选报价集中的产品构成的产品子集与删除次数还没有超过限制的产品构成的产品的子集取交集
							deleteable_product_ids = list(set(optimal_offerset_exchange).intersection(set(numpy.where(removal_count < max_removal)[0].tolist())))
						else:
							# 否则所有在候选报价集中的产品都可以用于删除
							deleteable_product_ids = optimal_offerset_exchange[:]
						
						# 遍历产品找到能使得收益提升最多的一个产品块
						exchanged_product_block, max_improvement = BaseAlgorithm.greedy_exchange(model=model,
																								 current_offerset=optimal_offerset_exchange[:],
																								 current_revenue=max_revenue_exchange,
																								 addable_product_ids=addable_product_ids,
																								 deleteable_product_ids=deleteable_product_ids,
																								 max_exchangeable_block_size=max_exchangeable_block_size)	
						# 更新最优报价集与最大收益														
						if exchanged_product_block[0] is not None:
							optimal_offerset_exchange = list(set(optimal_offerset_exchange) - set(exchanged_product_block[1]))
							optimal_offerset_exchange.extend(exchanged_product_block[0])
							max_revenue_exchange += max_improvement
					
				
				# 三种操作都不可行，算法终止
				if added_product_block is None and deleted_product_block is None and exchanged_product_block[0] is None:
					assert exchanged_product_block[1] is None
					break
				
				# 若优先执行增加操作（从产品空集正向迭代可能会出现这种情况），并且增加操作能够带来更大的收益
				if do_add_first and added_product_block is not None:
					assert max_revenue_add > local_max_revenue				
					# 更新局部变量
					local_optimal_offerset = optimal_offerset_add[:]
					local_max_revenue = max_revenue_add
					for product_id in added_product_block:
						addition_count[product_id] += 1
						
				
				# 若优先执行删除操作（从产品全集反向迭代可能会出现这种情况），并且删除操作能够带来更大的收益
				elif do_delete_first and deleted_product_block is not None:
					assert max_revenue_delete > local_max_revenue
					# 更新局部变量
					local_optimal_offerset = optimal_offerset_delete[:]
					local_max_revenue = max_revenue_delete
					for product_id in deleted_product_block:
						removal_count[product_id] += 1
				
				# 否则就比较三种操作带来的收益
				else:
					max_revenue_of_adx = max([max_revenue_add, max_revenue_delete, max_revenue_exchange])
					if max_revenue_of_adx == max_revenue_add and added_product_block is not None:
						# 执行增加操作的局部变量更新
						local_optimal_offerset = optimal_offerset_add[:]
						local_max_revenue = max_revenue_add
						for product_id in added_product_block:
							addition_count[product_id] += 1	
					elif max_revenue_of_adx == max_revenue_delete and deleted_product_block is not None:
						# 执行删除操作的局部变量更新
						local_optimal_offerset = optimal_offerset_delete[:]
						local_max_revenue = max_revenue_delete
						for product_id in deleted_product_block:
							removal_count[product_id] += 1		
					else:		
						# 执行交换操作的局部变量更新	
						local_optimal_offerset = optimal_offerset_exchange[:]
						local_max_revenue = max_revenue_exchange
						for product_id in exchanged_product_block[0]:
							addition_count[product_id] += 1	
						for product_id in exchanged_product_block[1]:
							removal_count[product_id] += 1	
				
			# 更新全局变量
			if local_max_revenue > global_max_revenue:
				global_max_revenue = local_max_revenue
				global_optimal_offerset = local_optimal_offerset[:]
				global_initial_offerset = list(initial_offerset)		
			
		return global_max_revenue, global_optimal_offerset
	
# --------------------------------------------------------------------
# -*-*-*-*-*-*- 这-*-里-*-是-*-华-*-丽-*-的-*-分-*-割-*-线 -*-*-*-*-*-*-
# --------------------------------------------------------------------
# ！！！！！！！！！！！！！！！重要提示 ！！！！！！！ ！！！！！！！
# 以下几个子类都可以忽略了，因为BaseAlgorithm集成了统一的框架；
# 只需要修改算法配置即可实现不同的算法逻辑，这里保留代码仅供参考一些默认配置参数；
# 具体可见src.simulation_tools中的generate_params_for_algorithm函数；
# --------------------------------------------------------------------
# -*-*-*-*-*-*- 这-*-里-*-是-*-华-*-丽-*-的-*-分-*-割-*-线 -*-*-*-*-*-*-
# --------------------------------------------------------------------

class NaiveGreedy(BaseAlgorithm):
	"""平凡的贪心算法：只是增加操作"""
	def __init__(self, algorithm_args, *args, **kwargs):
		# 默认算法配置
		self.default_kwargs = {
			'do_add'					: True,
			'do_add_first'				: True,
			'do_delete'					: False,
			'do_delete_first'			: False,
			'do_exchange'				: False,
			'max_removal'				: 0.,
			'max_addition'				: float('inf'),
			'initial_size'				: 0,
			'addable_block_size'		: 1,
			'deleteable_block_size'		: 1,
			'exchangeable_block_size'	: 1,
		} if len(kwargs) == 0 else kwargs.copy()
		super(NaiveGreedy, self).__init__(algorithm_args=algorithm_args, *args, **self.default_kwargs)	


class GreedyOpt(BaseAlgorithm):
	"""2011年的GreedyOpt算法：考虑增加与交换两种操作，且优先考虑增加操作"""
	def __init__(self, algorithm_args, *args, **kwargs):
		# 默认算法配置
		self.default_kwargs = {
			'do_add'					: True,
			'do_add_first'				: True,
			'do_delete'					: False,
			'do_delete_first'			: False,
			'do_exchange'				: True,
			'max_removal'				: 1.,
			'max_addition'				: float('inf'),
			'initial_size'				: 0,
			'addable_block_size'		: 1,
			'deleteable_block_size'		: 1,
			'exchangeable_block_size'	: 1,
		} if len(kwargs) == 0 else kwargs.copy()
		super(GreedyOpt, self).__init__(algorithm_args=algorithm_args, *args, **self.default_kwargs)


class ADXOpt2014(BaseAlgorithm):
	"""2014年的ADXOpt算法：考虑增删换三种操作，且各种操作的优先级相同"""
	def __init__(self, algorithm_args, *args, **kwargs):
		# 默认算法配置
		self.default_kwargs = {
			'do_add'					: True,
			'do_add_first'				: False,
			'do_delete'					: True,
			'do_delete_first'			: False,
			'do_exchange'				: True,
			'max_removal'				: 1.,
			'max_addition'				: float('inf'),
			'initial_size'				: 0,
			'addable_block_size'		: 1,
			'deleteable_block_size'		: 1,
			'exchangeable_block_size'	: 1,
		} if len(kwargs) == 0 else kwargs.copy()
		super(ADXOpt2014, self).__init__(algorithm_args=algorithm_args, *args, **self.default_kwargs)


class ADXOpt2016(BaseAlgorithm):
	"""2014年的ADXOpt算法：考虑增删换三种操作，且优先考虑增加操作"""
	def __init__(self, algorithm_args, *args, **kwargs):
		# 默认算法配置
		self.default_kwargs = {
			'do_add'					: True,
			'do_add_first'				: True,
			'do_delete'					: True,
			'do_delete_first'			: False,
			'do_exchange'				: True,
			'max_removal'				: 1.,
			'max_addition'				: float('inf'),
			'initial_size'				: 0,
			'addable_block_size'		: 1,
			'deleteable_block_size'		: 1,
			'exchangeable_block_size'	: 1,
		} if len(kwargs) == 0 else kwargs.copy()
		super(ADXOpt2016, self).__init__(algorithm_args=algorithm_args, *args, **self.default_kwargs)
