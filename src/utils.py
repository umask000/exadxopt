# -*- coding: utf-8 -*-
# @author: caoyang
# @email: caoyang@163.sufe.edu.cn
# 工具函数
if __name__ == '__main__':
	import sys
	sys.path.append('../')

import json
import numpy
import logging
import argparse

from copy import deepcopy
from itertools import combinations

from setting import *

# 初始化日志配置
def initialize_logging(filename, filemode='w'):
	logging.basicConfig(
		level=logging.DEBUG,
		format='%(asctime)s | %(filename)s | %(levelname)s | %(message)s',
		datefmt='%Y-%m-%d %H:%M:%S',
		filename=filename,
		filemode=filemode,
	)
	console = logging.StreamHandler()
	console.setLevel(logging.INFO)
	formatter = logging.Formatter('%(asctime)s | %(filename)s | %(levelname)s | %(message)s')
	console.setFormatter(formatter)
	logging.getLogger().addHandler(console)

# 加载配置参数
def load_args(Config):
	config = Config()
	parser = config.parser
	try:
		return parser.parse_args()
	except:
		return parser.parse_known_args()[0]

# 保存配置参数
def save_args(args, save_path):
	
	class _MyEncoder(json.JSONEncoder):
		# 自定义特殊类型的序列化
		def default(self, obj):
			if isinstance(obj, type) or isinstance(obj, types.FunctionType):
				return str(obj)
			return json.JSONEncoder.default(self, obj)

	with open(save_path, 'w') as f:
		f.write(json.dumps(vars(args), cls=_MyEncoder, indent=4))


# 随机分组
# :param array			: 可以是列表和数组，但只会对第一个维度进行随机分组；
# :parma n_split		: 分组数；
# :param do_shuffle		: 是否打乱顺序；
# :param do_balance		: 每个组中元素数量是否尽量均衡；
# :return split_arrays	: 列表的列表，每个子列表是一个组
def random_split(array, n_splits, do_shuffle=True, do_balance=False):
	array_length = len(array)
	assert array_length > n_splits
	_array = deepcopy(array)
	if do_shuffle:
		numpy.random.shuffle(_array)
	index = list(range(array_length - 1))
	if do_balance:
		num_per_split = int(array_length / n_splits)
		split_points = [i * num_per_split - 1 for i in range(1, n_splits)]
	else:
		split_points = sorted(numpy.random.choice(a=index, size=n_splits - 1, replace=False))
	split_arrays = []
	current_point = 0
	for split_point in split_points:
		split_arrays.append(_array[current_point: split_point + 1])
		current_point = split_point + 1
	split_arrays.append(_array[current_point: ])
	return split_arrays

# 穷举子集生成器
# :param universal_set	: 全集；
# :param min_size		: 穷举子集的最小尺寸，默认忽略空集；
# :param max_size		: 穷举子集的最大尺寸，默认值None表示穷举所有子集，也可以限制子集大小以少枚举一些情况；
# :yield subset			: tuple类型的子集
def generate_subset(universal_set, min_size=1, max_size=None):	
	if max_size is None:
		max_size = len(universal_set)
	for size in range(min_size, max_size + 1):
		for subset in combinations(universal_set, size):
			yield subset
