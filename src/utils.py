# -*- coding: utf-8 -*-
# @author: caoyang
# @email: caoyang@163.sufe.edu.cn
# ���ߺ���
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

# ��ʼ����־����
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

# �������ò���
def load_args(Config):
	config = Config()
	parser = config.parser
	try:
		return parser.parse_args()
	except:
		return parser.parse_known_args()[0]

# �������ò���
def save_args(args, save_path):
	
	class _MyEncoder(json.JSONEncoder):
		# �Զ����������͵����л�
		def default(self, obj):
			if isinstance(obj, type) or isinstance(obj, types.FunctionType):
				return str(obj)
			return json.JSONEncoder.default(self, obj)

	with open(save_path, 'w') as f:
		f.write(json.dumps(vars(args), cls=_MyEncoder, indent=4))


# �������
# :param array			: �������б�����飬��ֻ��Ե�һ��ά�Ƚ���������飻
# :parma n_split		: ��������
# :param do_shuffle		: �Ƿ����˳��
# :param do_balance		: ÿ������Ԫ�������Ƿ������⣻
# :return split_arrays	: �б���б�ÿ�����б���һ����
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

# ����Ӽ�������
# :param universal_set	: ȫ����
# :param min_size		: ����Ӽ�����С�ߴ磬Ĭ�Ϻ��Կռ���
# :param max_size		: ����Ӽ������ߴ磬Ĭ��ֵNone��ʾ��������Ӽ���Ҳ���������Ӽ���С����ö��һЩ�����
# :yield subset			: tuple���͵��Ӽ�
def generate_subset(universal_set, min_size=1, max_size=None):	
	if max_size is None:
		max_size = len(universal_set)
	for size in range(min_size, max_size + 1):
		for subset in combinations(universal_set, size):
			yield subset
