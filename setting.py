# -*- coding: utf-8 -*-
# @author: caoyang
# @email: caoyang@163.sufe.edu.cn
# 存储全局变量的设置文件

# 文件夹路径
IMAGE_FOLDER = 'image'
LOGGING_FOLDER = 'logging'
TEMP_FOLDER = 'temp'


# 算法类映射：这个貌似没什么用了
ALGORITHM_CLASS_MAPPING = {
	'naivegreedy'	: {'class': 'NaiveGreedy'},
	'greedyopt'		: {'class': 'GreedyOpt'},
	'adxopt2014'	: {'class': 'ADXOpt2014'},
	'adxopt2016'	: {'class': 'ADXOpt2016'},
}

# 模型类与相关方法映射
MODEL_MAPPING = {
	'mnl'	: {'class': 'MultiNomialLogit',	'config': 'MNLConfig',	'param': 'generate_params_for_MNL'},
	'nl2'	: {'class': 'NestedLogit2',		'config': 'NL2Config',	'param': 'generate_params_for_NL2'},
	'ml'	: {'class': 'MixedLogit',		'config': 'MLConfig',	'param': 'generate_params_for_ML'},
}

# 目前已经开发的算法
ALGORITHM_NAMES = [
	# 'naivegreedy_forward',
	# 'naivegreedy_backward',
	'greedyopt_forward',
	'greedyopt_backward',
	'adxopt2014_forward',
	'adxopt2014_backward',
	'adxopt2016_forward',
	'adxopt2016_backward',
]

# 极小的数
EPSILON = 1e-6
