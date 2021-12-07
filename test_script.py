# -*- coding: utf-8 -*-
# @author: caoyang
# @email: caoyang@163.sufe.edu.cn
# 主程序

import os
import numpy
from pprint import pprint

from config import *
from setting import *

from src.algorithm import BaseAlgorithm, NaiveGreedy, GreedyOpt, ADXOpt2014, ADXOpt2016
from src.choice_model import MultiNomialLogit, NestedLogit2, MixedLogit
from src.plot_tools import plot_main_1_to_4, plot_main_adxopt2014_for_nl2_further_analysis, plot_main_adxopt2014_for_nl2_further_analysis_new
from src.simulation_tools import generate_params_for_ML, generate_params_for_MNL, generate_params_for_NL2
from src.utils import load_args
# ----------------------------------------------------------------------
# MNL测试
def test_MNL(fixed=True):
	args = load_args(MNLConfig)
	params = {
		'product_prices': numpy.array([100, 100]),
		'product_valuations': numpy.array([1, 2]),
		'no_purchase_valuation': 1.,
		'offerset_capacity': None,
	} if fixed else generate_params_for_MNL(args)
	pprint(params)
	model = MultiNomialLogit(**params)
	print(model.calc_product_choice_probabilitys([0, 1]))
	print(model.calc_revenue_expectation([0]))
	print(model.calc_revenue_expectation([1]))
	print(model.calc_revenue_expectation([0, 1]))
# ----------------------------------------------------------------------
# NL2测试
def test_NL2(fixed=True):
	args = load_args(NL2Config)
	params = {
		'product_prices': numpy.array([100, 100]),
		'product_valuations': numpy.array([1, 2]),
		'no_purchase_valuation': 1.,
		'offerset_capacity': None,
		'nests': [[0], [1]],
		'nest_dis_similaritys': [0.5, 1.],
		'nest_no_purchase_valuations': [0., 0.],	
	} if fixed else generate_params_for_NL2(args)
	pprint(params)
	model = NestedLogit2(**params)
	print(model.calc_product_choice_probabilitys([0]))
	print(model.calc_product_choice_probabilitys([1]))
	print(model.calc_product_choice_probabilitys([0, 1]))
	print(model.calc_revenue_expectation([0]))
	print(model.calc_revenue_expectation([1]))
	print(model.calc_revenue_expectation([0, 1]))
# ----------------------------------------------------------------------
# ML测试
def test_ML(fixed=True):
	args = load_args(MLConfig)
	params = generate_params_for_ML(args)
	params = {
		'product_prices'		: numpy.array([100, 100]),
		'product_valuations'	: numpy.array([[1, 2], [2, 1]]),	
		'no_purchase_valuation'	: numpy.array([0.5, 1]),
		'offerset_capacity'		: None,
		'class_weight'			: numpy.array([0.3, 0.7]),
	} if fixed else generate_params_for_ML(args)
	pprint(params)
	print(sum(params['class_weight']))
	model = MixedLogit(**params)
	print(model.calc_product_choice_probabilitys([0]))
	print(model.calc_product_choice_probabilitys([1]))
	print(model.calc_product_choice_probabilitys([0, 1]))
	print(model.calc_revenue_expectation([0]))
	print(model.calc_revenue_expectation([1]))
	print(model.calc_revenue_expectation([0, 1]))
# ----------------------------------------------------------------------
# naivegreedy测试
def test_naivegreedy(fixed_model=True, fixed_algorithm=True):
	model_args = load_args(MNLConfig)
	params = {
		'product_prices': numpy.array([2, 2, 2]),
		'product_valuations': numpy.array([numpy.exp(.5), numpy.exp(.5), numpy.exp(.7)]),
		'no_purchase_valuation': 1.,
		'offerset_capacity': 2,
	} if fixed_model else generate_params_for_MNL(model_args)
	model = MultiNomialLogit(**params)

	algorithm_args = {
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
	}

	algorithm_args = load_args(BaseAlgorithmConfig)
	# algorithm_args.initial_size = 1
	# algorithm_args.addable_block_size = 2
	naivegreedy = NaiveGreedy(algorithm_args)
	print(BaseAlgorithm.bruteforce(model, max_size=model.offerset_capacity))
	print(naivegreedy.run(model))
	
# ----------------------------------------------------------------------
# greedyopt测试
def test_greedyopt():
	model_args = load_args(MNLConfig)
	params = {
		'product_prices': numpy.array([2, 2, 1.9]),
		'product_valuations': numpy.array([numpy.exp(.5), numpy.exp(.5), numpy.exp(.7)]),
		'no_purchase_valuation': 1.,
		'offerset_capacity': 2,
	} if fixed else generate_params_for_MNL(model_args)
	model = MultiNomialLogit(**params)
	
	algorithm_args = load_args(BaseAlgorithmConfig)
	# algorithm_args.initial_size = 1
	# algorithm_args.addable_block_size = 2

	naivegreedy = NaiveGreedy(algorithm_args)
	print(BaseAlgorithm.bruteforce(model, max_size=model.offerset_capacity))
	print(naivegreedy.run(model))

# ----------------------------------------------------------------------
# adxopt2014测试
def test_adxopt2014():
	raise NotImplementedError
# ----------------------------------------------------------------------
# adxopt2016测试
def test_adxopt2016():
	raise NotImplementedError

if __name__ == '__main__':
	# test_MNL(False)
	# test_NL2(False)
	# test_ML(False)
	# test_naivegreedy(True, True)
	
	plot_main_1_to_4(do_export=False)
	plot_main_adxopt2014_for_nl2_further_analysis(do_export=False)
	plot_main_adxopt2014_for_nl2_further_analysis_new(do_export=True)
	
