#!/usr/bin/env python
# encoding: utf-8
import sys
import os
import fnmatch
import glob
sys.path.insert(0, sys.path[0]+'/waf_tools')

VERSION = '1.0.0'
APPNAME = 'simple_nn'

srcdir = '.'
blddir = 'build'

from waflib.Build import BuildContext
from waflib import Logs
from waflib.Tools import waf_unit_test
import boost
import eigen


def options(opt):
    opt.load('compiler_cxx')
    opt.load('compiler_c')
    opt.load('boost')
    opt.load('eigen')

    opt.add_option('--tests', action='store_true', help='compile tests or not', dest='tests')


def configure(conf):
    conf.load('compiler_cxx')
    conf.load('compiler_c')
    conf.load('waf_unit_test')
    conf.load('boost')
    conf.load('eigen')

    conf.check_boost(lib='unit_test_framework')
    conf.check_eigen(required=True)

    if conf.env.CXX_NAME in ["icc", "icpc"]:
        common_flags = "-Wall -std=c++11"
        opt_flags = " -O3 -xHost -mtune=native -unroll -g"
    elif conf.env.CXX_NAME in ["clang"]:
        common_flags = "-Wall -std=c++11"
        opt_flags = " -O3 -march=native -g"
    else:
        if int(conf.env['CC_VERSION'][0]+conf.env['CC_VERSION'][1]) < 47:
            common_flags = "-Wall -std=c++0x"
        else:
            common_flags = "-Wall -std=c++11"
        opt_flags = " -O3 -march=native -g"

    all_flags = common_flags + opt_flags
    conf.env['CXXFLAGS'] = conf.env['CXXFLAGS'] + all_flags.split(' ')
    print(conf.env['CXXFLAGS'])

def summary(bld):
    lst = getattr(bld, 'utest_results', [])
    total = 0
    tfail = 0
    if lst:
        total = len(lst)
        tfail = len([x for x in lst if x[1]])
    waf_unit_test.summary(bld)
    if tfail > 0:
        bld.fatal("Build failed, because some tests failed!")

def build(bld):
    if bld.options.tests:
        bld.recurse('src/tests')

    libs = 'BOOST EIGEN'

    bld.program(features = 'cxx',
                  install_path = None,
                  source = 'src/examples/regression.cpp',
                  includes = './src',
                  uselib = libs,
                  target = 'regression')

    bld.add_post_fun(summary)

    bld.install_files('${PREFIX}/include/simple_nn', 'src/simple_nn/neural_net.hpp')
    bld.install_files('${PREFIX}/include/simple_nn', 'src/simple_nn/layer.hpp')