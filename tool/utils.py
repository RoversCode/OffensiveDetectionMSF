#-*- coding:utf-8 -*-
#Description TODO
#author Rover  Email:1059885524@qq.com
#version 1.0
#date 2021/5/11 13:49
import os, errno

#---------------------------------------------------
#创建输出文件夹
def mkdir_p(path):
	if path == '':
		return
	try:
		os.makedirs(path)
	except OSError as exc: # Python >2.5
		if exc.errno == errno.EEXIST and os.path.isdir(path):
			pass
		else: raise
