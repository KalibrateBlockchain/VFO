import numpy as np
import logging
import pwd
import grp
import os
import time
import shutil
from utils import *
import json
import argparse
import subprocess
from os import path

# Get Directory List 

for x in os.listdir('/var/www/html/process_samples/'):
	# Take first file
	filename='/var/www/html/process_samples/'+x
	if path.exists(filename):
		file=open(filename,'rt')
		contents = file.read()
		my_dict = json.loads(contents)
		print(my_dict['user_id'])
		user_id = my_dict['user_id']
		data_dir = my_dict['data_dir']
		test_id = my_dict['test_id']
		audio_file = my_dict['audio_file']
		user_id = my_dict['user_id']
		mode = my_dict['mode']
		cmdcall = '/home/cisco/miniconda3/bin/python healthdrop_audio_processor.py --data_dir '+data_dir
		cmdcall = cmdcall +' --user_id '+user_id
		cmdcall = cmdcall +' --test_id '+test_id
		cmdcall = cmdcall +' --audio_file '+audio_file
		cmdcall = cmdcall +' --mode 1'
		print(cmdcall)
		os.remove(filename)
		os.chdir('/home/cisco/VFO')
		subprocess.call(cmdcall,shell=True)

	# delete file

