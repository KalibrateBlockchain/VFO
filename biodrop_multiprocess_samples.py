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
import glob
from os import path
from datetime import datetime


num_cpus = os.cpu_count()


process_directory = '/var/www/html/process_samples/'
job_filemask =process_directory+'*.job'

file_list = glob.glob(job_filemask)

procs_list = []
proc_filename_list = {}


nowork = True
if len(file_list) > 0:
	nowork = False

logfile=""

if nowork==False:
	logfile = open('/home/cisco/VFO/biodrop_multiprocess_samples.log','a')
	logfile.write('\n<START OF PROCESS>\n')

	date_time_stamp = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
	logfile.write('StartDateTime:'+date_time_stamp+'\n')

	logfile.write('CPUs:'+str(num_cpus)+'\n')

	if num_cpus > 1:
		num_cpus = num_cpus - 1 #reserve core (for server itself)

	logfile.write('CPUsAssignedToProcess:'+str(num_cpus)+'\n')
	logfile.write('ProcessDirectory:'+process_directory+'\n')
	logfile.write('JobFiles:'+str(len(file_list))+'\n')



for filename in file_list:
	if path.exists(filename):

		renamed_file = False
		try:
			os.rename(filename,filename+'.proc')
			renamed_file = True
		except Exception as err:
			logfile.write('*RENAME ERROR*:'+filename+'\n')

		logfile.write('PROCS_Count:'+str(len(procs_list))+'\n')
		if renamed_file == True:
			logfile.write('ProcessingJobFile:'+filename+'.proc'+'\n')
			# GET THE INFO AND CREATE THE CMD LINE CALL
			file=open(filename+'.proc','rt')
			contents = file.read()
			my_dict = json.loads(contents)

			user_id = my_dict['user_id']
			data_dir = my_dict['data_dir']
			test_id = my_dict['test_id']
			audio_file = my_dict['audio_file']
			user_id = my_dict['user_id']
			mode = my_dict['mode']

			cmdcall = '/home/cisco/miniconda3/bin/python3 /home/cisco/VFO/healthdrop_audio_processor.py --data_dir '+data_dir
			cmdcall = cmdcall +' --user_id '+user_id
			cmdcall = cmdcall +' --test_id '+test_id
			cmdcall = cmdcall +' --audio_file '+audio_file
			cmdcall = cmdcall +' --mode 1'

			logfile.write('JOB_UserID:'+my_dict['user_id']+'\n')
			logfile.write('JOB_CMD:'+cmdcall+'\n')

			date_time_stamp = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
			logfile.write('JOB_StartDateTime:'+date_time_stamp+'\n')
			a_proc=subprocess.Popen(cmdcall,shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
			procs_list.append(a_proc)
			proc_filename_list[str(a_proc.pid)]=filename+'.proc'
			logfile.write('JOB_PID:'+str(a_proc.pid)+'\n')

			if len(procs_list)>=num_cpus:
				logfile.write('Procs_Limit_Saturation_Detected:True\n')

			while len(procs_list)>=num_cpus:
				free_cpu= False
				for proc in procs_list:
					if proc.poll() is None:
						continue
					date_time_stamp = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
					logfile.write('JOB_EndDateTime:'+date_time_stamp+'\n')
					logfile.write('JOB_Release_PID'+str(proc.pid)+'\n')
					os.remove(proc_filename_list.get(str(proc.pid)))
					procs_list.remove(proc)
					proc_filename_list.remove(str(proc.pid))

					free_cpu=True
				if free_cpu==False:
					time.sleep(5.0)

if len(procs_list):
	logfile.write('Procs_To_Release:'+str(len(procs_list))+'\n')

while len(procs_list):
	time.sleep(5.0)
	for proc in procs_list:
		if proc.poll() is None:
			continue
		date_time_stamp = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
		logfile.write('Proc_Release_PID'+str(proc.pid)+'\n')
		logfile.write('Proc_EndDateTime:'+date_time_stamp+'\n')
		procs_list.remove(proc)
		os.remove(proc_filename_list.get(str(proc.pid))

if nowork==False:
	date_time_stamp = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
	logfile.write('EndDateTime:'+date_time_stamp+'\n')
	logfile.write('<END OF PROCESS>'+'\n')
	logfile.close()

