import numpy as np
import os
import logging
import shutil

def create_folder_rm(fd):
    try:
        os.makedirs(fd)
    except FileExistsError:
        logging.info(f"folder {fd} already exists")
        logging.info("Removing the folder content and rewritting")
        shutil.rmtree(fd)
        os.makedirs(fd)

def create_folder(fd):
    if not os.path.exists(fd):
        # creates problems when multiple scripts creating folders
        try:
            os.makedirs(fd)
        except:
            print('Folder already exits')

def create_logging(log_dir, filemode):
    create_folder(log_dir)
    i1 = 0
    
    while os.path.isfile(os.path.join(log_dir, '%04d.log' % i1)):
        i1 += 1
        
    log_path = os.path.join(log_dir, '%04d.log' % i1)
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                        datefmt='%a, %d %b %Y %H:%M:%S',
                        filename=log_path,
                        filemode=filemode)
                
    # Print to console   
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    
    return logging