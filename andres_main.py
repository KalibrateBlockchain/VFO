# ****************************************************************************
# Name       : main.py
# Author     : Andres Valdez
# Version    : 1.0
# Description: Main script to parse and analyze audio files to determine
#              if the subject-source is healthy or un-healthy
# Data	 : 15-02-2021
# ****************************************************************************

from __future__ import unicode_literals
import os, sys
from utils_odes import *

if __name__ == "__andres_main__":


    print('Usage: python [andres_main.py] [audio_file]')
    
    if(len(sys.argv) == 1):
        t,data,t_model,u0,x,u,y,v,title = foo_main()
        fig_name = 'None'
    else:
        audio = sys.argv[1]
        t,data,t_model,u0,x,u,y,v,title = foo_main(audio_file = audio)
        fig_name = audio[:-4]
    
    # Now we plot the solution
    plot_solution(t,data,t_model,u0,x,u,y,v,fig_name,title)
