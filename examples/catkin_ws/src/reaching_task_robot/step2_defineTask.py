# This file is part of DmpBbo, a set of libraries and programs for the 
# black-box optimization of dynamical movement primitives.
# Copyright (C) 2014 Freek Stulp, ENSTA-ParisTech
# 
# DmpBbo is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
# 
# DmpBbo is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
# 
# You should have received a copy of the GNU Lesser General Public License
# along with DmpBbo.  If not, see <http://www.gnu.org/licenses/>.


import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pickle
import inspect



lib_path = os.path.abspath('/home/ksavevska/stulp_dmpbbo/dmpbbo/python/')
sys.path.append(lib_path)

from TaskThrowBall import TaskThrowBall
from TaskReach import TaskReach
from dmp.Trajectory import Trajectory

if __name__=="__main__":
    
    output_task_file = None
    
    if (len(sys.argv)<2):
        print('Usage: '+sys.argv[0]+' <task file.p> <demonstrated trajectory file>')
        print('Example: python3 '+sys.argv[0]+' results/task.p trajectory.txt')
        sys.exit()
        
    if (len(sys.argv)>1):
        output_task_file = sys.argv[1]
        trajectory_file = sys.argv[2]

    ee_pos_goal = [0.85, -0.2, 1.0] 
    pos_margin = 0
    ref_cop = [0.0, 0.0]
    stability_weight = 0.3
    goal_weight = 0.7
    traj_weight = 0.0

    traj_demonstrated = Trajectory.readFromFile(trajectory_file)

    task = TaskReach(ee_pos_goal, pos_margin, ref_cop, stability_weight, goal_weight, traj_weight, traj_demonstrated)
    
    # Save the task instance itself
    print('  * Saving task to file "'+output_task_file+"'")
    pickle.dump(task, open(output_task_file, "wb" ))

    # Save the source code of the task for future reference
    #src_task = inspect.getsourcelines(task.__class__)
    #src_task = ' '.join(src_task[0])
    #src_task = src_task.replace("(Task)", "")    
    #filename = directory+'/the_task.py'
    #task_file = open(filename, "w")
    #task_file.write(src_task)
    #task_file.close()
