from video_navigation_wrapper import *
from video_navigation_wrapper import navigator as nav
import sys

flies = nav(model = Careful_Combo, environment = complex, num_steps = 11090, num_flies = 100, job=sys.argv[1], delta_t=1/89.94, seed_mult = 900, delta_pos = 0.154, init_theta=(0,360), wall_conditions = "walk",
            model_kwargs={'new_on':True},
        environment_kwargs={
         'data_dir':"../../../../data",'file':"2018_09_12_NA_3_3ds_5do_IS_1-frames.mat", 'bck_file':"2018_09_12_NA_3_3ds_5do_IS_1.mat", 'full_arena':True, 'start_frame':600}, 
            start_x_pos=(50,300), start_y_pos = (20,160))


flies.go()

