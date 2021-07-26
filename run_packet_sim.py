from new_packet_sweep_navigation_wrapper import *
from new_packet_sweep_navigation_wrapper import navigator as nav
import sys
 
 
flies = nav(model = Careful_Combo, environment = packets, num_steps = 12000, num_flies = 2000, job=sys.argv[1], delta_t=0.01, delta_pos = 0.154, init_theta=(0,360),                      
           wall_conditions = 'walk',                                                                                                                                                       
           model_kwargs = {'new_on':True, 'alpha':0*0.242, 'beta':1 * 3.872, 'whiff_threshold':0.04, 'signal_mean':1, 'signal_std':0, 'noise':False, 'w_sigma':0, 'on_sigma':0},         
           environment_kwargs={'dw_speed':300, 'cw_type':'Gaussian', 'eddy_D':1000, 'r0':5, 'rate':5, 'packet_D':10, 'source_x':0, 'max_x': 1649.75, 'delay_steps':int(400/300*100)},                      
                   start_x_pos=(50,400), start_y_pos = (-20,200), source_x_px = 0, source_y_px = 584, success_rad = 15)
 
flies.go()
flies.save_info()
