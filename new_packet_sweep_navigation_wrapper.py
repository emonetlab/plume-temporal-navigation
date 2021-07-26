from final_low_save_models import *
from new_final_environments import *
import numpy as np
import scipy as sp
import pickle as pkl

class navigator():

    def __init__(self, model, environment, num_steps, num_flies, delta_t, delta_pos, init_theta, start_x_pos, start_y_pos, job=0, seed_mult = 1, out_dir="./", 
           wall_conditions = 'turn', environment_kwargs={'arena_x_indices':(0,2048), 'arena_y_indices':(0,1200)}, model_kwargs = {}, start_from_file = False, 
          source_x_px = None, source_y_px = None, success_rad = 15): 

        #base simulation parameters

        self.num_steps = num_steps
        self.num_flies = num_flies
        self.delta_t = delta_t
        self.delta_pos = delta_pos
        self.wall_conditions = wall_conditions
        self.job = job
        self.start_from_file = start_from_file
        self.source_x_px = source_x_px
        self.source_y_px = source_y_px
        self.success_rad = success_rad
        #self.alpha_2 = alpha_2
        #self.beta_2 = beta_2
        
        #random numbers

        self.rand_gen = sp.random.RandomState(10000 * seed_mult * int(job) + int(num_flies))

        #environment-specific initialization

        self.environment_kwargs = environment_kwargs
        self.environment = environment(**self.environment_kwargs)
        self.environment.rand_gen = self.rand_gen
        self.environment.delta_t = self.delta_t 

        #self.update = self.model.update
       
        #data saving

        self.out_dir = out_dir


        #antenna intialization-measured in pixels (at 0 degrees ellipse is longer and narrower-1.5mm long and 0.5mm wide)

        self.antenna_width = 0.5 * 1/self.delta_pos
        self.antenna_height = 1.5 * 1/self.delta_pos
        self.antenna_dist = 1/(self.delta_pos)
        
        self.std_box = np.array([[0,0]])

        for i in range(0,int(self.antenna_height)+2):

            for j in range(0,int(self.antenna_height)+2):

                m = i - (int(self.antenna_height/2)+1) 
                n = j - (int(self.antenna_height/2)+1)

                self.std_box = np.append(self.std_box, [[m,n]], axis = 0)

        self.std_box = self.std_box[1:]


        #odor initialization

        #self.odor = sp.zeros((self.num_steps, self.num_flies))
        self.odor_L = sp.zeros(self.num_flies)
        self.odor_R = sp.zeros(self.num_flies)
        

        ########### INITIAL POSITION DATA #############                                                                                                                                                     
        self.x = sp.zeros(self.num_flies)

        if(start_from_file):
            x_file = open(str(start_dir) + 'init_xs','rb')
            self.x = sp.array(pkl.load(x_file))
            x_file.close()

        else:
            start_x = start_x_pos[0]
            start_x_range = start_x_pos[1] - start_x_pos[0]

            for i in range(self.num_flies):
                self.x[i] = start_x + start_x_range * self.rand_gen.random_sample()

        self.y = sp.zeros(self.num_flies)

        if(start_from_file):
            y_file = open(str(start_dir) + 'init_ys','rb')
            self.y = sp.array(pkl.load(y_file))
            y_file.close()

        else:
            start_y = start_y_pos[0]
            start_y_range = start_y_pos[1] - start_y_pos[0]

            for i in range(self.num_flies):
                self.y[i] = start_y + start_y_range * self.rand_gen.random_sample()

        self.theta = sp.zeros(self.num_flies)

        if(start_from_file):
            theta_file = open(str(start_dir) + 'init_thetas','rb')
            self.theta = sp.array(pkl.load(theta_file))
            theta_file.close()


        else:
            start_theta = init_theta[0]
            start_theta_range = init_theta[1] - init_theta[0]

            for i in range(self.num_flies):
                self.theta[i] = (start_theta + start_theta_range
                                 * self.rand_gen.random_sample())

        
        self.initial_x = self.x
        self.initial_y = self.y
        self.initial_thetas = self.theta 

        #model-specific initialization                                                                                                                                                                      
        self.model_kwargs = model_kwargs

        self.model = model(num_steps = self.num_steps, num_flies = self.num_flies, delta_t = self.delta_t, rand_gen = self.rand_gen, **self.model_kwargs)

    #### DEFINING FUNCTION THAT ACTUALLY SIMULATES FLY WALKS ####

    def go(self):

        self.success_list = np.zeros(self.num_flies)
        self.success_times = self.num_steps*self.delta_t + np.zeros(self.num_flies)

        ##get odor signal

        for i in range(self.num_steps-1):
            
            self.i = i

            self.x_idx = np.rint(self.x/self.delta_pos)
            self.y_idx = np.rint(self.y/self.delta_pos)

            if self.environment.type == 'frame':

                self.signal = self.environment.generate_env(time_step = self.i, delta_t = self.delta_t, rand_gen = self.rand_gen)

                frm_data = self.signal

                for j in range(0,self.num_flies):
                                                                                                                                                
                    left_box, right_box = full_antenna(self.std_box, self.theta[j], self.x_idx[j], self.y_idx[j], a=self.antenna_width/2, b=self.antenna_height/2, dist=self.antenna_dist)
                                                                                                                                                                                    
                    left_odors = np.zeros(len(left_box))
                    right_odors = np.zeros(len(right_box))

                    for i in range(0,len(left_box)):

                        if (0 < left_box[i][0] < np.shape(self.signal)[0]) and (0 < left_box[i][1] < np.shape(self.signal)[1]):

                            left_odors[i] = max(frm_data[int(left_box[i][0]), int(left_box[i][1])], 0)

                        if (0 < right_box[i][0] < np.shape(self.signal)[0]) and (0 < right_box[i][1] < np.shape(self.signal)[1]):
    
                            right_odors[i] = max(frm_data[int(right_box[i][0]), int(right_box[i][1])], 0)


                    self.odor_L[j] = np.mean(left_odors)
                    self.odor_R[j] = np.mean(right_odors)


            elif self.environment.type == 'packets':

                packet_pos, packet_sizes = self.environment.generate_packets(delta_t = self.delta_t, rand_gen = self.rand_gen)

                for j in range(0,self.num_flies):
                                                                                                                                                
                    left_box, right_box = full_antenna(self.std_box, self.theta[j], self.x_idx[j], self.y_idx[j], a=self.antenna_width/2, b=self.antenna_height/2, dist=self.antenna_dist)
                    
                    left_box = self.delta_pos * left_box

                    right_box = self.delta_pos * right_box                 

                    self.odor_L[j], self.odor_R[j] = self.environment.compute_sig(left_points = left_box, right_points = right_box, packet_pos = packet_pos, packet_sizes = packet_sizes, rand_gen = self.rand_gen)


            else:

                self.odor_L, self.odor_R = self.environment.generate_sig(rand_gen = self.rand_gen, num_flies = self.num_flies)




            #update position variables

            new_thetas, dx, dy = self.model.update(i = self.i, theta=self.theta, odor_L = self.odor_L, odor_R = self.odor_R, x = self.x, y = self.y)
            
            new_thetas = new_thetas%360

            self.theta = new_thetas

            self.x = self.x + dx
            self.y = self.y + dy


            self.success_list = self.success_list + (((self.x-self.source_x_px*self.delta_pos)**2+(self.y-self.source_y_px*self.delta_pos)**2) < (self.success_rad)**2)

            self.success_times[self.success_list==1] = self.i * self.delta_t
                

            #check wall conditions, correct position variables if necessary

            if self.wall_conditions == "turn":

                bad_x_1 = self.x > np.shape(self.signal)[0]*self.delta_pos - 3
                bad_x_2 = self.x < 0 + 3

                only_bad_x_1 = self.x[bad_x_1]
                only_bad_x_2 = self.x[bad_x_2]

                bad_y_1 = self.y < 0*self.delta_pos + 3
                bad_y_2 = self.y > np.shape(self.signal)[1]*self.delta_pos - 3

                only_bad_y_1 = self.y[bad_y_1]
                only_bad_y_2 = self.y[bad_y_2]

                self.x[bad_x_1+bad_x_2] = self.x[bad_x_1+bad_x_2]
                self.y[bad_y_1+bad_y_2] = self.y[bad_y_1+bad_y_2]

                self.theta[bad_x_1] = self.rand_gen.uniform(90,270,np.size(only_bad_x_1))
                self.theta[bad_x_2] = self.rand_gen.uniform(-90,90,np.size(only_bad_x_2))
                self.theta[bad_y_1] = self.rand_gen.uniform(0,180,np.size(only_bad_y_1))
                self.theta[bad_y_2] = self.rand_gen.uniform(180,360,np.size(only_bad_y_2))

                self.theta = self.theta%360

                        
    def save_info(self):

        job_str = "new_Job" + str(self.job) + "_" + "_"+str(self.success_rad)
        init_x_trans = self.initial_x
        init_y_trans = self.initial_y
        init_theta_trans = self.initial_thetas
        success_times = self.success_times
         
        sp.savetxt(self.out_dir+job_str + "init_xs", init_x_trans)
        sp.savetxt(self.out_dir+job_str + "init_ys", init_y_trans)
        sp.savetxt(self.out_dir+job_str + "init_thetas", init_theta_trans)
        sp.savetxt(self.out_dir+job_str + "success_times", success_times)
        

