## codified olfactory navigation models
## to be used with new_packet_sweep_navigation_wrapper
import numpy as np
import scipy as sp

class Careful_Combo:

	def __init__(self, num_steps, num_flies, delta_t, rand_gen, v_0 = 10.1, alpha_2 = 0,
         beta = 3.872, tau_ON = 2, beta_2 = 0,
         init_is_walking = True, whiff_threshold = 0.085, signal_mean = 1.0, signal_std = 0.0,
         whiff_sigma_coeff = 3, ws_lambda_0 = 0.78, sw_lambda_0 = 0.29, tau_A = 9.8,
         sw_tau_H = 2, turn_lambda = 1/0.75, alpha = 0.242, tau_turn = 2.0, turn_mean = 30.0,
         turn_std = 8, no_turn_mean = 0, no_turn_std_factor = 0, base_upturn_rate=0.5, new_on = True, noise = False, w_sigma = None, on_sigma = None):

		#Initializing base simulation parameters

		self.delta_t = delta_t
		self.v_0 = v_0
		self.speed = v_0
		self.num_steps = num_steps
		self.num_flies = num_flies

		#Nagel and New Parameters

		#self.k1 = k1
		#self.nu_1 = nu_1
		#self.nu_2 = nu_2
		#self.nu_3 = nu_3
		self.beta = beta
		#self.nu_5 = nu_5
		#self.k_d = k_d
		#self.k7 = k7
		self.tau_A = tau_A
		self.tau_ON = tau_ON
		#self.old_tau_ON = old_tau_ON
		#self.tau_fast = tau_fast
		#self.tau_slow = tau_slow
		self.alpha_2 = alpha_2
		self.beta_2 = beta_2

		self.signal_mean = signal_mean
		self.signal_std = signal_std
		self.new_on = new_on
		self.noise = noise
		self.w_sigma = w_sigma
		self.on_sigma = on_sigma

		#random numbers
	
		self.ranfs = rand_gen.random_sample(size=(self.num_flies * self.num_steps * 5))
		self.randns = rand_gen.normal(size=(self.num_flies * self.num_steps * 5))
		self.rand_gen = rand_gen
		self.ranf_count = 0
		self.randn_count = 0

		# walk-to-stop parameters                                                                                                                                                                  

		self.ws_lambda_0 = ws_lambda_0
		#self.ws_del_lambda = ws_del_lambda
		#self.ws_tau_R = ws_tau_R

		# stop-to-walk parameters                                                                                                                                 
	
		self.sw_lambda_0 = sw_lambda_0
		#self.sw_del_lambda = sw_del_lambda
		self.sw_tau_H = sw_tau_H

		# turn parameters                                                                                                                                                                          

		self.turn_lambda = turn_lambda
		self.alpha = alpha
		self.tau_turn = tau_turn
		self.turn_mean = turn_mean
		self.turn_std = turn_std
		self.no_turn_mean = no_turn_mean
		self.no_turn_std_factor = no_turn_std_factor
		self.no_turn_std = self.no_turn_std_factor*self.delta_t
		self.base_upturn_rate = base_upturn_rate

		# whiff parameters                                                                                                                                
	
		self.whiff_threshold = whiff_threshold
		self.whiff_min = signal_mean + whiff_sigma_coeff * signal_std
		self.odor_thresh = signal_mean + whiff_sigma_coeff*signal_std

                                                                                                                      
		self.whiff_hits = sp.zeros((self.num_steps, self.num_flies))
		self.hits_window = sp.array([self.whiff_hits[0]])
		self.last_switch = sp.zeros(self.num_flies, dtype=int)
		self.wt = sp.zeros(self.num_flies)
		self.last_whiff = sp.full(self.num_flies, -self.whiff_threshold)
		self.last_whiff_onset = sp.full(self.num_flies, -self.whiff_threshold)
		self.encountered_low = sp.ones(self.num_flies, dtype = bool)
		self.whiff_on = sp.zeros(self.num_flies, dtype = bool)
		#self.last_whiff_x = sp.zeros(self.num_flies)
		#self.last_whiff_y = sp.zeros(self.num_flies)
		#self.had_whiff = sp.zeros(self.num_flies)



		#state information                                                                                                           
	
		self.is_walking = sp.full(self.num_flies, init_is_walking)
		#self.walks = sp.zeros((self.num_steps, self.num_flies))
		#self.stops = sp.zeros((self.num_steps, self.num_flies))
		#self.turns = sp.zeros((self.num_steps, self.num_flies))
		self.ws_fn = sp.zeros(self.num_flies)
		self.sw_fn = sp.zeros(self.num_flies)
		self.time = sp.zeros((self.num_steps, self.num_flies))
		self.turn_fn = sp.zeros(self.num_flies)

		self.vx = sp.zeros(self.num_flies)
		self.vy = sp.zeros(self.num_flies)
		self.del_theta = sp.zeros(self.num_flies)

		self.Aa = sp.zeros(self.num_flies)
		#self.R1 = sp.zeros(self.num_flies)
		#self.R2 = sp.zeros(self.num_flies)
		self.Cc = sp.zeros(self.num_flies)
		self.CL = sp.zeros(self.num_flies)
		self.CR = sp.zeros(self.num_flies)
		self.ON = sp.zeros(self.num_flies)
		#self.OFF = sp.zeros(self.num_flies)
		#self.rho = sp.zeros(self.num_flies)
		self.Vv = sp.zeros(self.num_flies)


	def _dA_dt(self, odor, i):
		
		return (odor - self.Aa)/self.tau_A

	def _new_Cc_eval(self,odor,i):

		return 0.5 * (odor > self.odor_thresh)


	def _Cc_eval(self, odor, i):

		return odor/(odor + self.k_d + self.Aa)


	def _dON_dt(self, i):

		return (self.Cc - self.ON)/self.tau_ON


	#def _dR1_dt(self, i):

		#return (self.Cc - self.R1)/self.tau_fast


	#def _dR2_dt(self, i):

		#return (self.Cc - self.R2)/self.tau_slow


	#def _OFF_eval(self, i):

		#return (np.maximum(0, self.R2 - self.R1))


	def _Vv_eval(self, i):

		return self.v_0 + np.zeros(self.num_flies)



	#Nirag/Hope/New Helper Functions


	def _ws_rate(self,i):

		R = sp.zeros(self.num_flies)

		#whiff_fly_idxs = sp.where(self.last_whiff >= 0)[0]

		#R[whiff_fly_idxs] = sp.exp(-1*(self.time[i, whiff_fly_idxs] - self.last_whiff[whiff_fly_idxs])/self.ws_tau_R)

		res = self.ws_lambda_0 + R 

		return res

	def _turn_rate(self, i):

		R = sp.zeros(self.num_flies)

		res = R + self.turn_lambda  

		return res

	def _sw_rate(self, i):

		conv = sp.zeros(self.num_flies)

		"""

		for fly in range(self.num_flies):
			
			switch_idx = self.last_switch[fly]
			hits_past_switch = self.whiff_hits[switch_idx:i+1,fly]
			len_stop = len(hits_past_switch)

			# exponential decay filter                                                                                                                                                       
																																																	   
			decay_fn = sp.exp(sp.negative((self.time[switch_idx:i+1, fly]-
			sp.full_like(self.time[switch_idx:i+1,fly],self.time[switch_idx, fly])))/
			sp.full_like(self.time[switch_idx:i+1, fly],self.sw_tau_H))

			conv[fly] = (sp.convolve(decay_fn,hits_past_switch)[-min(len(decay_fn), len(hits_past_switch))])

		"""

		res = self.sw_lambda_0 + (self.alpha_2*self.wt + self.beta_2 * self.ON)


		return res


	def _turnL_prob(self, theta, i):

		one_idxs = sp.where(theta <= 180)[0]
		neg_one_idxs = sp.where(theta > 180)[0]

		turn_dir = sp.zeros(self.num_flies)
		base_turnL_rate = sp.zeros(self.num_flies)

		turn_dir[one_idxs] = 1.
		base_turnL_rate[one_idxs] = self.base_upturn_rate

		turn_dir[neg_one_idxs] = -1.
		base_turnL_rate[neg_one_idxs] = 1 - self.base_upturn_rate

		return sp.maximum(sp.minimum((1/(1+sp.exp(-turn_dir*(self.alpha * self.wt + self.beta*self.ON)))),
		 sp.full(self.num_flies,1)),
		 sp.full(self.num_flies,0))


	def _wt(self,i):

		# decays exponentially if no whiff, jumps by 1 on whiff onset

		whiff_flies = self.whiff_hits[i] == 1

		not_whiff_flies = self.whiff_hits[i] == 0

		self.wt[whiff_flies] = self.wt[whiff_flies] * np.exp(-self.delta_t/self.tau_turn) + 1

		self.wt[not_whiff_flies] = self.wt[not_whiff_flies] * np.exp(-self.delta_t/self.tau_turn)

		if self.noise == "Fixed":

			self.wt = self.wt + self.w_sigma * self.randns[self.randn_count : self.randn_count + self.num_flies]
			self.wt[self.wt < 0] = 0
			self.randn_count = self.randn_count + self.num_flies

		elif self.noise == "Relative":

			w_sigmas = self.wt/20
			self.wt = self.wt + self.rand_gen.normal(0, w_sigmas)
			self.wt[self.wt < 0] = 0


	def _happen(self, prob):

		rand = self.ranfs[self.ranf_count : self.ranf_count + self.num_flies]
		self.ranf_count = self.ranf_count + self.num_flies
		return (rand < prob)

	# rate is an array of length self.num_flies                                                                                                                                                          
																																														   
	def _poisson_happen(self, rate):

		prob = rate*self.delta_t

		return(self._happen(prob))


	def update(self, i, odor_L, odor_R, theta, x, y):

		#Euler integrate to update deterministic variables

		odor = 1/2*(odor_L+odor_R)

		#self.Aa = self.Aa + self.delta_t * self._dA_dt(odor,i)

		if self.new_on == False:

			self.Aa = self.Aa + self.delta_t*self._dA_dt(odor,i)
			self.Cc = self._Cc_eval(odor,i)
			self.CL = self._Cc_eval(odor_L,i)
			self.CR = self._Cc_eval(odor_R,i)
		
		else:

			self.Cc = self._new_Cc_eval(odor, i)
			self.CL = self._new_Cc_eval(odor_L, i)
			self.CR = self._new_Cc_eval(odor_R, i)


		self.ON = self.ON + self.delta_t * self._dON_dt(i)

		if self.noise == "Fixed":

			self.ON = self.ON + self.on_sigma * self.randns[self.randn_count : self.randn_count + self.num_flies]
			self.ON[self.ON < 0] = 0
			self.randn_count = self.randn_count + self.num_flies

		elif self.noise == "Relative":

			on_sigmas = self.ON/20

			self.ON = self.ON + self.rand_gen.normal(0, on_sigmas)
			self.ON[self.ON < 0] = 0


		#self.R1 = self.R1 + self.delta_t * self._dR1_dt(i)
		#self.R2 = self.R2 + self.delta_t * self._dR2_dt(i)
		#self.OFF = self._OFF_eval(i)

		#update position variables                                                                                                                                                              
																																													   
		self.Vv = self._Vv_eval(i)
	
		#update walking and stopping
		
		high_sig_idxs = odor > self.whiff_min

		new_high_sig_idxs = high_sig_idxs * self.encountered_low

		self.last_whiff_onset[new_high_sig_idxs==1] = self.time[i,0]

		low_sig_idxs = 1 - high_sig_idxs

		self.encountered_low[low_sig_idxs==1] = 1
		self.encountered_low[high_sig_idxs==1] = 0

		whiff_end_idxs = self.whiff_on * low_sig_idxs

		#self.had_whiff[whiff_end_idxs==1] = 1

		#print("Calculation = ", self.time[i,:] - self.last_whiff)

		T_sufficient = (self.time[i,:] - self.last_whiff) >= self.whiff_threshold

		whiff_idxs = new_high_sig_idxs*T_sufficient

		not_whiff_idxs = 1 - whiff_idxs

		#self.last_whiff_x[whiff_end_idxs==1] = x[i][whiff_end_idxs==1]
		#self.last_whiff_y[whiff_end_idxs==1] = y[i][whiff_end_idxs==1]


		self.last_whiff[whiff_idxs==1] = self.time[i,0]

		self.whiff_on = whiff_idxs + self.whiff_on * high_sig_idxs

		#self.time_since_last_whiff_off[i,whiff_end_idxs==1] = 0
		#self.time_since_last_whiff_off[i,(1-whiff_end_idxs)==1] = self.time_since_last_whiff_off[i,1-whiff_end_idxs] + self.delta_t 

		#self.time_since_last_whiff_on[i,whiff_idxs==1] = 0
		#self.time_since_last_whiff_on[i,(1-whiff_idxs)==1] = self.time_since_last_whiff_on[i,1-whiff_idxs] + self.delta_t 			

		self.whiff_hits[i, whiff_idxs==1] = 1.0
		self.whiff_hits[i, not_whiff_idxs==1] = 0.0
	
		
		if(len(self.hits_window) < 500):
 
			self.hits_window = self.whiff_hits[:i+1]
		
		else:

			self.hits_window = self.whiff_hits[i-500:i+1]		


		# update sw_fn, ws_fn                                                                                                                                                                                                                                                                                                                                                                                 
	
		self.ws_fn = self._ws_rate(i) * self.is_walking
		self.sw_fn = self._sw_rate(i) * sp.logical_not(self.is_walking)
		self.turn_fn = self._turn_rate(i)

		# determine wt for this step                                                                                                                                                               
		self._wt(i)

		walking_flies = sp.where(self.is_walking)[0]
		stopped_flies = sp.where(sp.logical_not(self.is_walking))[0]


		stopping_flies = sp.intersect1d(walking_flies, sp.where(sp.full(self.num_flies, False))[0])
		turning_flies = sp.intersect1d(walking_flies, sp.where(self._poisson_happen(self.turn_fn))[0])

		turning_flies = sp.setdiff1d(turning_flies, stopping_flies)
		keep_walking_flies = sp.setdiff1d(sp.setdiff1d(walking_flies, stopping_flies), turning_flies)

		# if stopped, determine the probability of walking                                                                                                                                         
																																																	  
		start_walking_flies = sp.intersect1d(stopped_flies, sp.where(self._poisson_happen(self.sw_fn))[0])

		#start_walking_turn_suff_time = (self.time[i,:] - self.last_whiff_onset) >= self.whiff_threshold

		#start_walking_turn_suff_odor = low_sig_idxs

		#start_walking_turn_suff_pos = np.logical_or(np.not_equal(self.last_whiff_x, x[i]), np.not_equal(self.last_whiff_y, y[i]))

		#start_walking_turn_suff = start_walking_turn_suff_time * start_walking_turn_suff_pos * start_walking_turn_suff_odor * self.had_whiff

		#print("start_walking_turn_suff = ", start_walking_turn_suff)

		#start_walking_turn_idxs = sp.intersect1d(start_walking_flies, np.arange(0,self.num_flies)[start_walking_turn_suff==1])

		#stopped_not_turning_flies = sp.setdiff1d(stopped_flies, start_walking_turn_idxs)

		#start_walking_interm_vec_x = self.last_whiff_x[start_walking_turn_idxs] - x[i, start_walking_turn_idxs]
		#start_walking_interm_vec_y = self.last_whiff_y[start_walking_turn_idxs] - y[i, start_walking_turn_idxs]

		#start_walking_turn_interm_thetas = (np.arctan2(start_walking_interm_vec_y, start_walking_interm_vec_x)*180/np.pi)%360

		#start_turn_thetas = np.zeros(len(start_walking_turn_idxs))

		#start_turn_thetas[start_walking_turn_interm_thetas<0] = 180 - 180/np.pi*start_walking_turn_interm_thetas[start_walking_turn_interm_thetas<=0]
		#start_turn_thetas[start_walking_turn_interm_thetas>=0] = 180/np.pi*start_walking_turn_interm_thetas[start_walking_turn_interm_thetas>=0]

		#start_turn_L_suff = (start_walking_turn_interm_thetas>=theta[i, start_walking_turn_idxs])*(start_walking_turn_interm_thetas<=180+theta[i, start_walking_turn_idxs]) + (start_walking_turn_interm_thetas<=theta[i, start_walking_turn_idxs])*(start_walking_turn_interm_thetas<=theta[i, start_walking_turn_idxs]-180)

		#print("start turn l suff = ", start_turn_L_suff)

		#start_walking_turn_L_idxs = start_walking_turn_idxs[start_turn_L_suff==1]

		#start_walking_turn_R_idxs = start_walking_turn_idxs[np.logical_not(start_turn_L_suff)==1]

		"""

		if start_walking_turn_idxs:

			print("start walking turn idxs = ", start_walking_turn_idxs)

			print("current position for stop-turning flies = ", [x[i, start_walking_turn_idxs], y[i, start_walking_turn_idxs]])
			print("position of last whiff for stop-turning flies = ", [self.last_whiff_x[start_walking_turn_idxs], self.last_whiff_y[start_walking_turn_idxs]])

			print("current thetas for stop-turning flies = ", theta[i, start_walking_turn_idxs])

			print("thetas to last whiff = ", start_walking_turn_interm_thetas)

			print("start walking turn L idxs = ", start_walking_turn_L_idxs)
			print("start walking turn R idxs = ", start_walking_turn_R_idxs)	
		"""

		keep_stopped_flies = sp.setdiff1d(stopped_flies, start_walking_flies)

		all_turn_L_idxs = sp.intersect1d(turning_flies,sp.where(self._happen(self._turnL_prob(theta, i)))[0])
		all_turn_R_idxs = sp.setdiff1d(turning_flies, all_turn_L_idxs)

		#all_turn_L_idxs = np.union1d(turnL_idxs, start_walking_turn_L_idxs)

		#all_turn_R_idxs = np.union1d(turnR_idxs, start_walking_turn_R_idxs)

		#if start_walking_turn_idxs:

			#print("all turn L idxs = ", all_turn_L_idxs)
			#print("all turn R idxs = ", all_turn_R_idxs)

		# left turns                                                                                                                                                                                                                                                                                                                  
		del_theta = (self.turn_mean + self.turn_std * self.randns[self.randn_count : self.randn_count + len(all_turn_L_idxs)])


		self.randn_count = self.randn_count + len(all_turn_L_idxs)
		self.del_theta[all_turn_L_idxs] = del_theta

		# right turns                                                                                                                                                                              
																																																	 
		del_theta = (-self.turn_mean + self.turn_std * self.randns[self.randn_count : self.randn_count + len(all_turn_R_idxs)])
		self.randn_count = self.randn_count + len(all_turn_R_idxs)
		self.del_theta[all_turn_R_idxs] = del_theta
		
		self.is_walking[stopping_flies] = False
		#self.walks[i, stopping_flies] = 0.0
		#self.stops[i, stopping_flies] = 1.0
		#self.turns[i, stopping_flies] = 0.0
		self.last_switch[stopping_flies] = i
		self.del_theta[stopping_flies] = 0.0

		# turning flies                                                                                                                                                                            
																																																	   
		#self.walks[i, turning_flies] = 1.0
		#self.stops[i, turning_flies] = 0.0
		#self.turns[i, turning_flies] = 1.0

		# walking flies that won't turn or stop                                                                                                                                                    
																																																		
		del_theta = (self.no_turn_mean + self.no_turn_std * self.randns[self.randn_count : self.randn_count + len(keep_walking_flies)])
		self.randn_count = self.randn_count + len(keep_walking_flies)
		self.del_theta[keep_walking_flies] = del_theta

		#self.walks[i, keep_walking_flies] = 1.0
		#self.stops[i, keep_walking_flies] = 0.0
		#self.turns[i, keep_walking_flies] = 0.0


		# stopped flies                                                                                                                                                                            
		#self.turns[i, stopped_flies] = 0.0
		self.del_theta[stopped_flies] = 0.0

		# stopped flies that will walk                                                                                                                                                             
																																																	   
		self.is_walking[start_walking_flies] = 1.0
		#self.walks[i, start_walking_flies] = 1.0
		#self.stops[i, start_walking_flies] = 0.0
		self.last_switch[start_walking_flies] = i

		# stopped flies that will stay stopped                                                                                                                                                                                                                                                                                                                                                                
		#self.walks[i, keep_stopped_flies] = 0.0
		#self.stops[i, keep_stopped_flies] = 1.0

		self.time[i+1] = self.time[i] + self.delta_t

		# update positions                                                                                                                                                                          
		new_thetas = theta + self.del_theta 

		#if start_walking_turn_idxs:

			#print("new thetas for stop turn flies = ", new_thetas[start_walking_turn_idxs])

		#if start_walking_turn_idxs == 0:

			#print("actual final thetas = ", new_thetas)

		self.vx[walking_flies] = sp.cos(new_thetas[walking_flies] * sp.pi / 180.) * self.Vv[walking_flies]
		self.vx[stopped_flies] = 0.0
		self.vy[walking_flies] = sp.sin(new_thetas[walking_flies] * sp.pi / 180.) * self.Vv[walking_flies]
		self.vy[stopped_flies] = 0.0

		dx = self.vx * self.delta_t
		dy = self.vy * self.delta_t

			
		return new_thetas, dx, dy

	"""

	def save_data(self, out_dir, job_str):

		walks_trans = sp.transpose(self.walks)
		turns_trans = sp.transpose(self.turns)
		hits_trans = sp.transpose(self.whiff_hits)
		wt_trans = sp.transpose(self.wt)

		sp.savetxt(out_dir + job_str + "walks", walks_trans)
		sp.savetxt(out_dir + job_str + "turns", turns_trans)
		sp.savetxt(out_dir + job_str + "whiffs", hits_trans)
		sp.savetxt(out_dir + job_str + "wts", wt_trans)
		ON_trans = sp.transpose(self.ON)
		OFF_trans = sp.transpose(self.OFF)

		sp.savetxt(out_dir+job_str + "ONs", ON_trans)
		sp.savetxt(out_dir+job_str + "OFFs", OFF_trans)
	
	"""


def full_antenna(std_box, theta, px, py, a=1.5, b=5.5, dist=0):

		ellipse_val_1 = (np.cos(theta*np.pi/180)*std_box[:,0] + np.sin(theta*np.pi/180)*std_box[:,1])**2/(a**2)

		ellipse_val_2 = (-np.sin(theta*np.pi/180)*std_box[:,0] + np.cos(theta*np.pi/180)*std_box[:,1])**2/(b**2)

		ellipse_points = std_box[ellipse_val_1+ellipse_val_2 <= 1]

		s_l_cond_1 = (theta+180) >= (np.arctan2(ellipse_points[:,1],ellipse_points[:,0])*180/np.pi)%360

		s_l_cond_2 = (np.arctan2(ellipse_points[:,1],ellipse_points[:,0])*180/np.pi)%360 >= theta

		s_r_cond_1 = (np.arctan2(ellipse_points[:,1],ellipse_points[:,0])*180/np.pi)%360 <= theta

		s_r_cond_2 = (np.arctan2(ellipse_points[:,1],ellipse_points[:,0])*180/np.pi)%360 >= (theta+180)


		b_l_cond_1 = (np.arctan2(ellipse_points[:,1],ellipse_points[:,0])*180/np.pi)%360 >= theta

		b_l_cond_2 = (np.arctan2(ellipse_points[:,1],ellipse_points[:,0])*180/np.pi)%360 <= theta-180

		b_r_cond_1 = theta -180 <= (np.arctan2(ellipse_points[:,1],ellipse_points[:,0])*180/np.pi)%360

		b_r_cond_2 = (np.arctan2(ellipse_points[:,1],ellipse_points[:,0])*180/np.pi)%360 <= theta


		if 0 <= theta <= 180:

			left_box = ellipse_points[s_l_cond_1*s_l_cond_2]

			right_box = ellipse_points[s_r_cond_1+s_r_cond_2]


		if theta >= 180:

			left_box = ellipse_points[b_l_cond_1 + b_l_cond_2]

			right_box = ellipse_points[b_r_cond_1*b_r_cond_2]


		left_box = np.append(left_box,[[0,0]], axis = 0)
		right_box = np.append(right_box,[[0,0]], axis = 0)

		left_box = np.unique(left_box, axis=0)
		right_box = np.unique(right_box, axis=0)

		translation_x = np.rint(dist*np.cos(theta*np.pi/180))

		translation_y = np.rint(dist*np.sin(theta*np.pi/180))

		t_v_x = px + translation_x

		t_v_y = py + translation_y

		left_box = np.array([t_v_x,t_v_y]) + left_box

		right_box = np.array([t_v_x,t_v_y]) + right_box

		return left_box, right_box

