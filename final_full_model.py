## codified olfactory navigation models
## to be used with navigation_wrapper.py
import numpy as np
import scipy as sp


class Careful_Combo:

	def __init__(self, num_steps, num_flies, delta_t, rand_gen, v_0 = 10.1,
	 beta = 3.872, tau_ON = 2, k_d = 3, tau_A = 9.8,
	 whiff_threshold = 0.085, signal_mean = -0.139, signal_std = 1.08, 
	 whiff_sigma_coeff = 2.749, turn_lambda = 1/0.75, alpha = 0.242, tau_turn = 2.0, turn_mean = 30.0,
	 turn_std = 8, base_upturn_rate=0.5, new_on = True, noise = False):

		#Initializing base simulation parameters

		self.delta_t = delta_t
		self.v_0 = v_0
		self.speed = v_0
		self.num_steps = num_steps
		self.num_flies = num_flies

		#Nagel and New Parameters

		self.alpha = alpha
		self.beta = beta
		self.k_d = k_d
		self.tau_A = tau_A
		self.tau_ON = tau_ON

		self.signal_mean = signal_mean
		self.signal_std = signal_std
		self.new_on = new_on

		self.new_on = new_on
		self.noise = noise

		#random numbers
	
		self.ranfs = rand_gen.random_sample(size=(self.num_flies * self.num_steps * 5))
		self.randns = rand_gen.normal(size=(self.num_flies * self.num_steps))
		self.ranf_count = 0
		self.randn_count = 0

		# turn parameters                                                                                                                                                                          

		self.turn_lambda = turn_lambda
		self.tau_turn = tau_turn
		self.turn_mean = turn_mean
		self.turn_std = turn_std

		# whiff parameters                                                                                                                                
	
		self.whiff_threshold = whiff_threshold
		self.whiff_min = signal_mean + whiff_sigma_coeff * signal_std
		self.odor_thresh = signal_mean + whiff_sigma_coeff * signal_std

                                                                                                                      
		self.whiff_hits = sp.zeros((self.num_steps, self.num_flies))
		self.wt = sp.zeros((self.num_steps, self.num_flies))
		self.last_whiff = sp.full(self.num_flies, -self.whiff_threshold)
		self.encountered_low = sp.ones(self.num_flies, dtype = bool)


		#state information                                                                                                           
		self.turns = sp.zeros((self.num_steps, self.num_flies))
		self.time = sp.zeros((self.num_steps, self.num_flies))
		self.turn_fn = sp.zeros((self.num_steps, self.num_flies))

		self.vx = sp.zeros((self.num_steps, self.num_flies))
		self.vy = sp.zeros((self.num_steps, self.num_flies))
		self.del_theta = sp.zeros((self.num_steps, self.num_flies))

		self.Aa = sp.zeros((self.num_steps, self.num_flies))
		self.Cc = sp.zeros((self.num_steps, self.num_flies))
		self.CL = sp.zeros((self.num_steps, self.num_flies))
		self.CR = sp.zeros((self.num_steps, self.num_flies))
		self.ON = sp.zeros((self.num_steps, self.num_flies))


	def _dA_dt(self, odor, i):
	
		return (odor[i] - self.Aa[i])/self.tau_A

	def _new_Cc_eval(self,odor,i):

		return 0.5 * (odor > self.odor_thresh)


	def _Cc_eval(self, odor, i):

		return odor/(odor + self.k_d + self.Aa[i+1])


	def _dON_dt(self, i):

		return (self.Cc[i+1] - self.ON[i])/self.tau_ON


	def _turn_rate(self, i):

		R = sp.zeros(self.num_flies)

		res = R + self.turn_lambda 

		return res


	def _turnL_prob(self, theta, i):

		one_idxs = theta[i] <= 180
		neg_one_idxs = theta[i] > 180

		turn_dir = sp.zeros(self.num_flies)

		turn_dir[one_idxs] = 1.

		turn_dir[neg_one_idxs] = -1.

		return sp.maximum(sp.minimum((1/(1+sp.exp(-turn_dir*(self.alpha*self.wt[i+1]+self.beta*self.ON[i+1])))),
		  sp.full(self.num_flies,1)),
		  sp.full(self.num_flies,0))


	def _wt(self,i):

		# decays exponentially if no whiff, jumps by 1 on whiff onset

		whiff_flies = self.whiff_hits[i] == 1

		not_whiff_flies = self.whiff_hits[i] == 0

		self.wt[i+1, whiff_flies] = self.wt[i, whiff_flies] * np.exp(-self.delta_t/self.tau_turn) + 1

		self.wt[i+1, not_whiff_flies] = self.wt[i, not_whiff_flies] * np.exp(-self.delta_t/self.tau_turn)

		if self.noise == "Fixed":

			self.wt[i+1] = self.wt[i+1] + self.w_sigma * self.randns[self.randn_count : self.randn_count + self.num_flies]
			self.wt[i+1, self.wt[i+1] < 0] = 0
			self.randn_count = self.randn_count + self.num_flies

		elif self.noise == "Relative":

			w_sigmas = self.wt[i+1]/20
			self.wt[i+1] = self.wt[i+1] + self.rand_gen.normal(0, w_sigmas)
			self.wt[i+1, self.wt[i+1] < 0] = 0


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

		self.Aa[i + 1] = self.Aa[i] + self.delta_t * self._dA_dt(odor,i)

		if self.new_on == False:

			self.Cc[i+1] = self._Cc_eval(odor[i],i)
			self.CL[i+1] = self._Cc_eval(odor_L[i],i)
			self.CR[i+1] = self._Cc_eval(odor_R[i],i)
		
		else:

			self.Cc[i+1] = self._new_Cc_eval(odor[i], i)
			self.CL[i+1] = self._new_Cc_eval(odor_L[i],i)
			self.CR[i+1] = self._new_Cc_eval(odor_R[i],i)

			

		self.ON[i+1] = self.ON[i] + self.delta_t * self._dON_dt(i)																																							  
	
		#update walking and stopping
		
		high_sig_idxs = odor[i] > self.whiff_min

		new_high_sig_idxs = high_sig_idxs * self.encountered_low

		low_sig_idxs = 1 - high_sig_idxs

		self.encountered_low[low_sig_idxs==1] = 1
		self.encountered_low[high_sig_idxs==1] = 0

		T_sufficient = (self.time[i,:] - self.last_whiff) >= self.whiff_threshold

		whiff_idxs = new_high_sig_idxs*T_sufficient

		not_whiff_idxs = 1 - whiff_idxs

		self.last_whiff[whiff_idxs==1] = self.time[i,0]

		self.whiff_hits[i, whiff_idxs==1] = 1.0
		self.whiff_hits[i, not_whiff_idxs==1] = 0.0

		self.turn_fn[i] = self._turn_rate(i)

		# determine wt for this step                                                                                                                                                               
		self._wt(i)

		turning_flies = self._poisson_happen(self.turn_fn[i])
		not_turning_flies = np.logical_not(turning_flies)

		turnL_idxs = turning_flies * self._happen(self._turnL_prob(theta, i))
		turnR_idxs = turning_flies * np.logical_not(turnL_idxs)
		self.turns[i, turning_flies] = 1.0


		# left turns                                                                                                                                                                                                                                                                                                                  
		del_theta = (self.turn_mean + self.turn_std * self.randns[self.randn_count : self.randn_count + np.sum(turnL_idxs)])


		self.randn_count = self.randn_count + np.sum(turnL_idxs)
		self.del_theta[i, turnL_idxs] = del_theta

		# right turns                                                                                                                                                                              
																																																	 
		del_theta = (-self.turn_mean + self.turn_std * self.randns[self.randn_count : self.randn_count + np.sum(turnR_idxs)])
		self.randn_count = self.randn_count + np.sum(turnR_idxs)
		self.del_theta[i, turnR_idxs] = del_theta
		
		# walking flies that won't turn

		self.del_theta[i, not_turning_flies] = 0
		self.turns[i, not_turning_flies] = 0.0

		self.time[i+1] = self.time[i] + self.delta_t

		# update positions                                                                                                                                                                          
		new_thetas = theta[i] + self.del_theta[i] 

		self.vx[i] = self.v_0*sp.cos(new_thetas * sp.pi / 180.)
		self.vy[i] = self.v_0*sp.sin(new_thetas * sp.pi / 180.) 

		dx = self.vx[i] * self.delta_t
		dy = self.vy[i] * self.delta_t

			
		return new_thetas, dx, dy


	def save_data(self, out_dir, job_str):
		turns_trans = sp.transpose(self.turns)
		hits_trans = sp.transpose(self.whiff_hits)
		wt_trans = sp.transpose(self.wt)
		ON_trans = sp.transpose(self.ON)
		CC_trans = sp.transpose(self.Cc)

		sp.savetxt(out_dir + job_str + "turns", turns_trans)
		sp.savetxt(out_dir + job_str + "whiffs", hits_trans)
		sp.savetxt(out_dir + job_str + "wts", wt_trans)
		sp.savetxt(out_dir+job_str + "ONs", ON_trans)



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

