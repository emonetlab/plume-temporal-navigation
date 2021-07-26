import numpy as np
import scipy.spatial


class packets:

    def __init__(self, rate = 10, dw_speed = 150, init_intensity = 3827.24, cw_type = 'Gaussian', eddy_D = 0.1*(10**4), 
    	       r0 = 10, packet_D = 10, source_x = 10, source_y = 90, max_x = None, var_size = False, delay_steps = 267, delta_t = 0.01, 
    	      rand_gen = np.random.RandomState(0), signal_noise = False, noise_std = 0.1):

        self.type = 'packets'
        self.rate = rate
        self.eddy_D = eddy_D
        self.r0 = r0
        self.packet_D = packet_D
        self.source_x = source_x
        self.source_y = source_y
        self.init_intensity = init_intensity
        self.max_x = max_x
        self.rand_gen = rand_gen


        self.packet_xs = np.array([source_x])
        self.packet_ys = np.array([source_y])
        self.packet_durations = np.array([0]) 
        self.packet_sizes = np.array([r0])

        self.cw_type = cw_type
        self.dw_speed = dw_speed

        self.var_size = var_size
        self.noise = signal_noise
        self.noise_std = noise_std

        for i in range(0, delay_steps):

        	packet_pos, packet_sizes = self.generate_packets(delta_t = delta_t, rand_gen = rand_gen)


    def generate_packets(self, delta_t, rand_gen):

        rand_unif = rand_gen.random_sample(1)

        prob = 1 - np.exp(-self.rate*delta_t)

        if rand_unif < prob:

            self.packet_xs = np.append(self.packet_xs, self.source_x)
            self.packet_ys = np.append(self.packet_ys, self.source_y)
            self.packet_durations = np.append(self.packet_durations, 0)

            if self.var_size == False:

            	self.packet_sizes = np.append(self.packet_sizes, self.r0)

            else:

            	u = rand_gen.random_sample(1)

            	if u <= 1/3:

            		self.packet_sizes = np.append(self.packet_sizes, 3*u*self.r0)
            	
            	else:

            		size = 4*self.r0/(3-3*u)**2

            		self.packet_sizes = np.append(self.packet_sizes, size)


        self.packet_xs = delta_t*self.dw_speed + self.packet_xs

        if self.max_x != None:

            bools = self.packet_xs < self.max_x

            self.packet_xs = self.packet_xs[bools]

            self.packet_ys = self.packet_ys[bools]

            self.packet_durations = self.packet_durations[bools]

            self.packet_sizes = self.packet_sizes[bools]


        if self.cw_type == 'Gaussian':

            perts = rand_gen.normal(loc = 0, scale = np.sqrt(2*self.eddy_D*delta_t), size = (len(self.packet_ys), 2))

        elif self.cw_type == 'Cauchy':

            perts = np.sqrt(self.eddy_D*delta_t) * rand_gen.standard_cauchy((len(self.packet_ys),2))

        self.packet_ys = self.packet_ys + perts[:,0]

        self.packet_xs = self.packet_xs + perts[:,1]

        self.packet_durations = self.packet_durations + delta_t

        self.packet_sizes = (self.r0 ** (2) + 4*self.packet_D*self.packet_durations)**0.5


        packet_pos_mat = np.zeros((len(self.packet_xs), 2))

        packet_pos_mat[:,0] = self.packet_xs

        packet_pos_mat[:,1] = self.packet_ys

        return packet_pos_mat, self.packet_sizes


    def compute_sig(self, left_points, right_points, packet_pos, packet_sizes, rand_gen):

        all_points = np.vstack((left_points, right_points))

        all_distances = scipy.spatial.distance_matrix(all_points, packet_pos)

        scaled_all_distances = all_distances/(packet_sizes[None,:])

        gaussian_part = np.exp(-(scaled_all_distances)**2)

        packet_prefactor = self.init_intensity/(np.pi*packet_sizes**2)

        all_signals = gaussian_part * packet_prefactor[None, :]

        all_total_signals = np.sum(all_signals, axis = 1)

        total_left_sig = all_total_signals[0:len(left_points[:,0])-1]

        total_right_sig = all_total_signals[len(left_points[:,0]):]

        left_sig = np.mean(total_left_sig)
        right_sig = np.mean(total_right_sig)

        if self.noise == True:

            sig_noise = rand_gen.normal(loc = 0, scale = self.noise_std, size = (2))

            left_sig = left_sig + sig_noise[0]
            right_sig = right_sig + sig_noise[1]

            left_sig = left_sig * (left_sig > 0) 
            right_sig = right_sig * (right_sig > 0)

        return left_sig, right_sig 
















