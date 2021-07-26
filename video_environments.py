import numpy as np
import scipy.spatial
from load_data import *


class complex():

    def __init__(self, data_dir, file, arena_x_indices=None, arena_y_indices=None, start_frame = 0, bck_file = None, bck_sub = 1, full_arena = False, high_int_vid = False,
       intermediate = False, low_comp = False, bck_img_array = None, num_frames = None, loop = False, signal_thresh = 0):

        self.arena_x_indices = arena_x_indices
        self.arena_y_indices = arena_y_indices
        self.data_dir = data_dir
        self.file = file
        self.bck_file = bck_file
        self.bck_sub = bck_sub
        self.full_arena = full_arena
        self.start_frame = start_frame
        self.num_frames = num_frames
        self.loop = loop
        self.intermediate = intermediate
        self.high_int_vid = high_int_vid
        self.signal_thresh = signal_thresh
        self.low_comp = low_comp 
        self.type = 'frame'

        if bck_img_array == None:

            self.bck_img_array = None

        else:

            self.bck_img_array = np.load(self.data_dir + "/" + bck_img_array)


    def generate_env(self, time_step, delta_t, rand_gen):

        if self.full_arena == True and self.loop == False:

            if self.high_int_vid == True:

                signal = high_int_load_vid_by_frm(subdir=self.data_dir, file=self.file, frame = self.start_frame + time_step, bck_sub=self.bck_sub, bck_file=self.bck_file)


            elif self.intermediate == True:

                signal = load_int_vid_by_frm(subdir=self.data_dir, file=self.file, frame = self.start_frame + time_step, bck_sub=self.bck_sub, bck_file=self.bck_file)
            
            elif self.low_comp == True:

                signal = load_low_comp_by_frm(subdir=self.data_dir, file=self.file, frame = self.start_frame + time_step, bck_img_array = self.bck_img_array)

            else:

                signal = load_vid_by_frm(subdir=self.data_dir, file=self.file, frame = self.start_frame + time_step, bck_sub=self.bck_sub, bck_file=self.bck_file)
                
        
        elif self.full_arena==True and self.loop == True:

            frame = (self.start_frame + time_step)%num_frames + self.start_frame - 1

            if self.high_int_vid == False:

                if self.intermediate == False:

                    signal = load_vid_by_frm(subdir=self.data_dir, file=self.file, frame = frame, bck_sub=self.bck_sub, bck_file=self.bck_file)

                else:

                    signal = load_int_vid_by_frm(subdir=self.data_dir, file=self.file, frame = frame, bck_sub=self.bck_sub, bck_file=self.bck_file)

            else:

                signal = high_int_load_vid_by_frm(subdir=self.data_dir, file=self.file, frame = frame, bck_sub=self.bck_sub, bck_file=self.bck_file)

        else:

            signal = load_vid_by_frm(subdir=self.data_dir, file=self.file, frame= self.start_frame + time_step, bck_sub=self.bck_sub,
                                        bck_file=self.bck_file)[self.arena_x_indices[0]:self.arena_x_indices[1], self.arena_y_indices[0]:self.arena_y_indices[1]]


        signal[signal<self.signal_thresh] = 0
        
        return signal













