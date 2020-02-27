import os, sys
import pkg_resources
import logging
import numpy as np
from tqdm import tqdm
import glob
import strax
import blosc
import lz4.frame
import matplotlib.pyplot as plt
import seaborn as sns


class manager:
    """
    """

    def __init__(self):

        self.logger = logging.getLogger(__name__)
        self.logger.info('nvdaq_monitor version 1.0')

        self.data_dir_name = '/Users/mzks/xenon/daq_test/data/redax-data/'
        self.data_name_list = []

        self.init_bin_baseline = 10

        self.figsize_height = 16
        self.figsize_width = 16


    def find_latest_run(self, dir_name=None):
        if dir_name == None:
            dir_name = self.data_dir_name

        list_of_files = glob.glob(dir_name+'*')
        latest_file = max(list_of_files, key=os.path.getctime)
        print(latest_file)
        self.run_dir_name = latest_file + '/'


    def select_run(self, run_index_number=None):

        print('data_dir_name: ', self.data_dir_name)
        list_of_files = np.sort(glob.glob(self.data_dir_name+'*'))
        if run_index_number == None:
            for i, file in enumerate(list_of_files):
                print(i, '\t', os.path.split(file)[1])
                if i % 5 == 4: print('-----------------------------------------')

        if run_index_number != None:
            self.run_dir_name = list_of_files[run_index_number] + '/'
            print('Target: ', self.run_dir_name)


    def summarize_subruns(self):

        processing_threads = 8
        n_normal = 0
        subruns_list = []
        n_subruns = []
        n_subruns_post = []
        for i in range(0, 10000):
            subrun_name = str(i).zfill(6)
            if glob.glob(self.run_dir_name + subrun_name) != []:
                subruns_list.append(i)
                file_list = glob.glob(self.run_dir_name + subrun_name + '/*')
                file_list_post = glob.glob(self.run_dir_name + subrun_name + '_post/*')
                n_subruns.append(len(file_list))
                n_subruns_post.append(len(file_list_post))
        print('Run Dir: ', self.run_dir_name)
        print(len(subruns_list), 'subruns exist.')

        for i in range(len(subruns_list)):
            if(n_subruns[i] == processing_threads and n_subruns_post[i] == processing_threads):
                n_normal += 1
            else:
                print('Subrun'+ str(i) + ' has ' + str(n_subruns[i]) +' and '+str(n_subruns_post[i]) +' files in dir and dir_post.')

        print(str(n_normal) +' files are collect.')


    def add_all_subruns(self):

        for i in range(0, 10000):
            subrun_name = str(i).zfill(6)
            if glob.glob(self.run_dir_name + subrun_name) != []:
                self.add_srun_files(self.run_dir_name + subrun_name + '/')
            if glob.glob(self.run_dir_name + subrun_name +'_post') != []:
                self.add_srun_files(self.run_dir_name + subrun_name + '_post/')


    def add_subruns(self, subrun_list = (1,2,3)):

        for i in subrun_list:
            subrun_name = str(i).zfill(6)
            if glob.glob(self.run_dir_name + subrun_name) != []:
                self.add_srun_files(self.run_dir_name + subrun_name + '/')
            if glob.glob(self.run_dir_name + subrun_name +'_post') != []:
                self.add_srun_files(self.run_dir_name + subrun_name + '_post/')


    def add_srun_files(self, subrun_path):

        file_list = glob.glob(subrun_path + '*')
        self.data_name_list.extend([file for file in file_list if os.stat(file).st_size != 0])


    def add_file(self, file_path):
        self.data_name_list.append(file_path)


    def clear_all_subruns(self):
        self.data_name_list = []


    def select_sub(self, list=[]):
        if list == 'all':
            self.add_all_subruns()
            return

        if list != []:
            self.add_subruns(list)
        else:
            self.summarize_subruns()

        return

    def __load_data(self, data_name, compressor='blosc'):

        #print('Target file: ', data_name)
        try:
            self.file = open(data_name, 'rb')
            if compressor == "blosc":
                self.data = blosc.decompress(self.file.read())
            elif compressor == "lz4":
                self.data = lz4.frame.decompress(self.file.read())
            self.darr = np.frombuffer(self.data, dtype=strax.record_dtype())
        except:
            self.logger.warning('Skipped '+ data_name)
            return False

        return True

    def __count_record(self, event):
        return sum([len(event) for event in event])


    def process(self, compressor='blosc'):

        # Prepare
        self.num_of_channel = 32

        self.calced_baselines = [[] for i in range(self.num_of_channel)]
        self.calced_areas = [[] for i in range(self.num_of_channel)]
        self.timestamps= [[] for i in range(self.num_of_channel)]
        self.peak_timings = [[] for i in range(self.num_of_channel)]
        self.waveforms= [[] for i in range(self.num_of_channel)]

        # Loop
        for data_name in tqdm(self.data_name_list):
            if not self.__load_data(data_name, compressor=compressor): continue
            event = []

            for record in self.darr:
                if not event:  # First record in the event
                    event_timestamp = record['time']
                event.append(record['data'])

                if self.__count_record(event)<record['pulse_length']:
                    continue

                else:
                    # End of Event
                    merged_event = np.array([item for sublist in event for item in sublist])[0:record['pulse_length']]
                    calced_baseline = merged_event[0:self.init_bin_baseline].sum()/self.init_bin_baseline
                    calced_area = (calced_baseline-merged_event).sum()

                    self.waveforms[record['channel']].append(merged_event)
                    self.timestamps[record['channel']].append(event_timestamp)
                    self.calced_baselines[record['channel']].append(calced_baseline)
                    self.calced_areas[record['channel']].append(calced_area)
                    self.peak_timings[record['channel']].append(np.argmin(merged_event))

                    event = []

        return True


    def show_rates(self):

        event_numbers = [len(time) for time in self.timestamps]  ## Event numbers
        live_time = np.max(np.max(self.timestamps)) - np.min([np.min(time) for time in self.timestamps if time!=[]])
        rates = [event / live_time * 1.e9 if (event and live_time) else 0 for event in event_numbers]
        print('Live time : ', live_time*1.e-9, ' sec.')

        plt.bar(np.arange(0, self.num_of_channel), rates)
        plt.xlabel('Channel')
        plt.ylabel('Rate (Hz)')


    def show_counts(self):

        event_numbers = [len(time) for time in self.timestamps]  ## Event numbers

        plt.bar(np.arange(0, self.num_of_channel), event_numbers)
        plt.xlabel('Channel')
        plt.ylabel('Counts')


    def show_pulse(self, channel=0, event=0):
        plt.plot(np.arange(len(self.waveforms[channel][event])), self.waveforms[channel][event])
        plt.xlabel('Sample')
        plt.ylabel('ADC Value')


    def show_area(self, channel=0, hist_range=None, bins=None):
        plt.hist(self.calced_areas[channel], lw=0, range=hist_range, bins=bins)
        plt.xlabel('ADC Integration')
        plt.ylabel('Counts')


    def show_areas(self, hist_range=None, bins=None):

        fig1, axs1 = plt.subplots(8, 4, figsize=(self.figsize_height,self.figsize_width), constrained_layout=True)
        for i in range(8):
            for j in range(4):
                channel = i*4 + j
                axs1[i, j].hist(self.calced_areas[channel], lw=0, label='ch.'+str(channel), range=hist_range, bins=bins)
                axs1[i, j].legend()
                axs1[i, j].set_xlabel('ADC Integration')
                axs1[i, j].set_ylabel('Counts')


    def show_baseline(self, channel=0, hist_range=None, bins=None):
        plt.hist(self.calced_baselines[channel], lw=0, range=hist_range, bins=bins)
        plt.xlabel('ADC Value')
        plt.ylabel('Counts')


    def show_baselines_bar(self):

        means = [np.mean(baseline) if (baseline != []) else 0 for baseline in self.calced_baselines]
        stds = [np.std(baseline) if (baseline != []) else 0 for baseline in self.calced_baselines]
        plt.bar(np.arange(0, self.num_of_channel), means, yerr=stds)
        plt.xlabel('Channel')
        plt.ylabel('Baseline ADC Value')


    def show_baselines_rms_bar(self):

        stds = [np.std(baseline) if (baseline != []) else 0 for baseline in self.calced_baselines]
        plt.bar(np.arange(0, self.num_of_channel), stds)
        plt.xlabel('Channel')
        plt.ylabel('Baseline RMS ADC Value')


    def show_baselines(self, hist_range=None, bins=None):

        fig1, axs1 = plt.subplots(8, 4, figsize=(self.figsize_height,self.figsize_width), constrained_layout=True)
        for i in range(8):
            for j in range(4):
                channel = i*4 + j
                axs1[i, j].hist(self.calced_baselines[channel], lw=0, label='ch.'+str(channel), range=hist_range, bins=bins)
                axs1[i, j].legend()
                axs1[i, j].set_xlabel('ADC Value')
                axs1[i, j].set_ylabel('Counts')


    def show_timing(self, channel=0, hist_range=None, bins=None):
        plt.hist(self.peak_timings[channel], lw=0, range=hist_range, bins=bins)
        plt.xlabel('Peak timing (index)')
        plt.ylabel('Counts')


    def show_timings(self, hist_range=None, bins=None):

        fig1, axs1 = plt.subplots(8, 4, figsize=(self.figsize_height,self.figsize_width), constrained_layout=True)
        for i in range(8):
            for j in range(4):
                channel = i*4 + j
                axs1[i, j].hist(self.peak_timings[channel], lw=0, label='ch.'+str(channel), range=hist_range, bins=bins)
                axs1[i, j].legend()
                axs1[i, j].set_xlabel('Peak timing (index)')
                axs1[i, j].set_ylabel('Counts')


    def show_diff_time(self, channel=0, hist_range=None, bins=None):

        sorted_timestamp = np.sort(self.timestamps[channel])
        buf1 = np.diff(sorted_timestamp)
        plt.hist(np.diff(sorted_timestamp), lw=0, range=hist_range, bins=bins)
        plt.xlabel('Time from previous event (ns)')
        plt.ylabel('Counts')


    def show_diff_times(self, hist_range=None, bins=None):

        sorted_timestamps = [np.sort(timestamp) for timestamp in self.timestamps]
        fig1, axs1 = plt.subplots(8, 4, figsize=(self.figsize_height,self.figsize_width), constrained_layout=True)
        for i in range(8):
            for j in range(4):
                channel = i*4 + j
                axs1[i, j].hist(np.diff(sorted_timestamps[channel]), lw=0, label='ch.'+str(channel), range=hist_range, bins=bins)
                axs1[i, j].legend()
                axs1[i, j].set_xlabel('Time from previous event (ns)')
                axs1[i, j].set_ylabel('Counts')


    def show_timestamp(self, channel=0):
        sorted_timestamp = np.sort(self.timestamps[channel])
        plt.plot(sorted_timestamp)
        plt.ylabel('Timestamp (ns)')
        plt.xlabel('Events')


    def show_timestamps(self):
        sorted_timestamps = [np.sort(timestamp) for timestamp in self.timestamps]
        fig1, axs1 = plt.subplots(8, 4, figsize=(self.figsize_height, self.figsize_width), constrained_layout=True)
        for i in range(8):
            for j in range(4):
                channel = i * 4 + j
                axs1[i, j].plot(sorted_timestamps[channel], label='ch.' + str(channel))
                axs1[i, j].legend()
                axs1[i, j].set_ylabel('Timestamp (ns)')
                axs1[i, j].set_xlabel('Events')


    def show_summary(self):

        self.show_rates()
        self.show_areas()
        self.show_baselines()
        self.show_timestamps()
        self.show_diff_times()

    def help(self):
        print('Usage:')

        print('Step 1: Set target run')
        print("`man.run_dir_name = '/path/to/TEST000001_xxxxxxxxxxx/'`")
        print('Or, ')
        print("`man.find_latest_run()`")
        print('Or, interactive select')
        print("`man.select_run()`")
        print('then, select the index from list')
        print("`man.select_run(5)`")
        print('')

        print('Step 2: Merge subrun files')
        print("`man.add_all_subruns()`")
        print('Or, ')
        print("`man.add_subruns((0,1,2,3))`")
        print('')

        print('Step 3: Check and process')
        print("`man.data_name_list`")
        print("`man.process()`")
        print('')

        print('Step 4: Visualize')
        print('You can use the following functions:')
        print('show_counts(), show_rates(), show_pulse(), show_area(), show_areas()')
        print('show_baseline(), show_baselines(), show_baselines_bar(), show_baselines_rms_bar(),')
        print('show_timing(), show_timings()','show_diff_time(), show_diff_times(), show_timestamp(), show_timestamps()')
        print('')



if __name__ == '__main__':

    logging.basicConfig(level='DEBUG')
    man = manager()

    man.help()

    man.find_latest_run()
    man.add_all_subruns()

    man.process()
    man.show_rates()

