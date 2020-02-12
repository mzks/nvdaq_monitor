import os, sys
import pkg_resources
import logging
import numpy as np
from tqdm import tqdm
import glob
import strax
import blosc
import matplotlib.pyplot as plt
import seaborn as sns


class manager:
    """
    """

    def __init__(self):

        self.logger = logging.getLogger(__name__)
        self.logger.info('Monitor ver. 0.1.0 Trial 1')

        # Parameters
        # self.__data_path = pkg_resources.resource_filename('nvdaq_monitor', 'data/')
        # self.data_name = self.__data_path + 'xenondaq_reader_0_140443912218368' # for test
        self.data_name_list = []


    #def find_latest_file(self, dir_name=None):
    #    if dir_name == None:
    #        dir_name = self.__data_path

    #    list_of_files = glob.glob(dir_name+'*')
    #    latest_file = max(list_of_files, key=os.path.getctime)
    #    self.data_name = latest_file

    def add_subrun_file(self, file_path):

        self.data_name_list.append(file_path)


    def add_subrun_files(self, subrun_path='/Users/mzks/xenon/daq_test/data/TEST000003_02102020122501/000023/'):

        file_list = glob.glob(subrun_path + '*')
        self.data_name_list.extend([file for file in file_list if os.stat(file).st_size != 0])



    def load_data(self, data_name):

        #print('Target file: ', data_name)
        try:
            self.file = open(data_name, 'rb')
            self.data = blosc.decompress(self.file.read())
            self.darr = np.frombuffer(self.data, dtype=strax.record_dtype())
        except:
            self.logger.warning('Skipped '+ data_name)
            return False

        return True

    def count_record(self, event):
        return sum([len(event) for event in event])


    def process(self):

        # Prepare
        self.init_bin_baseline = 10
        self.num_of_channel = 16

        self.calced_baselines = [[] for i in range(self.num_of_channel)]
        self.calced_areas = [[] for i in range(self.num_of_channel)]
        self.timestamps= [[] for i in range(self.num_of_channel)]
        self.peak_timings = [[] for i in range(self.num_of_channel)]
        self.waveforms= [[] for i in range(self.num_of_channel)]

        # Loop
        for data_name in tqdm(self.data_name_list):
            if not self.load_data(data_name): continue
            event = []

            for record in self.darr:
                if not event:  # First record in the event
                    event_timestamp = record['time']
                event.append(record['data'])

                if self.count_record(event)<record['pulse_length']:
                    #print('len', count_record(event), 'pulse-l', record['pulse_length'])
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
        start_time = [(np.min(time) if (time != []) else None) for time in self.timestamps]
        stop_time = [(np.max(time) if (time != []) else None) for time in self.timestamps]
        live_time = [stop - start if (stop and start) else None for stop, start in zip(stop_time, start_time)]
        rates = [event / live * 1.e9 if (event and live) else 0 for event, live in zip(event_numbers, live_time)]

        plt.bar(np.arange(0, self.num_of_channel), rates)
        plt.xlabel('Channel')
        plt.ylabel('Rate (Hz)')


    def show_pulse(self, channel=0, event=0):
        plt.plot(np.arange(len(self.waveforms[channel][event])), self.waveforms[channel][event])
        plt.xlabel('Sample')
        plt.ylabel('ADC Value')


    def show_area(self, channel=0, hist_range=None, bins=None):
        plt.hist(self.calced_areas[channel], lw=0, range=hist_range, bins=bins)
        plt.xlabel('ADC Integration')
        plt.ylabel('Counts')


    def show_areas(self, hist_range=None, bins=None):

        fig1, axs1 = plt.subplots(4, 4, figsize=(16,10), constrained_layout=True)
        for i in range(4):
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


    def show_baselines(self, hist_range=None, bins=None):

        fig1, axs1 = plt.subplots(4, 4, figsize=(16,10), constrained_layout=True)
        for i in range(4):
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

        fig1, axs1 = plt.subplots(4, 4, figsize=(16,10), constrained_layout=True)
        for i in range(4):
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
        fig1, axs1 = plt.subplots(4, 4, figsize=(16,10), constrained_layout=True)
        for i in range(4):
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
        fig1, axs1 = plt.subplots(4, 4, figsize=(16, 10), constrained_layout=True)
        for i in range(4):
            for j in range(4):
                channel = i * 4 + j
                axs1[i, j].plot(sorted_timestamps[channel], label='ch.' + str(channel))
                axs1[i, j].legend()
                axs1[i, j].set_ylabel('Timestamp (ns)')
                axs1[i, j].set_xlabel('Events')


if __name__ == '__main__':

    logging.basicConfig(level='DEBUG')

    man = manager()

    man.add_subrun_files('/Users/mzks/xenon/daq_test/data/TEST000003_02102020122501/000023/')
    man.process()
    man.show_rates()
    man.show_diff_time(0)
