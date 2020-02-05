import os, sys
import pkg_resources
import logging
import numpy as np
from tqdm import tqdm
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
        self.__data_path = pkg_resources.resource_filename('nvdaq_monitor', 'data/')
        self.data_name = self.__data_path + 'xenondaq_reader_0_140443912218368' # for test


    def load_data(self):

        self.file = open(self.data_name, 'rb')
        self.data = blosc.decompress(self.file.read())
        self.darr = np.frombuffer(self.data, dtype=strax.record_dtype())


    def process(self):


        self.init_bin_baseline = 10
        self.num_of_channel = 16

        self.calced_baselines = [[] for i in range(self.num_of_channel)]
        self.calced_areas = [[] for i in range(self.num_of_channel)]
        self.timestamps= [[] for i in range(self.num_of_channel)]
        self.peak_timings = [[] for i in range(self.num_of_channel)]
        self.waveforms= [[] for i in range(self.num_of_channel)]

        #print('Total events: ', len(self.darr))
        self.init_timestamp = None
        i_loop = 0
        event = []

        def count_record(event):
            return sum([len(event) for event in event])

        for record in tqdm(self.darr):

            if self.init_timestamp is None:
                self.init_timestamp = record['time']

            if not event:
                event_timestamp = record['time']

            event.append(record['data'])

            if count_record(event)<record['pulse_length']:
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



                self.final_timestamp = record['time']
                event = []

                i_loop += 1
                if False & i_loop > 10:
                    break

        return True

    def show_rates(self):
        event_numbers = [len(area) for area in self.calced_areas]  ## Event numbers
        livetime = (self.final_timestamp - self.init_timestamp) * 1.e-9
        rates = event_numbers / livetime  # (Hz)
        plt.bar(np.arange(0, self.num_of_channel), rates)
        plt.xlabel('Channel')
        plt.ylabel('Rate (Hz)')


    def show_pulse(self, channel=0, event=0):
        plt.plot(np.arange(len(self.waveforms[channel][event])), self.waveforms[channel][event])
        plt.xlabel('Sample')
        plt.ylabel('ADC Value')


    def show_area(self, channel=0):
        plt.hist(self.calced_areas[channel], lw=0, bins=100)
        plt.xlabel('ADC Integration')
        plt.ylabel('Counts')


    def show_areas(self):

        fig1, axs1 = plt.subplots(4, 4, figsize=(16,10), constrained_layout=True)
        for i in range(4):
            for j in range(4):
                channel = i*4 + j
                axs1[i,j].hist(self.calced_areas[channel], lw=0, bins=100, label='ch.'+str(channel))
                axs1[i,j].legend()
                axs1[i,j].set_xlabel('ADC Integration')
                axs1[i,j].set_ylabel('Counts')


    def show_baseline(self, channel=0):
        plt.hist(self.calced_baselines[channel], lw=0, bins=100)
        plt.xlabel('ADC Value')
        plt.ylabel('Counts')


    def show_baselines(self):

        fig1, axs1 = plt.subplots(4, 4, figsize=(16,10), constrained_layout=True)
        for i in range(4):
            for j in range(4):
                channel = i*4 + j
                axs1[i,j].hist(self.calced_baselines[channel], lw=0, bins=100, label='ch.'+str(channel))
                axs1[i,j].legend()
                axs1[i,j].set_xlabel('ADC Value')
                axs1[i,j].set_ylabel('Counts')


    def show_timing(self, channel=0):
        plt.hist(self.peak_timings[channel], lw=0, bins=100)
        plt.xlabel('Peak timing (index)')
        plt.ylabel('Counts')


    def show_timings(self):

        fig1, axs1 = plt.subplots(4, 4, figsize=(16,10), constrained_layout=True)
        for i in range(4):
            for j in range(4):
                channel = i*4 + j
                axs1[i,j].hist(self.peak_timings[channel], lw=0, bins=100, label='ch.'+str(channel))
                axs1[i,j].legend()
                axs1[i,j].set_xlabel('Peak timing (index)')
                axs1[i,j].set_ylabel('Counts')



if __name__ == '__main__':

    logging.basicConfig(level='DEBUG')

    man = manager()

    man.load_data()
    man.process()
    man.show_rates()
