
import pyedflib
import numpy as np
from scipy import signal
import scipy.stats as stats
import scipy.io
import os
import glob

base_project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__)))

save_dir_time = os.path.join(base_project_dir, 'Data', 'time')

if not os.path.exists(save_dir_time):
    os.mkdir(save_dir_time)

root_dir = os.path.join(base_project_dir, 'Data', 'PRED-CT', 'PD REST', 'ON')
Channel_indexs = np.array (())


def buffer(data, duration, dataOverlap):
	numberOfSegments = int(np.floor((data.shape[1]-dataOverlap)/(duration-dataOverlap)))
	tempBuf = np.zeros((numberOfSegments,data.shape[0], duration))
	#print(data.shape)
	for j in range(numberOfSegments):
		for i in range(0,data.shape[1]-duration,(duration-int(dataOverlap))):
			tempBuf[j,:,:] = data[:,i:i+duration]
	return tempBuf



def signal_segmentation_time(root_dir, Channel_indexs, fs_base = 500, fs_new = 256, order = 5, lowfreq = 0.5, highfreq = 64, time_segment = 2, dataoverlap = 0):
	'''Read data'''
	#signals_root_dir = glob.glob(root_dir + str("/*.bdf"))
	signals_root_dir = glob.glob(root_dir + str("/*.mat"))

	for index in range(len(signals_root_dir)):

		directory = signals_root_dir[index][len(root_dir)+1 : -4]
		print(directory)

		file_name = signals_root_dir[index]

		'''f = pyedflib.EdfReader(file_name)
		n = f.signals_in_file
		signal_labels = f.getSignalLabels()
		sigbufs = np.zeros((n, f.getNSamples()[0]))
		for i in np.arange(n):
			sigbufs[i, :] = f.readSignal(i)'''
		
		sigbufs = scipy.io.loadmat(file_name)
		sigbufs = sigbufs.get('EEG')[0, 0]['data']

		'''Channel selection'''
		if Channel_indexs.any() != False:
			channel_index = np.array(Channel_indexs)
			sigbufs_selected = np.zeros((channel_index.shape[0], f.getNSamples()[0]))
			j = 0
			for i in channel_index:
				sigbufs_selected[j, :] = sigbufs[i, :]
				j = j+1
		else:
			sigbufs_selected = sigbufs



		'''resampling & filtering'''

		new_length  = np.int32(np.floor(sigbufs_selected.shape[1] *(fs_new/fs_base)))
		sigbufs_s_resampled = np.zeros((sigbufs_selected.shape[0], new_length))
		b, a = signal.butter(order, [lowfreq,highfreq] , btype='band', fs = fs_new)

		for i in range(sigbufs_selected.shape[0]):
			temp = signal.resample(sigbufs_selected [i,:], new_length)
			sigbufs_s_resampled [i,:] = signal.filtfilt(b,a, temp)

		'''Segmentation'''
		duration = np.int32 (fs_new * time_segment)
		

		bufOut=buffer(sigbufs_s_resampled, duration, dataoverlap)
		for j in range(bufOut.shape[0]):

			Normalized_segment =  stats.zscore(bufOut[j])
			save_path_segments = os.path.join(save_dir_time, directory + "_seg_" + str (j) + str(".mat"))
			scipy.io.savemat(save_path_segments, {"time": Normalized_segment})


if __name__ == "__main__":

	signal_segmentation_time(root_dir, Channel_indexs)