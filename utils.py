import os
import numpy as np
from tqdm import tqdm
from scipy.io import loadmat
from itertools import compress
import glob
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def load_dataset_UcSanDiego(base_project_dir = '.'):
    print('Loading \'Preprocessed Uc San Diego\' ...')
    dataset_path = os.path.join(base_project_dir, "Data", 'Preprocessed Uc San Diego')
    HC_data_path = os.path.join(dataset_path, "HC")
    PD_OFF_data_path = os.path.join(dataset_path, "PD_OFF")
    PD_ON_data_path = os.path.join(dataset_path, "PD_ON")

    HC_data = np.array([loadmat(os.path.join(HC_data_path, file_name)).get('time') for file_name in tqdm(os.listdir(HC_data_path))]) 

    PD_OFF_data = np.array([loadmat(os.path.join(PD_OFF_data_path, file_name)).get('time') for file_name in tqdm(os.listdir(PD_OFF_data_path))])
    PD_ON_data = np.array([loadmat(os.path.join(PD_ON_data_path, file_name)).get('time') for file_name in tqdm(os.listdir(PD_ON_data_path))])

    return HC_data, PD_OFF_data, PD_ON_data



def load_dataset_PRED_CT(base_project_dir = '.'):
    print('Loading \'Preprocessed PRED-CT\' ...')
    dataset_path = os.path.join(base_project_dir, "Data", 'Preprocessed PRED-CT')

    PD_OFF_data_path = os.path.join(dataset_path, "PD_OFF")
    PD_ON_data_path = os.path.join(dataset_path, "PD_ON")

    PD_OFF_data = np.array([loadmat(os.path.join(PD_OFF_data_path, file_name)).get('time') for file_name in tqdm(os.listdir(PD_OFF_data_path))])
    PD_ON_data = np.array([loadmat(os.path.join(PD_ON_data_path, file_name)).get('time') for file_name in tqdm(os.listdir(PD_ON_data_path))])

    return PD_OFF_data, PD_ON_data

def load_dataset_UI(base_project_dir = '.'):
    print('Loading \'Preprocessed UI\' ...')
    dataset_path = os.path.join(base_project_dir, "Data", 'Preprocessed UI')

    HC_data_path = os.path.join(dataset_path, "Control")
    PD_data_path = os.path.join(dataset_path, "PD")
    
    HC_data = np.array([loadmat(os.path.join(HC_data_path, file_name)).get('time')[:63] for file_name in tqdm(os.listdir(HC_data_path))]) 
    PD_data = np.array([loadmat(os.path.join(PD_data_path, file_name)).get('time')[:63] for file_name in tqdm(os.listdir(PD_data_path))])

    HC_labels = np.zeros((HC_data.shape[0], 1))
    PD_labels = np.ones((PD_data.shape[0], 1))

    return HC_data, HC_labels, PD_data, PD_labels


class Dataset10FoldSpliter():
    def __init__(self, X, Y, shuffle=False, seed=0):
        self.X = X
        self.Y = Y
        self.shuffle = shuffle
        self.seed = seed

        batch_size = X.shape[0]
        fold_size = batch_size // 10
        #creating fold numbers for each ECG
        self.fold_numbers = np.arange(1, 11)
        self.fold_numbers = np.repeat(self.fold_numbers, fold_size)
        for _ in range(batch_size - fold_size*10):
            self.fold_numbers = np.append(self.fold_numbers, 10) 

        self.test_trackers = np.arange(batch_size)
        #print(self.test_trackers)
        #exit()
        if shuffle:
            print(f'Shuffling dataset with seed: {seed}.')
            np.random.seed(seed)
            permutation = np.random.permutation(batch_size)
            self.X = self.X[permutation]
            self.Y = self.Y[permutation].reshape(-1, 1)
            self.test_trackers = self.test_trackers[permutation].reshape(-1, 1)
            
        self.val_fold = 1
        self.test_fold = 2
    
    def split(self):
        not_val_idxs = self.fold_numbers != self.val_fold
        not_test_idxs = self.fold_numbers != self.test_fold
        train_idxs = not_val_idxs * not_test_idxs

        val_idxs = self.fold_numbers == self.val_fold
        test_idxs = self.fold_numbers == self.test_fold

        X_train = self.X[train_idxs]
        Y_train = self.Y[train_idxs].reshape(-1, 1)
        X_val = self.X[val_idxs]
        Y_val = self.Y[val_idxs].reshape(-1, 1)
        X_test = self.X[test_idxs]
        Y_test = self.Y[test_idxs].reshape(-1, 1)
        test_fold_tracker = self.test_trackers[test_idxs].reshape(-1, 1)
        
        self.val_fold += 1
        self.test_fold = self.test_fold+1 if self.val_fold != 10 else 1

        return X_train, Y_train, X_val, Y_val, X_test, Y_test, test_fold_tracker
        




def get_UcSanDiego_subject_names(base_project_dir):
    dataset_path = os.path.join(base_project_dir, "Data", 'Preprocessed Uc San Diego')
    HC_data_path = os.path.join(dataset_path, "HC")
    PD_OFF_data_path = os.path.join(dataset_path, "PD_OFF")
    
    get_subject_name = lambda file_name:file_name.split('_')[0]
    HC_names = list(set(map(get_subject_name, os.listdir(HC_data_path))))
    PD_OFF_names = list(set(map(get_subject_name, os.listdir(PD_OFF_data_path))))

    return HC_names, PD_OFF_names



class SubjectNames10FoldSpliter():
    def __init__(self, subject_names, shuffle=False, seed=0):
        self.subject_names = subject_names

        batch_size = len(subject_names)
        fold_size = batch_size // 10
        #creating fold numbers for each subject names
        self.fold_numbers = np.arange(1, 11)
        self.fold_numbers = np.repeat(self.fold_numbers, fold_size)
        for _ in range(batch_size - fold_size*10):
            self.fold_numbers = np.append(self.fold_numbers, 10) 

        self.test_trackers = np.arange(batch_size)

        if shuffle:
            print(f'Shuffling dataset with seed: {seed}.')
            np.random.seed(seed)
            permutation = np.random.permutation(batch_size)
            self.subject_names = [self.subject_names[i] for i in permutation]
            self.test_trackers = self.test_trackers[permutation].reshape(-1, 1)
            
        self.val_fold = 1
        self.test_fold = 2
    
    def split(self):
        not_val_idxs = self.fold_numbers != self.val_fold
        not_test_idxs = self.fold_numbers != self.test_fold
        train_idxs = not_val_idxs * not_test_idxs

        val_idxs = self.fold_numbers == self.val_fold
        test_idxs = self.fold_numbers == self.test_fold

        subject_names_train = list(compress(self.subject_names, train_idxs))
        subject_names_val = list(compress(self.subject_names, val_idxs))
        subject_names_test = list(compress(self.subject_names, test_idxs))
        test_fold_tracker = self.test_trackers[test_idxs].reshape(-1, 1)
        
        self.val_fold += 1
        self.test_fold = self.test_fold+1 if self.val_fold != 10 else 1

        return subject_names_train, subject_names_val, subject_names_test, test_fold_tracker


def load_UcSanDiego_based_on_subject_names(subject_names, base_project_dir):
    dataset_path = os.path.join(base_project_dir, "Data", 'Preprocessed Uc San Diego')
    HC_data_path = os.path.join(dataset_path, "HC")
    PD_OFF_data_path = os.path.join(dataset_path, "PD_OFF")
    data = []
    labels = []
    for subject_name in tqdm(subject_names):
        if subject_name[4:6] == 'pd':
            name_format = os.path.join(PD_OFF_data_path, subject_name + '*')
            PD_OFF_files = glob.glob(name_format)
            data.append(np.array([loadmat(file).get('time') for file in PD_OFF_files]))
            labels.append(np.ones((len(PD_OFF_files), 1)))
        elif subject_name[4:6] == 'hc':
            name_format = os.path.join(HC_data_path, subject_name + '*')
            HC_files = glob.glob(name_format)
            data.append(np.array([loadmat(file).get('time') for file in HC_files]))
            labels.append(np.zeros((len(HC_files), 1)))
    
    # stacking data and labels of all subjects
    stacked_data = data[0]
    stacked_labels = labels[0]
    for i in range(1, len(data)):
        stacked_data = np.vstack((stacked_data, data[i]))
        stacked_labels = np.vstack((stacked_labels, labels[i]))

    return stacked_data, stacked_labels






def plot_32_channel_eeg_with_background_attention(eeg_data, attention_weights, channel_names, sampling_rate=256, label='', alpha=0.12, squeez_factore=1, save_path="eeg_background_attention_plot.png"):
    """
    Plots a 32-channel EEG signal with black signals and a background that transitions from white (low attention) to red (high attention).
    Saves the figure as a full-sized image.

    - eeg_data: numpy array of shape (32, time_points), representing EEG signals from 32 channels.
    - attention_weights: numpy array of shape (time_points,), representing attention assigned to each time point.
    - channel_names: list of strings, names of the EEG channels.
    - sampling_rate: Sampling rate of EEG signal (default is 50 Hz).
    - save_path: Path to save the figure.
    """

    eeg_data = eeg_data.T  # Ensure EEG data is in the correct shape (time, channels)
    num_channels, num_timepoints = eeg_data.shape
    time_axis = np.arange(num_timepoints) / sampling_rate

    # Normalize attention weights for background shading
    norm_attention = (attention_weights - np.min(attention_weights)) / (np.max(attention_weights) - np.min(attention_weights))

    # Define a custom colormap from white (low attention) to red (high attention)
    custom_cmap = mcolors.LinearSegmentedColormap.from_list("custom_background", ["white", "xkcd:brick red"])
    norm = mcolors.Normalize(vmin=0, vmax=1)

    fig, ax = plt.subplots(figsize=(16, 8))

    # Fill background color based on attention weights
    for i in range(num_timepoints - 1):
        ax.axvspan(time_axis[i], time_axis[i+1], color=custom_cmap(norm_attention[i]), alpha=alpha)

    # Plot each channel in black over the background
    for ch in range(num_channels):
        ax.plot(time_axis, eeg_data[ch] + ch * squeez_factore, color='black', linewidth=2.0)

    # Set labels
    ax.set_yticks(np.arange(num_channels) * squeez_factore)
    ax.set_yticklabels(channel_names, fontsize=12, fontweight='bold')
    ax.set_xlabel("Time (seconds)", fontsize=20)
    ax.set_title(f"32-Channel EEG with Background Attention - {label}", fontsize=24, fontweight='bold')

    ax.tick_params(axis='x', labelsize=18)

    # Add a colorbar to indicate attention intensity
    sm = plt.cm.ScalarMappable(cmap=custom_cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("Attention Weight Intensity", fontsize=18)
    cbar.ax.tick_params(labelsize=16)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')  # Save figure instead of showing
    plt.close()
    print(f"Saved EEG attention plot with background color-coded attention to {save_path}")



def plot_32_channel_eeg_with_custom_colormap(eeg_data, attention_weights, channel_names, sampling_rate=256, label='', squeez_factore=1, save_path="eeg_custom_attention_plot.png"):
    """
    Plots a 32-channel EEG signal with additive attention weights using a custom colormap.
    Black = Low Attention, Dark Red = High Attention.
    Saves the figure as a full-sized image.

    - eeg_data: numpy array of shape (32, time_points), representing EEG signals from 32 channels.
    - attention_weights: numpy array of shape (time_points,), representing attention assigned to each time point.
    - channel_names: list of strings, names of the EEG channels.
    - sampling_rate: Sampling rate of EEG signal (default is 50 Hz).
    - save_path: Path to save the figure.
    """

    eeg_data = eeg_data.T  # Ensure EEG data is in the correct shape (time, channels)
    num_channels, num_timepoints = eeg_data.shape
    time_axis = np.arange(num_timepoints) / sampling_rate

    # Normalize attention weights for color mapping (scale between 0 and 1)
    norm_attention = (attention_weights - np.min(attention_weights)) / (np.max(attention_weights) - np.min(attention_weights))

    # Define a custom colormap from black (low attention) to dark red (high attention)
    custom_cmap = mcolors.LinearSegmentedColormap.from_list("custom_attention", ["black", "red"])
    norm = mcolors.Normalize(vmin=0, vmax=1)

    fig, ax = plt.subplots(figsize=(16, 8))  # Increase figure size for better visibility

    # Plot each channel with color-coded attention weights
    for ch in range(num_channels):
        for i in range(1, num_timepoints):
            color = custom_cmap(norm(norm_attention[i]))  # Get color from custom colormap
            ax.plot(time_axis[i-1:i+1], eeg_data[ch, i-1:i+1] + ch * squeez_factore, color=color, linewidth=2.0)  # Adjust height for spacing

   # Set labels
    ax.set_yticks(np.arange(num_channels) * squeez_factore)
    ax.set_yticklabels(channel_names, fontsize=12, fontweight='bold')
    ax.set_xlabel("Time (seconds)", fontsize=20)
    ax.set_title(f"32-Channel EEG with Background Attention - {label}", fontsize=24, fontweight='bold')

    ax.tick_params(axis='x', labelsize=18)  # Adjust the font size for x-ticks

    # Add a colorbar to indicate attention intensity
    sm = plt.cm.ScalarMappable(cmap=custom_cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("Attention Weight Intensity", fontsize=18)
    cbar.ax.tick_params(labelsize=16)  # Change 16 to desired font size

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')  # Save figure instead of showing
    plt.close()
    print(f"Saved EEG attention plot with background color-coded attention to {save_path}")

