# Folder structure

# my_eeg_project/                
# └── EEG_analysis_replay/              # <-- Project root and contains everything (os.getcwd() points here)
#     ├── bids_output/           # <-- This is your BIDS root directory
#     │   ├── sub-001/
#     │   │   └── eeg/
#     │   │       ├── sub-001_task-thermalactive_eeg.vhdr
#     │   │       ├── sub-001_task-thermalactive_eeg.eeg
#     │   │       ├── sub-001_task-thermalactive_eeg.vmrk
#     │   │       ├── sub-001_task-thermalactive_channels.tsv
#     │   │       └── sub-001_task-thermalactive_events.tsv
#     │   └── sub-002/
#     │       └── eeg/
#     │           └── ...
#     │
#     ├── coll_lab_eeg_pipeline.py
#     ├── config.py
#     └── ... (other scripts, modules, etc.)


#by this point we should have all our files into bids format - if there are any problems with this formating it should be fixed with the to_bids folder
#We will need to epoch our data in this file, the following lines will go into details into what epochs strategy we will use
#we esssentially have 4 task - rest state, funcloc, learning, and cuedmental stim, we are intrested in replay activation only so a large portion of our recordings are not neccessary
#The rest state has a single trigger - fix after that we do not care much about epochs 
#during funcloc we will need to access brain activity linked with each images - this mean that our epochs should be fixed around the fix event (-x ms)
#learn prob has plenty of triggers, but we are only intrested in the post learning rest - we will need to split our data so that we get the 5 min rest state following the last trigger
#Cued stim we are intrested in replay after the image is presented - fix stim will again be our event for epochs




#find venv - cd C:\Users\labmp\Desktop\EEG_analysis_replay
# e:\EEG_analysis_replay

#Activate it - .\eeg_project_venv\Scripts\activate


#to run pipeline
#python -m coll_lab_eeg_pipeline -r C:\Users\labmp\Desktop\EEG_analysis_replay\replay_config.py

#python -m coll_lab_eeg_pipeline -r e:\EEG_analysis_replay\replay_config.py

#E:\EEG_analysis_replay
########################################################
# This is a custom config file for the labmp EEG pipeline.
###########################################################

# It includes the default config from mne-bids pipeline and some custom settings.

# Imports
from pathlib import Path
import os
from mne_bids import BIDSPath, get_entities_from_fname
import numpy as np

random_state = 42
# Global settings
# bids_root: Path to the BIDS dataset
#find path
project_root = os.getcwd()
bids_root = os.path.join(project_root, 'bids_output')

# external_root: Path to the external data folder (e.g. Schaefer atlas)
external_root = os.path.join(project_root, 'external')
# log_type: how to log the messages from the pipeline. Use 'console' to print everything to the console or 'file' to log all to a file in the preprocessed folder (recommended)
log_type = 'file'
# use_pyprep: Set to True to use pyprep for bad channels detection. If false, no automated bad channels dection
use_pyprep = True
# use_icalabel: Set to True to use mne-icalabel for ICA component classification. If false, the mne-bids-pipeline default classification based on eog and ecg is used
use_icalabel = True
# use_custom_tfr: Set to True to use the custom TFR function to compute time-frequency representations. If false, no TFRs are computed
custom_tfr = False
# compute_rest_features: Set to True to compute rest features after preprocessing. If false, no features are computed
compute_rest_features = False
# tasks_to_process: List of tasks to process.

tasks_to_process = ['cuedstim',
                    'postlearnrest',
                    'funcloc',
                    'reststate']
#config_validation: mne-bids-pipeline config validation. Leave to False because we use custom options.
config_validation = 'warn'
# subjects: List of subjects to process or 'all' to process all subjects.
subjects = 'all'
# sessions: List of sessions to process or leave empty to pcrocess all sessions.
sessions = []
#task: Task to process. This will be updated iteratively by the pipeline for each task. No need to change.
task = 'reststate'
# deriv_root: Path to the derivatives folder where the preprocessed data will be saved.
deriv_root = os.path.join(bids_root, 'derivatives', 'task_' + task, 'preprocessed')
# select_subjects: If True, only the subjects with a file for the current task will be processed. If False, pipeline will crash if missing task
select_subjects = True
if select_subjects:
    task_subs = list(set(str(f) for f in BIDSPath(root=bids_root, task=task, session=None, datatype='eeg', suffix='eeg', extension='vhdr').match()))
    task_subs = [get_entities_from_fname(f).get('subject') for f in task_subs]
    if subjects != 'all':
        # If subjects is not 'all', filter the task_subs list
        subjects = [sub for sub in task_subs if sub in subjects]
    else:
        # If subjects is 'all', use all available subjects
        subjects = task_subs

########################################################
# Options for pyprep bad channels detection
###########################################################
# pyprep_bids_path: Path to the BIDS dataset for pyprep, do not change unless you want a different path from the bids files
pyprep_bids_path = bids_root
# pyprep_pipeline_path: Path to the derivatives folder where the preprocessed data will be saved for pyprep, do not change
pyprep_pipeline_path = deriv_root
# pyprep_task: Task to process for pyprep, do not change
pyprep_task = task
# pyprep_ransac: Set to True to use RANSAC for bad channels detection, False to use only the other methods
pyprep_ransac = False
# pyprep_repeats: Number of repeats for the bad channel detection. This can improve detection by removing very bad channels and iterating again
pyprep_repeats = 3
# pyprep_average_reref: Set to True to average rereference the data before bad channels detection, False to use the original data
pyprep_average_reref = False
# pyprep_file_extension: File extension to use for the data files, default is .vhdr for BrainVision files
pyprep_file_extension = '.vhdr'
# pyprep_montage: Montage to use for the data, default is easycap-M1 for BrainVision files
pyprep_montage = 'easycap-M1'
# pyprep_l_pass: Low pass filter frequency for the data, default is 100.0 Hz
pyprep_l_pass = 100.0
# pyprep_notch: Notch filter frequency for the data, default is 60.0 Hz
pyprep_notch = 60.0
# pyprep_consider_previous_bads: Set to True to consider previous bad channels in the data (e.g. visually identified), False to ignore and clear them (e.g. when re-running the pipeline)
pyprep_consider_previous_bads = False
# pyprep_rename_anot_dict: Dictionary to rename the annotations to the format expected by MNE  (e.g. BAD_)
pyprep_rename_anot_dict = None
# pyprep_overwrite_chans_tsv: Set to True to overwrite the channels.tsv file with the bad channels detected by pyprep, False to keep the original file and create a second file. mne-bids-pipeline will only use original file so not recommended to set to False
pyprep_overwrite_chans_tsv = True
# pyprep_n_jobs: Number of jobs to use for pyprep, default is 1
pyprep_n_jobs = 3
# pyprep_subjects: List of subjects to process for pyprep, default is same as the rest of the pipeline
pyprep_subjects = subjects
# pyprep_delete_breaks: Set to True to delete breaks in the data (only for this operation, the data file is not modified), False to keep them
pyprep_delete_breaks = False
pyprep_breaks_min_length = 20  # Minimum length of breaks in seconds to consider them as breaks
pyprep_t_start_after_previous = 2  # Time in seconds to start after the last event
pyprep_t_stop_before_next = 2  # Time in seconds to stop before the next event
# pyprep_custom_bad_dict: Dictionary to specify custom bad channels for each subject. The format should be {taskname :{subject:[bad_chan_list]}} for example: {'eegtask': {'001': ['TP8']}} If not specified, the bad channels will only be detected automatically.
pyprep_custom_bad_dict = None



########################################################
# Options for Icalabel 
###########################################################
# icalabel_bids_path: Path to the BIDS dataset for icalabel, do not change unless you want a different path from the bids files
icalabel_bids_path = bids_root
# icalabel_pipeline_path: Path to the derivatives folder where the preprocessed data will be saved for icalabel, do not change
icalabel_pipeline_path = deriv_root
# icalabel_task: Task to process for icalabel, do not change
icalabel_task = task
# icalabel_prob_threshold: Probability threshold to use for icalabel, default is 0.8
icalabel_prob_threshold = 0.8
# icalabel_labels_to_keep: List of labels to keep for icalabel, default is ['brain', 'other']
icalabel_labels_to_keep = ['brain', 'other']
# icalabel_n_jobs: Number of jobs to use for icalabel, default is 1
icalabel_n_jobs = 5
# icalabel_subjects: List of subjects to process for icalabel, default is same as the rest of the pipeline
icalabel_subjects = subjects
# icalabel_keep_mnebids_bads: Set to True to keep the bad ica already flagged in the components.tsv file (e.g. visual inspection)
icalabel_keep_mnebids_bads = False




# --------------------------------------------------------------------
# GENERAL SETTINGS
# --------------------------------------------------------------------

# WHAT IT IS: Number of cores to use for parallel processing.
n_jobs = 5


# WHAT IT IS: The types of channels we want to process.
ch_types = ["eeg"]

#we need to specificy what channels are used to analyse eye blinks

eog_channels = ["HEOG", "VEOG"]


#we need to also set the reference 
#from what I can tell average referencing is norm
eeg_reference = "average"

eeg_template_montage = "easycap-M1"


#without this line the pipeline will crash because of bids mismatch naming, just let it be False unless another problem arises
process_rest = False

# --------------------------------------------------------------------
# FILTERING AND RESAMPLING
# --------------------------------------------------------------------

#Filetering is basically always necessary, we will set a high pass filter at 1hz and a low pass at 100

#here we have our high pass filter that should remove slow drift 
l_freq = 1.0

#here we have our low pass filter that should remove high-frequency noise 
h_freq = 40

#we can and also should always resample our data - we will do this at 500hz 
raw_resample_sfreq = 500


# --------------------------------------------------------------------
# EPOCHING 
# --------------------------------------------------------------------

#epoching is necessary for our analysis, without epochs we esssentially have continous data that will make no sens

# we will create a dictionnary with all the task and what triggers we use for epochs, we also need to specificy how long before and after the trigger each epoch is 


task_epoch_settings = {
    'funcloc': {
        'conditions': ['2'], #this should be the name of the trigger we want to epoch around - see problem in to bids, we will use number 5 = stim1
        'epochs_tmin': -0.5,  # 0.5 seconds before the trigger
        'epochs_tmax': 1.0,   # 1 seconds after the trigger
        'baseline': None
    },
    'cuedstim': {
        'conditions': ['2'], 
        'epochs_tmin': -0.5,
        'epochs_tmax': 1.0,
        'baseline': None
    },
    #this part is running on the rest state data, the postlearnrest should have been created in the to_bids_replay code - number 1 = fix
    'postlearnrest': {              
        'conditions': None,         
        'epochs_tmin': 0,           
        'epochs_tmax': 10.0,        
        'baseline': None            
    },
        'reststate': {
        'conditions': None, 
        'epochs_tmin': 0,
        'epochs_tmax': 10.0, # We will create 10-second long epochs
        'baseline': None
    }
}


#the conditions will be set from the dictionnary ^
conditions = task_epoch_settings[task]['conditions']
epochs_tmin = task_epoch_settings[task]['epochs_tmin']
epochs_tmax = task_epoch_settings[task]['epochs_tmax']
baseline = task_epoch_settings[task]['baseline']


#Determine what task are rest_state
if task in ['reststate', 'postlearnrest']:
    task_is_rest = True
    rest_epochs_duration = 10.0
    rest_epochs_overlap = 5.0 
else:
    task_is_rest = False

# --------------------------------------------------------------------
# ARTIFACT REJECTION
# --------------------------------------------------------------------



#we need to choose the method to remove artifacts, we will use ICA 
spatial_filter = "ica"

#ICA needs an algorithm we will use picard 

ica_algorithm = "picard"

#A high-pass filter applied just for the ICA calculation to improve its performance.
ica_l_freq = 1.0
#and we reject extremly noisy epochs before running ICA
ica_reject = {"eeg": 500e-6}

reject = "autoreject_local"

run_source_estimation = False



########################################################
# Rest features extraction config
###########################################################
# features_bids_path: Path to the BIDS dataset for features extraction, do not change unless you want a different path from the bids files
features_bids_path = bids_root
# features_out_path: Path to the derivatives folder where the preprocessed data will be saved for features extraction
features_out_path = deriv_root.replace('preprocessed', 'features')
# features_task: Task to process for features extraction, do not change
features_task = task
# features_sourcecoords_file: Path to the source coordinates file for features extraction, default is Schaefer 2018 atlas
features_sourcecoords_file = os.path.join(external_root, 'Schaefer2018_100Parcels_7Networks_order_FSLMNI152_1mm.Centroid_RAS.csv')
# features_freq_res: Frequency resolution for the PSD, default is 0.1 Hz
features_freq_res = 0.1
# features_freq_bands: Frequency bands to use for the PSD, default is theta, alpha, beta, gamma
features_freq_bands = {
    'theta': (4, 8 - features_freq_res),
    'alpha': (8, 13 - features_freq_res),
    'beta': (13, 30),
    'gamma': (30 + features_freq_res, 80),
}
# features_psd_freqmax: Maximum frequency for the PSD, default is 100 Hz
features_psd_freqmax = 100
# features_psd_freqmin: Minimum frequency for the PSD, default is 1 Hz
features_psd_freqmin = 1
# features_somato_chans: List of somatosensory channels to use for the features extraction, default is ['C3', 'C4', 'Cz']
features_somato_chans = ['C3', 'C4', 'Cz']
# features_subjects: List of specific subjects to compute features for, default is same as rest of pipeline)
features_subjects = subjects
# features_compute_sourcespace_features: Set to True to compute source space features, False to skip this step
features_compute_sourcespace_features = False
# features_n_jobs: Number of jobs to use for features extraction, default is 1
features_n_jobs = 1
# features_subjects: List of subjects to process for features extraction, default is same as the rest of the pipeline



########################################################
# custom tfr config
###########################################################
# custom_tfr_pipeline_path: Path to the preprocessed epochs, do not change unless you want a different path from the preprocessed epochs files
custom_tfr_pipeline_path = deriv_root
# custom_tfr_task: Task to process for TFR, do not change
custom_tfr_task = task
# custom_tfr_n_jobs: Number of jobs to use for TFR computation, default is 5
custom_tfr_n_jobs = 5
# custom_tfr_freqs: Frequencies to compute for TFR, default is np.arange(1, 100, 1)
custom_tfr_freqs = np.arange(1, 100, 1)
# custom_tfr_crop: Time interval to crop the TFR, default is None (no cropping)
custom_tfr_crop = None
# custom_tfr_n_cycles: Number of cycles for TFR computation, default is freqs/3.0
custom_tfr_n_cycles = custom_tfr_freqs / 3.0
# custom_tfr_decim: Decimation factor for TFR computation, default is 1
custom_tfr_decim = 2
# custom_tfr_return_itc: Whether to return inter-trial coherence, default is False
custom_tfr_return_itc = False
# custom_tfr_interpolate_bads: Whether to interpolate bad channels before computing the TFR, default is True
custom_tfr_interpolate_bads = True
# custom_tfr_average: Whether to average TFR across epochs, default is False
custom_tfr_average = False
# custom_tfr_return_average: Whether to return the average TFR in addition to the single trials, default is True
custom_tfr_return_average = True

