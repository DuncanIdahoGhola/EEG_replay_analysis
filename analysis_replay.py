import mne
import os 
import glob
from pathlib import Path
import json 
import pandas as pd
import numpy as np
from glob import glob
from mne_bids import BIDSPath
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
import joblib
import seaborn as sns
from mne.decoding import Vectorizer, SlidingEstimator, cross_val_multiscore

#create a sub list

sub = ['sub-027']


#to start analysing data we first need to get all paths to the fif data that came out of our pipeline
root = os.getcwd()
bids_der = os.path.join(root, 'bids_output', 'derivatives')
cued_stim_eeg = os.path.join(bids_der, 'task_cuedstim', 'preprocessed')
func_loc_eeg = os.path.join(bids_der, 'task_funcloc', 'preprocessed')
learn_prob_eeg = os.path.join(bids_der, 'task_postlearnrest', 'preprocessed')
rest_state_eeg = os.path.join(bids_der, 'task_reststate', 'preprocessed')
bids_root = os.path.join(root, 'bids_output')


#we will need to have access our meta data files so that we can train our classifier
behaviour_root = os.path.join(root, 'behaviour_data')

for sub in sub :
    #Find all sub files for each task - first is cued stim
    cuedstim_folder = os.path.join(cued_stim_eeg, sub, 'eeg',)
    pattern = f'{sub}_task-cuedstim_proc-clean_epo.fif'
    matching_files_cuedstim = glob(os.path.join(cuedstim_folder, pattern))
    assert len(matching_files_cuedstim) == 1
    #we can then do func_loc
    funcloc_folder = os.path.join(func_loc_eeg, sub, 'eeg',)
    pattern = f'{sub}_task-funcloc_proc-clean_epo.fif'
    matching_files_funcloc = glob(os.path.join(funcloc_folder, pattern))
    assert len(matching_files_funcloc) == 1
    #learn prob rest state
    learnprob_folder = os.path.join(learn_prob_eeg, sub, 'eeg',)
    pattern = f'{sub}_task-postlearnrest_proc-clean_epo.fif'
    matching_files_learnprob = glob(os.path.join(learnprob_folder, pattern))
    assert len(matching_files_learnprob) == 1
    #rest state
    reststate_folder = os.path.join(rest_state_eeg, sub, 'eeg',)
    pattern = f'{sub}_task-reststate_proc-clean_epo.fif'
    matching_files_reststate = glob(os.path.join(reststate_folder, pattern))
    assert len(matching_files_reststate) == 1


    #this is an example on how to plot epochs and psd
    epochs_func_loc = mne.read_epochs(matching_files_funcloc[0])
    epochs_func_loc.plot(n_epochs=1, title=f"{sub} - FuncLoc")
    epochs_func_loc.plot_psd(fmin=1, fmax=40, average=True)
    #how to get meta data
    print(epochs_func_loc.metadata)
    print(epochs_func_loc.metadata.columns)


    #we will start by reattaching meta data to the func loc and cued stim - the pipeline drops the meta data only keeping events
    #Lets first find the tsv files for both funcloc and cued stim
    sub_fodler = os.path.join(bids_root, sub, 'eeg')
    pattern_1 = f'{sub}_task-funcloc_events.tsv'
    pattern_2 = f'{sub}_task-cuedstim_events.tsv'
    matching_meta_funcloc = glob(os.path.join(sub_fodler, pattern_1))
    matching_meta_cuedstim = glob(os.path.join(sub_fodler, pattern_2))
    assert len(matching_files_funcloc) == 1
    assert len(matching_files_cuedstim) == 1   
    #read the meta data as csv - into dataframe 
    meta_data_func_loc = pd.read_csv(matching_meta_funcloc[0], sep='\t')
    meta_data_cue_stim = pd.read_csv(matching_meta_cuedstim[0], sep='\t')

    #FUNCLOC
    #Start by loading the cleand epochs without meta data
    epochs_func_loc = mne.read_epochs(matching_files_funcloc[0], preload=True)
    #we can find what epochs were kept after ICA using .selection
    selection_indices = epochs_func_loc.selection
    #we can then clean our meta data using this selection_indices
    cleand_metadata_func = meta_data_func_loc.iloc[selection_indices]
    cleand_metadata_func = cleand_metadata_func.reset_index(drop=True)
    epochs_func_loc.metadata = cleand_metadata_func

    #CUEDSTIM
    #We will load the meta data, clean it and then append it to the cued_stim_epochs
    epochs_cued_stim = mne.read_epochs(matching_files_cuedstim[0], preload=True)
    selection_indices = epochs_cued_stim.selection
    cleand_metadata_cue = meta_data_cue_stim.iloc[selection_indices]
    cleand_metadata_cue = cleand_metadata_cue.reset_index(drop=True)
    epochs_cued_stim.metadata = cleand_metadata_cue

    #Our meta data is now cleaned and fixed to our epochs for both func loc and cued_stim

    #MACHINE LEARNING TIME - :) 
    #We want to train a classifier on the functional localizer data
    #we want to predict what image was show to the participant based on the eeg data
    
    #we will define our featuers (x) and our target (y)
    #X = eeg data from all channels and time points in epoch
    #y = the label of the image 

    X = epochs_func_loc.get_data(picks='eeg')
    y = epochs_func_loc.metadata['image_file']

    # Create a pipeline with a vectorizer, a scaler, and a logistic regression classifier
    # Vectorizer: Flattens the 3D data (epochs, channels, time) into 2D (epochs, features)
    # StandardScaler: Standardizes features (important for linear models)
    # LogisticRegression: The classifier
    pipeline = make_pipeline(
        Vectorizer(),
        StandardScaler(),
        LogisticRegression(solver='liblinear', random_state=42, max_iter=1000)
    )
   
    # Define the cross-validation strategy
    # StratifiedKFold is excellent here as it preserves the percentage of samples for each class.
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
     # Perform cross-validation and get accuracy scores for each fold
    scores_whole_epoch = cross_val_score(pipeline, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
    print(f"Mean Whole-Epoch Decoding Accuracy: {scores_whole_epoch.mean():.3f} ± {scores_whole_epoch.std():.3f}")

    # Plot the results as a box plot
    chance_level = 1 / len(np.unique(y))
    plt.figure(figsize=(6, 5))
    sns.boxplot(y=scores_whole_epoch, color='skyblue')
    sns.stripplot(y=scores_whole_epoch, color='black', alpha=0.7) # Show individual points
    plt.axhline(chance_level, color='red', linestyle='--', label=f'Chance Accuracy ({chance_level:.2f})')
    plt.title(f'{sub} - Whole-Epoch Decoding Accuracy')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1])
    plt.legend()
    plt.show()


    #time resolved decoding 
    # Define the classifier pipeline (no Vectorizer needed here, as SlidingEstimator handles the time dimension)
    clf = make_pipeline(StandardScaler(), 
                        LogisticRegression(solver="liblinear", random_state=42))
    
    # Create the SlidingEstimator instance. This object will slide a classifier along the time axis.
    # We use n_jobs=-1 to use all available CPU cores, which significantly speeds up the process.
    time_decoder = SlidingEstimator(clf, n_jobs=-1, scoring="accuracy", verbose=True)   

    # Run the cross-validation. This will be done for each time point.
    # The output 'scores_time' will be an array of shape (n_splits, n_timepoints).
    scores_time = cross_val_multiscore(time_decoder, X, y, cv=cv, n_jobs=-1)
    # Average the scores across the cross-validation folds
    mean_scores_time = scores_time.mean(axis=0)
    std_scores_time = scores_time.std(axis=0)

    print(f"Sliding window analysis complete. Peak accuracy: {mean_scores_time.max():.3f}")
    # Plot the sliding window decoding scores
    plt.figure(figsize=(12, 6))
    plt.plot(epochs_func_loc.times, mean_scores_time, label='Mean Decoding Accuracy')
    plt.axhline(chance_level, color='red', linestyle='--', label='Chance Accuracy')
    plt.axvline(0, color='black', linestyle='-.', label='Stimulus Onset (t=0)')
    
    # Add a shaded area to represent the standard deviation
    plt.fill_between(epochs_func_loc.times, 
                     mean_scores_time - std_scores_time, 
                     mean_scores_time + std_scores_time, 
                     alpha=0.2, color='blue', label='±1 Standard Deviation')
                     
    plt.title(f'{sub} - Time-Resolved Decoding Accuracy')
    plt.xlabel('Time (s)')
    plt.ylabel('Classifier Accuracy')
    plt.legend(loc='upper left')
    plt.grid(True, linestyle=':')
    plt.show()





    
    











