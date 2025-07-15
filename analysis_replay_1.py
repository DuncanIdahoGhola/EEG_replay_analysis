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
from sklearn.model_selection import cross_val_predict
import joblib
import seaborn as sns
from mne.decoding import Vectorizer, SlidingEstimator, cross_val_multiscore
from scipy import stats 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix



#create a sub list

sub = ['sub-027', 'sub-099']
sub_permutation_done= ['sub-027'] #this list is used to check if the permutation test has been done for the subject

#to start analysing data we first need to get all paths to the fif data that came out of our pipeline
root = os.getcwd()
bids_der = os.path.join(root, 'bids_output', 'derivatives')
cued_stim_eeg = os.path.join(bids_der, 'task_cuedstim', 'preprocessed')
func_loc_eeg = os.path.join(bids_der, 'task_funcloc', 'preprocessed')
learn_prob_eeg = os.path.join(bids_der, 'task_postlearnrest', 'preprocessed')
rest_state_eeg = os.path.join(bids_der, 'task_reststate', 'preprocessed')
bids_root = os.path.join(root, 'bids_output')
deriv_root = os.path.join(root, 'analyse_deriv')

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

    #create a sub folder in the deriv root
    sub_folder = os.path.join(deriv_root, sub)
    os.makedirs(sub_folder, exist_ok=True)


    #this is an example on how to plot epochs and psd
    epochs_func_loc = mne.read_epochs(matching_files_funcloc[0])
    epochs_func_loc.plot(n_epochs=1, title=f"{sub} - FuncLoc")
    epochs_func_loc.plot_psd(fmin=1, fmax=40, average=True)
    #how to get meta data
    print(epochs_func_loc.metadata)
    print(epochs_func_loc.metadata.columns)

    #######################################
    # META DATA REATTACHMENT AND CLEANING #
    #######################################
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
    
    #only keep meta data rows with event_type == 'stime1' for func loc
    meta_data_func_loc = meta_data_func_loc[meta_data_func_loc['event_type'] == 'stim1']

    #clean the meta data to keep the classifier only the relevant columns
    #we will keep the image_file, the event_type, presented_word, is_match, columns
    meta_data_func_loc = meta_data_func_loc[['image_file', 'event_type', 'presented_word', 'is_match']]
    #reset index
    meta_data_func_loc = meta_data_func_loc.reset_index(drop=True)


    #only keep meta data rows with event_type == 'fix' for cued stim
    meta_data_cue_stim = meta_data_cue_stim[meta_data_cue_stim['event_type'] == 'fix']
    #clean meta data, keep cue_direction, cue_text, probe_image_file,
    meta_data_cue_stim = meta_data_cue_stim[['cue_direction', 'cue_text', 'probe_image_file']]
    #reset index
    meta_data_cue_stim = meta_data_cue_stim.reset_index(drop=True)
    #we can now reattach the meta data to the epochs
    #Start by loading the cleand epochs without meta data
    epochs_func_loc = mne.read_epochs(matching_files_funcloc[0], preload=True)
    #We also need to add baseline correction to the epochs
    epochs_func_loc.apply_baseline(baseline=(-0.5, 0))
    #we can find what epochs were kept after ICA using .selection
    selection_indices = epochs_func_loc.selection
    #we can then clean our meta data using this selection_indices
    cleand_metadata_func = meta_data_func_loc.iloc[selection_indices]
    cleand_metadata_func = cleand_metadata_func.reset_index(drop=True)
    epochs_func_loc.metadata = cleand_metadata_func

    #CUEDSTIM
    #We will load the meta data, clean it and then append it to the cued_stim_epochs
    epochs_cued_stim = mne.read_epochs(matching_files_cuedstim[0], preload=True)
    #We also need to add baseline correction to the epochs
    epochs_cued_stim.apply_baseline(baseline=(-0.5, 0))

    selection_indices = epochs_cued_stim.selection
    cleand_metadata_cue = meta_data_cue_stim.iloc[selection_indices]
    cleand_metadata_cue = cleand_metadata_cue.reset_index(drop=True)
    epochs_cued_stim.metadata = cleand_metadata_cue
    #Our meta data is now cleaned and fixed to our epochs for both func loc and cued_stim














    ########################################
    # MACHINE LEARNING ON FUNCTIONAL LOCALIZER DATA #
    ########################################


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
    #save this plot in sub folder
    plot_path = os.path.join(sub_folder, f'{sub}_whole_epoch_decoding_accuracy.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
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
    #save this plot in sub folder
    plot_path = os.path.join(sub_folder, f'{sub}_time_resolved_decoding_accuracy.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()

    #So far we have trained a classifier on the functional localizer data and predicted the image shown to the participant
    #We have done so on parts of the data to look at accuracy and time resolved decoding
    #we can now train a classifier on the entire dataset and on the highest accuracy time point
    
    # Find when the peak accuracy occurs    
    peak_index = mean_scores_time.argmax()
    peak_time = epochs_func_loc.times[peak_index]
    print(f"Peak decoding accuracy of {mean_scores_time[peak_index]:.3f} occurs at {peak_time:.3f} seconds.")

    #TRAINING FINAL CLASSIFIER ON THE ENTIRE DATASET
    #We use the pipeline from the cross-validation, but now we train it on ALL data
    final_whole_epoch_model = pipeline.fit(X, y)

    #we can then save the model 
    models_dir = os.path.join(root, 'models', sub)
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, f'{sub}_whole_epoch_classifier.joblib')

    joblib.dump(final_whole_epoch_model, model_path)

    #we can also save the model at peak time
    # Extract the data ONLY from the peak time point.
    # The shape will go from (n_epochs, n_channels, n_times) to (n_epochs, n_channels)
    X_peak_time = X[:, :, peak_index]

    #pipeline with already 2d data
    peak_time_pipeline = make_pipeline(
        StandardScaler(),
        LogisticRegression(solver='liblinear', random_state=42, max_iter=1000)
    )
    final_peak_model = peak_time_pipeline.fit(X_peak_time, y)
    peak_model_path = os.path.join(models_dir, f'{sub}_peak_time_{int(peak_time*1000)}ms_classifier.joblib')
    joblib.dump(final_peak_model, peak_model_path)


    #we can now use the trained models on unseen data

    #we now have our accuracy scores for the whole epoch and time resolved decoding
    #we know how accurate our model is and when it is most accurate
    #we can now go and check the temporal information linked to the classification

    # The SlidingEstimator will apply our classifier at each time point.
    time_decoder_proba = SlidingEstimator(clf, n_jobs=-1, scoring=None, verbose=True)
    # Use cross_val_predict to get the probabilities for each trial.
    y_pred_proba = cross_val_predict(time_decoder_proba, X, y, cv=cv, method='predict_proba', n_jobs=-1)
    # The classifier outputs probabilities in the order of the unique class labels.
    # Let's get this order to map probabilities back to our image names.
    class_labels = np.unique(y)
    n_classes = len(class_labels)
    # Create a dictionary to store the average probability trace for each image class.
    mean_probas_correct_class = dict()
    # Now, let's calculate the average probability trace for each class.
    for i, label in enumerate(class_labels):
            indices = np.where(y == label)[0]
            proba_trace_for_correct_label = y_pred_proba[indices, :, i] 
            mean_trace = proba_trace_for_correct_label.mean(axis=0)
            mean_probas_correct_class[label] = mean_trace
    
    #plot the results
    plt.figure(figsize=(14, 8))
    # Use a color map to automatically assign different colors to each line
    colors = plt.cm.viridis(np.linspace(0, 1, n_classes))

    for i, label in enumerate(class_labels):
        plt.plot(
            epochs_func_loc.times,
            mean_probas_correct_class[label],
            label=f'Evidence for {label.replace(".jpg", "")}',
            color=colors[i]
        )

    plt.axhline(chance_level, color='black', linestyle=':', label=f'Chance ({chance_level:.2f})')
    plt.axvline(0, color='black', linestyle='-.', label='Stimulus Onset')
    plt.title(f'{sub} - Time Course of Predicted Probability for Correct Class')
    plt.xlabel('Time (s)')
    plt.ylabel('Average Predicted Probability')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left') 
    plt.grid(True, linestyle=':')
    plt.tight_layout() 
    #save this plot in sub folder
    plot_path = os.path.join(sub_folder, f'{sub}_time_course_predicted_probability.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')    
    plt.show()

    ########################################
    # END OF FUNCTIONAL LOCALIZER ANALYSIS #
    ########################################

    ########################################
    # find replay sequence in the data     #
    ########################################

    # 1. Load the participant's behavioral data from the learning phase
    pattern = os.path.join(behaviour_root, sub, 'learn_prob', f'{sub}_learn_probe*.csv')
    files = glob(pattern)
    assert len(files) == 1, f"Expected 1 behaviour file, found {len(files)} for {sub}"
    behaviour_pd = pd.read_csv(files[0])
    
    # 2. Get all unique learning pairs (stim1 -> stim2)
    # The 'pair_name' column tells us which pairs were part of the sequence learning
    learning_pairs_df = behaviour_pd[behaviour_pd['pair_name'].isin(['A_B', 'B_C', 'C_D'])]
    unique_pairs = learning_pairs_df[['stim1_img', 'stim2_img']].drop_duplicates().reset_index(drop=True)
    assert len(unique_pairs) == 3, f"Expected 3 unique learning pairs, found {len(unique_pairs)} for {sub}"

    # 3. Create a transition map (a dictionary: {start_item: end_item})
    transitions = dict(zip(unique_pairs['stim1_img'], unique_pairs['stim2_img']))
    
    # 4. Find the true starting item of the sequence
    # The start item is the one that is a 'stim1_img' but never a 'stim2_img'
    all_starts = set(unique_pairs['stim1_img'])
    all_ends = set(unique_pairs['stim2_img'])
    start_node_set = all_starts - all_ends
    assert len(start_node_set) == 1, "Failed to find a unique start node for the sequence"
    start_node = start_node_set.pop()

    # 5. Walk the chain to reconstruct the full sequence
    reconstructed_sequence = [start_node]
    current_node = start_node
    while current_node in transitions:
        current_node = transitions[current_node]
        reconstructed_sequence.append(current_node)
    
    assert len(reconstructed_sequence) == 4, "Failed to reconstruct a 4-item sequence"
    
    # 6. Apply the final renaming to match the classifier's labels (with .png)
    new_image_file_map = {
        'stimuli/ciseau.png' : 'ciseau',
        'stimuli/face.png' : 'face',
        'stimuli/banane.png' : 'banane',
        'stimuli/zèbre.png' : 'zèbre',
    }
    replay_sequence = [new_image_file_map[item] for item in reconstructed_sequence]


    # Step A: Load the trained classifier
    peak_model_path = os.path.join(models_dir, f'{sub}_peak_time_{int(peak_time*1000)}ms_classifier.joblib')
    tdlm_classifier = joblib.load(peak_model_path)
    class_labels = tdlm_classifier[-1].classes_

    # Step B: Generate evidence trace for POST-learning rest
    epochs_learn_prob = mne.read_epochs(matching_files_learnprob[0], preload=True)
    data = epochs_learn_prob.get_data()
    data_continuous = data.transpose(1, 0, 2).reshape(data.shape[1], -1)
    raw_learn_prob = mne.io.RawArray(data_continuous, epochs_learn_prob.info)
    X_post_rest_reshaped = raw_learn_prob.get_data(picks='eeg').T
    post_rest_evidence_traces = tdlm_classifier.predict_proba(X_post_rest_reshaped)
    evidence_df = pd.DataFrame(post_rest_evidence_traces, columns=class_labels, index=raw_learn_prob.times)

    # Step C: Generate evidence trace for PRE-learning rest (Sanity Check)
    epochs_rest = mne.read_epochs(matching_files_reststate[0], preload=True)
    data_rest = epochs_rest.get_data()
    data_continuous_rest = data_rest.transpose(1, 0, 2).reshape(data_rest.shape[1], -1)
    raw_rest = mne.io.RawArray(data_continuous_rest, epochs_rest.info)
    X_pre_rest_reshaped = raw_rest.get_data(picks='eeg').T
    pre_rest_evidence_traces = tdlm_classifier.predict_proba(X_pre_rest_reshaped)
    evidence_df_rest = pd.DataFrame(pre_rest_evidence_traces, columns=class_labels, index=raw_rest.times)

    ########################################
    # Start of TDLM analysis               #
    ########################################

    # --- Step 1: Define the function to calculate "sequenceness" ---
    def calculate_directional_sequenceness(evidence_df, sequence, lags_samples):
        """
        Calculates both forward and backward sequenceness strength.
        Returns two arrays: (sequenceness_forward, sequenceness_backward)
        """
        n_items = len(sequence)
        sequence_map = {item: i for i, item in enumerate(sequence)}
        
        # Idealized forward transition matrix (e.g., A->B, B->C)
        T_forward = np.zeros((n_items, n_items))
        for i in range(n_items - 1):
            T_forward[sequence_map[sequence[i]], sequence_map[sequence[i+1]]] = 1
            
        sequenceness_forward = []
        sequenceness_backward = []

        # Reorder evidence_df columns to match the sequence order for matrix operations
        evidence_reordered = evidence_df.loc[:, sequence].values

        for lag in lags_samples:
            max_abs_lag = np.abs(lags_samples).max()
            
            y_t = evidence_reordered[max_abs_lag:-max_abs_lag]
            y_t_lag = evidence_reordered[max_abs_lag - lag : -max_abs_lag - lag]
            
            T_empirical = y_t_lag.T @ y_t
            T_empirical /= len(y_t)

            forward_match = np.sum(T_empirical * T_forward)
            backward_match = np.sum(T_empirical * T_forward.T) # T_forward.T is the backward matrix

            sequenceness_forward.append(forward_match - backward_match)
            sequenceness_backward.append(backward_match - forward_match)

        return np.array(sequenceness_forward), np.array(sequenceness_backward)

    # --- Step 2: Define parameters ---
    sfreq = raw_learn_prob.info['sfreq']
    min_lag_ms, max_lag_ms, step_ms = 10, 100, 10 # Focus on fast, plausible replay lags
    lags_ms = np.arange(min_lag_ms, max_lag_ms + step_ms, step_ms)
    lags_samples = (lags_ms / 1000 * sfreq).astype(int)

    # --- Step 3: Analyze POST-learning rest ---
    print("Analyzing POST-learning rest data...")
    sequenceness_post, _ = calculate_directional_sequenceness(evidence_df, replay_sequence, lags_samples)

    # --- Step 4: Analyze PRE-learning rest (the SANITY CHECK) ---
    print("Analyzing PRE-learning rest data...")
    sequenceness_pre, _ = calculate_directional_sequenceness(evidence_df_rest, replay_sequence, lags_samples)

    # --- Step 5: Run Permutation Test to define significance threshold ---
    print("Running permutation test to establish chance level...")
    n_permutations = 500 # Use 500 for speed, 1000+ for publication
    null_sequenceness = np.zeros(n_permutations)
    
    for i in range(n_permutations):
        # Create a null distribution by shuffling the item labels in the sequence
        shuffled_sequence = np.random.permutation(replay_sequence)
        # Calculate sequenceness for the shuffled sequence on the real data.
        # We use POST data as it's the longer recording and provides a stable baseline.
        shuffled_seq_vals, _ = calculate_directional_sequenceness(evidence_df, shuffled_sequence, lags_samples)
        # The test statistic is the maximum sequenceness across all lags
        null_sequenceness[i] = np.max(shuffled_seq_vals)
    
    # Significance threshold is the 95th percentile of the null distribution
    threshold = np.percentile(null_sequenceness, 95)
    print(f"Significance threshold (95th percentile) = {threshold:.4f}")

    # --- Step 6: Count significant replay events ---
    # We define an "event" as any time the sequenceness crosses the significance threshold
    n_events_post = np.sum(np.max(sequenceness_post) > threshold)
    n_events_pre = np.sum(np.max(sequenceness_pre) > threshold)
    # A more continuous measure is the mean sequenceness above threshold
    mean_sig_sequenceness_post = np.mean(sequenceness_post[sequenceness_post > threshold])
    mean_sig_sequenceness_pre = np.mean(sequenceness_pre[sequenceness_pre > threshold])


    # --- Step 7: Visualize and report the results ---
    plt.figure(figsize=(14, 6))

    # Plot Post-learning
    ax1 = plt.subplot(1, 2, 1)
    ax1.plot(lags_ms, sequenceness_post, 'o-', label='Post-Learning Sequenceness')
    ax1.axhline(threshold, color='red', linestyle='--', label='Significance Threshold (p<0.05)')
    ax1.set_title(f'POST-Learning Rest\n{n_events_post} Significant Event(s)')
    ax1.set_xlabel('Time Lag (ms)')
    ax1.set_ylabel('Forward Sequenceness')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot Pre-learning
    ax2 = plt.subplot(1, 2, 2)
    ax2.plot(lags_ms, sequenceness_pre, 'o-', color='green', label='Pre-Learning Sequenceness')
    ax2.axhline(threshold, color='red', linestyle='--', label='Significance Threshold (p<0.05)')
    ax2.set_title(f'PRE-Learning Rest (Sanity Check)\n{n_events_pre} Significant Event(s)')
    ax2.set_xlabel('Time Lag (ms)')
    ax2.set_ylabel('Forward Sequenceness')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle(f'{sub} - Replay Event Detection Analysis', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save the new plot
    plot_path = os.path.join(sub_folder, f'{sub}_replay_event_detection.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()


    ########################################
    # END OF TDLM ANALYSIS REST STATE      #
    ########################################


    
    # Filter epochs for 'forward' and 'backward' cues
    epochs_forward = epochs_cued_stim[epochs_cued_stim.metadata['cue_direction'] == 'forward']
    epochs_backward = epochs_cued_stim[epochs_cued_stim.metadata['cue_direction'] == 'backward']

    # --- Step 2: Generate continuous evidence traces for each condition ---
    # FORWARD-CUED TRIALS
    data_fwd = epochs_forward.get_data()
    data_continuous_fwd = data_fwd.transpose(1, 0, 2).reshape(data_fwd.shape[1], -1)
    raw_fwd = mne.io.RawArray(data_continuous_fwd, epochs_forward.info)
    X_fwd_reshaped = raw_fwd.get_data(picks='eeg').T
    evidence_traces_fwd = tdlm_classifier.predict_proba(X_fwd_reshaped)
    evidence_df_fwd = pd.DataFrame(evidence_traces_fwd, columns=class_labels, index=raw_fwd.times)
    print("Generated evidence traces for FORWARD trials.")

    # BACKWARD-CUED TRIALS
    data_bwd = epochs_backward.get_data()
    data_continuous_bwd = data_bwd.transpose(1, 0, 2).reshape(data_bwd.shape[1], -1)
    raw_bwd = mne.io.RawArray(data_continuous_bwd, epochs_backward.info)
    X_bwd_reshaped = raw_bwd.get_data(picks='eeg').T
    evidence_traces_bwd = tdlm_classifier.predict_proba(X_bwd_reshaped)
    evidence_df_bwd = pd.DataFrame(evidence_traces_bwd, columns=class_labels, index=raw_bwd.times)
    print("Generated evidence traces for BACKWARD trials.")

    # --- Step 3: Calculate directional sequenceness for each condition ---
    # For forward-cued trials, we expect FORWARD sequenceness to be high
    fwd_seq_fwd_trials, bwd_seq_fwd_trials = calculate_directional_sequenceness(evidence_df_fwd, replay_sequence, lags_samples)

    # For backward-cued trials, we expect BACKWARD sequenceness to be high
    fwd_seq_bwd_trials, bwd_seq_bwd_trials = calculate_directional_sequenceness(evidence_df_bwd, replay_sequence, lags_samples)

    # --- Step 4: Visualize the results against the same significance threshold ---
    # We re-use the threshold from the resting-state analysis as a consistent
    # measure of chance-level sequenceness based on the data's intrinsic structure.
    print(f"Using significance threshold from rest: {threshold:.4f}")

    plt.figure(figsize=(14, 6))

    # --- Plot for FORWARD-CUED trials ---
    ax1 = plt.subplot(1, 2, 1)
    ax1.plot(lags_ms, fwd_seq_fwd_trials, 'o-', label='Forward Sequenceness')
    ax1.plot(lags_ms, bwd_seq_fwd_trials, 'o-', color='orange', label='Backward Sequenceness')
    ax1.axhline(threshold, color='red', linestyle='--', label='Significance Threshold (p<0.05)')
    ax1.set_title('Condition: FORWARD-Cued Trials')
    ax1.set_xlabel('Time Lag (ms)')
    ax1.set_ylabel('Sequenceness Score')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axhline(0, color='black', lw=0.5, linestyle=':')

    # --- Plot for BACKWARD-CUED trials ---
    ax2 = plt.subplot(1, 2, 2)
    ax2.plot(lags_ms, fwd_seq_bwd_trials, 'o-', label='Forward Sequenceness')
    ax2.plot(lags_ms, bwd_seq_bwd_trials, 'o-', color='orange', label='Backward Sequenceness')
    ax2.axhline(threshold, color='red', linestyle='--', label='Significance Threshold (p<0.05)')
    ax2.set_title('Condition: BACKWARD-Cued Trials')
    ax2.set_xlabel('Time Lag (ms)')
    ax2.set_ylabel('Sequenceness Score')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axhline(0, color='black', lw=0.5, linestyle=':')

    plt.suptitle(f'{sub} - Directional Replay in Cued Mental Simulation', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save the new plot
    plot_path = os.path.join(sub_folder, f'{sub}_cued_replay_detection.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()

    ######################################################
    # VISUALIZING A REPLAY-TRIGGERED AVERAGE
    ######################################################

    print("\n--- Generating Replay-Triggered Average Plot ---")
    
    # We will use the FORWARD-cued trials, as this is where we would most
    # expect to see a forward replay event.
    
    # --- Step 1: Find the trigger points ---
    # A trigger is a moment when the evidence for the FIRST item in the
    # sequence is very high.
    
    start_item = replay_sequence[0]
    start_item_evidence = evidence_df_fwd[start_item].values
    
    # Define a high threshold to find strong activation peaks.
    # The 95th percentile is a good choice.
    trigger_threshold = np.percentile(start_item_evidence, 95)
    
    # Find the indices of all time points that cross this threshold.
    trigger_indices = np.where(start_item_evidence > trigger_threshold)[0]
    
    # To avoid having overlapping events, we'll only keep the first index
    # in any consecutive block of triggers.
    if len(trigger_indices) > 0:
        triggers = [trigger_indices[0]]
        for i in range(1, len(trigger_indices)):
            if trigger_indices[i] > trigger_indices[i-1] + 1:
                triggers.append(trigger_indices[i])
        trigger_indices = np.array(triggers)
        print(f"Found {len(trigger_indices)} candidate replay event triggers.")
    else:
        print("No strong trigger events found for the first item. Cannot generate plot.")
        trigger_indices = [] # Ensure it's an empty list if no triggers are found

    # --- Step 2: Extract a window of data around each trigger ---
    
    # Define the window length for our plot (e.g., 200 ms)
    window_duration_ms = 200
    sfreq = raw_fwd.info['sfreq']
    window_samples = int(window_duration_ms / 1000 * sfreq)
    
    event_windows = []
    if len(trigger_indices) > 0:
        for trigger_idx in trigger_indices:
            # Ensure the window doesn't go past the end of the data
            if trigger_idx + window_samples < len(evidence_df_fwd):
                window = evidence_df_fwd.iloc[trigger_idx : trigger_idx + window_samples]
                # Reorder columns to match the learned sequence
                window = window[replay_sequence]
                event_windows.append(window.values)

    # --- Step 3: Average the windows and plot ---
    if len(event_windows) > 0:
        # Stack all windows into a 3D array and average them
        averaged_event = np.mean(np.stack(event_windows, axis=0), axis=0)

        # Create the plot
        fig, ax = plt.subplots(figsize=(8, 10))
        
        # Plot the heatmap
        im = ax.imshow(averaged_event, cmap='afmhot', interpolation='bilinear', aspect='auto', vmin=0)
        
        # --- Customize Axes and Labels ---
        # X-axis (States)
        ax.set_xticks(np.arange(len(replay_sequence)))
        ax.set_xticklabels([item.replace('.png','') for item in replay_sequence], fontsize=12)
        ax.xaxis.tick_top()
        
        # Y-axis (Time)
        ax.set_ylabel("Time from Event Onset (ms)", fontsize=14)
        # Create meaningful time labels for the y-axis
        tick_positions = np.linspace(0, window_samples, 5)
        tick_labels = np.linspace(0, window_duration_ms, 5).astype(int)
        ax.set_yticks(tick_positions)
        ax.set_yticklabels(tick_labels)
        
        # Colorbar
        cbar = fig.colorbar(im, ax=ax, shrink=0.5, pad=0.02)
        cbar.set_label("Classifier Probability", fontsize=14, labelpad=10)
        
        # Title
        ax.set_title(f"{sub} - Replay-Triggered Average\n(Averaged over {len(event_windows)} events)", fontsize=16, pad=20)
        
        # Save the new plot
        plot_path = os.path.join(sub_folder, f'{sub}_replay_triggered_average.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()

    else:
        print("Could not generate the replay-triggered average plot because no trigger events were found.")


    

    print(f'analysis for {sub} completed')


        























    
    










