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
from mne.decoding import Vectorizer
from scipy.signal import find_peaks

# --- SETUP ---
subjects = ['sub-027','sub-098','sub-099']
root = os.getcwd()
bids_der = os.path.join(root, 'bids_output', 'derivatives')
cued_stim_eeg = os.path.join(bids_der, 'task_cuedstim', 'preprocessed')
func_loc_eeg = os.path.join(bids_der, 'task_funcloc', 'preprocessed')
learn_prob_eeg = os.path.join(bids_der, 'task_postlearnrest', 'preprocessed')
rest_state_eeg = os.path.join(bids_der, 'task_reststate', 'preprocessed')
bids_root = os.path.join(root, 'bids_output')
deriv_root = os.path.join(root, 'analyse_deriv')
behaviour_root = os.path.join(root, 'behaviour_data')

for sub in subjects:
    # --- File Path Gathering ---
    cuedstim_folder = os.path.join(cued_stim_eeg, sub, 'eeg',)
    pattern = f'{sub}_task-cuedstim_proc-clean_epo.fif'
    matching_files_cuedstim = glob(os.path.join(cuedstim_folder, pattern))
    assert len(matching_files_cuedstim) == 1
    funcloc_folder = os.path.join(func_loc_eeg, sub, 'eeg',)
    pattern = f'{sub}_task-funcloc_proc-clean_epo.fif'
    matching_files_funcloc = glob(os.path.join(funcloc_folder, pattern))
    assert len(matching_files_funcloc) == 1
    learnprob_folder = os.path.join(learn_prob_eeg, sub, 'eeg',)
    pattern = f'{sub}_task-postlearnrest_proc-clean_epo.fif'
    matching_files_learnprob = glob(os.path.join(learnprob_folder, pattern))
    assert len(matching_files_learnprob) == 1
    reststate_folder = os.path.join(rest_state_eeg, sub, 'eeg',)
    pattern = f'{sub}_task-reststate_proc-clean_epo.fif'
    matching_files_reststate = glob(os.path.join(reststate_folder, pattern))
    assert len(matching_files_reststate) == 1

    sub_folder = os.path.join(deriv_root, sub)
    os.makedirs(sub_folder, exist_ok=True)

    # --- META DATA REATTACHMENT AND CLEANING ---
    sub_fodler_bids = os.path.join(bids_root, sub, 'eeg')
    pattern_1 = f'{sub}_task-funcloc_events.tsv'
    pattern_2 = f'{sub}_task-cuedstim_events.tsv'
    matching_meta_funcloc = glob(os.path.join(sub_fodler_bids, pattern_1))
    matching_meta_cuedstim = glob(os.path.join(sub_fodler_bids, pattern_2))
    assert len(matching_meta_funcloc) == 1 and len(matching_meta_cuedstim) == 1
    
    meta_data_func_loc = pd.read_csv(matching_meta_funcloc[0], sep='\t')
    meta_data_cue_stim = pd.read_csv(matching_meta_cuedstim[0], sep='\t')
    
    meta_data_func_loc = meta_data_func_loc[meta_data_func_loc['event_type'] == 'stim1']
    meta_data_func_loc = meta_data_func_loc[['image_file', 'event_type', 'presented_word', 'is_match']].reset_index(drop=True)

    meta_data_cue_stim = meta_data_cue_stim[meta_data_cue_stim['event_type'] == 'fix']
    meta_data_cue_stim = meta_data_cue_stim[['cue_direction', 'cue_text', 'probe_image_file']].reset_index(drop=True)
    
    epochs_func_loc = mne.read_epochs(matching_files_funcloc[0], preload=True, verbose=False)
    epochs_func_loc.apply_baseline(baseline=(-0.5, 0))
    selection_indices_func = epochs_func_loc.selection
    cleand_metadata_func = meta_data_func_loc.iloc[selection_indices_func].reset_index(drop=True)
    epochs_func_loc.metadata = cleand_metadata_func

    epochs_cued_stim = mne.read_epochs(matching_files_cuedstim[0], preload=True, verbose=False)
    epochs_cued_stim.apply_baseline(baseline=(-0.5, 0))
    selection_indices_cued = epochs_cued_stim.selection
    cleand_metadata_cue = meta_data_cue_stim.iloc[selection_indices_cued].reset_index(drop=True)
    epochs_cued_stim.metadata = cleand_metadata_cue
    
    # --- MACHINE LEARNING ON FUNCTIONAL LOCALIZER DATA ---
    bin_width_ms = 10
    sfreq = epochs_func_loc.info['sfreq']
    samples_per_bin = int((bin_width_ms / 1000) * sfreq)
    print(f"\nFor {sub}, using bin width of {bin_width_ms}ms ({samples_per_bin} samples).")

    epochs_for_training = epochs_func_loc.copy().pick('eeg')
    print(f"Channels selected for training: {len(epochs_for_training.ch_names)}")
    X = epochs_for_training.get_data()
    y = epochs_for_training.metadata['image_file']
    model_ch_names = epochs_for_training.ch_names
    n_times = X.shape[2]

    pipeline = make_pipeline(Vectorizer(), StandardScaler(), LogisticRegression(solver='liblinear', random_state=42, max_iter=1000))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    bin_centers_time, all_binned_scores, std_binned_scores = [], [], []
    for i in range(0, n_times - samples_per_bin + 1, samples_per_bin):
        X_bin = X[:, :, i:i + samples_per_bin]
        bin_center_time_sec = epochs_for_training.times[i + samples_per_bin // 2]
        bin_centers_time.append(bin_center_time_sec)
        scores_bin = cross_val_score(pipeline, X_bin, y, cv=cv, scoring='accuracy', n_jobs=-1)
        all_binned_scores.append(scores_bin.mean())
        std_binned_scores.append(scores_bin.std())
    all_binned_scores = np.array(all_binned_scores)
    std_binned_scores = np.array(std_binned_scores)

    chance_level = 1 / len(np.unique(y))
    plt.figure(figsize=(12, 6))
    plt.plot(bin_centers_time, all_binned_scores, label='Mean Binned Decoding Accuracy')
    plt.axhline(chance_level, color='red', linestyle='--', label=f'Chance Accuracy ({chance_level:.2f})')
    plt.axvline(0, color='black', linestyle='-.', label='Stimulus Onset (t=0)')
    plt.fill_between(bin_centers_time, all_binned_scores - std_binned_scores, all_binned_scores + std_binned_scores,
                     alpha=0.2, color='blue', label='±1 Standard Deviation')
    plt.title(f'{sub} - Binned ({bin_width_ms}ms) Time-Resolved Decoding Accuracy')
    plt.xlabel('Time (s)'); plt.ylabel('Classifier Accuracy'); plt.legend(loc='upper left'); plt.grid(True, linestyle=':')
    plt.savefig(os.path.join(sub_folder, f'{sub}_binned_time_resolved_decoding_accuracy.png'), dpi=300)
    plt.show()

    peak_bin_idx = np.argmax(all_binned_scores)
    peak_time_sec = bin_centers_time[peak_bin_idx]
    print(f"Peak decoding accuracy of {all_binned_scores.max():.3f} found at {peak_time_sec:.3f}s.")
    start_sample_peak = peak_bin_idx * samples_per_bin
    X_best_bin = X[:, :, start_sample_peak:start_sample_peak + samples_per_bin]
    
    # <<< CONFUSION MATRIX RESTORED >>>
    print(f"\n--- Generating Confusion Matrix for {sub} using best time bin ---")
    y_pred = cross_val_predict(pipeline, X_best_bin, y, cv=cv, n_jobs=-1)
    labels = sorted(y.unique())
    conf_mx = confusion_matrix(y, y_pred, labels=labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_mx, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.title(f'Confusion Matrix for {sub} (Best Time Bin)')
    plt.ylabel('True Label'); plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(sub_folder, f'{sub}_confusion_matrix_binned.png'), dpi=300)
    plt.show()

    final_binned_model = pipeline.fit(X_best_bin, y)
    models_dir = os.path.join(root, 'models', sub)
    os.makedirs(models_dir, exist_ok=True)
    binned_model_path = os.path.join(models_dir, f'{sub}_final_{bin_width_ms}ms_binned_classifier.joblib')
    joblib.dump(final_binned_model, binned_model_path)
    epochs_cued_stim.pick_channels(model_ch_names, ordered=True)

    # --- IDENTIFY REPLAY SEQUENCE FROM BEHAVIOR ---
    pattern = os.path.join(behaviour_root, sub, 'learn_prob', f'{sub}_learn_probe*.csv')
    files = glob(pattern)
    assert len(files) == 1
    behaviour_pd = pd.read_csv(files[0])
    learning_pairs_df = behaviour_pd[behaviour_pd['pair_name'].isin(['A_B', 'B_C', 'C_D'])]
    unique_pairs = learning_pairs_df[['stim1_img', 'stim2_img']].drop_duplicates()
    transitions = dict(zip(unique_pairs['stim1_img'], unique_pairs['stim2_img']))
    start_node = (set(unique_pairs['stim1_img']) - set(unique_pairs['stim2_img'])).pop()
    reconstructed_sequence = [start_node]
    current_node = start_node
    while current_node in transitions:
        current_node = transitions[current_node]
        reconstructed_sequence.append(current_node)
    new_image_file_map = {'stimuli/ciseau.png':'ciseau', 'stimuli/face.png':'face', 'stimuli/banane.png':'banane', 'stimuli/zèbre.png':'zèbre'}
    replay_sequence = [new_image_file_map.get(item, item) for item in reconstructed_sequence]
    
    # --- TDLM REPLAY ANALYSIS (BINNED METHOD) ---
    tdlm_classifier = joblib.load(binned_model_path)
    class_labels = tdlm_classifier.named_steps['logisticregression'].classes_
    vectorizer_fitted = tdlm_classifier.named_steps['vectorizer']

    def get_binned_evidence(epochs_obj, samples_per_bin, vectorizer):
        data_continuous = epochs_obj.get_data().transpose(1, 0, 2).reshape(epochs_obj.info['nchan'], -1)
        n_chans, n_times = data_continuous.shape
        n_bins = n_times // samples_per_bin
        trimmed_data = data_continuous[:, :n_bins * samples_per_bin]
        reshaped_data = trimmed_data.T.reshape(n_bins, samples_per_bin, n_chans).transpose(0, 2, 1)
        binned_vectorized = vectorizer.transform(reshaped_data)
        binned_scaled = tdlm_classifier.named_steps['standardscaler'].transform(binned_vectorized)
        evidence_traces = tdlm_classifier.named_steps['logisticregression'].predict_proba(binned_scaled)
        return pd.DataFrame(evidence_traces, columns=class_labels)

    epochs_learn_prob = mne.read_epochs(matching_files_learnprob[0], preload=True, verbose=False)
    epochs_learn_prob.pick_channels(model_ch_names, ordered=True)
    evidence_df_post_rest = get_binned_evidence(epochs_learn_prob, samples_per_bin, vectorizer_fitted)

    epochs_rest = mne.read_epochs(matching_files_reststate[0], preload=True, verbose=False)
    epochs_rest.pick_channels(model_ch_names, ordered=True)
    evidence_df_pre_rest = get_binned_evidence(epochs_rest, samples_per_bin, vectorizer_fitted)
    
    epochs_forward = epochs_cued_stim[epochs_cued_stim.metadata['cue_direction'] == 'forward']
    evidence_df_fwd = get_binned_evidence(epochs_forward, samples_per_bin, vectorizer_fitted)
    epochs_backward = epochs_cued_stim[epochs_cued_stim.metadata['cue_direction'] == 'backward']
    evidence_df_bwd = get_binned_evidence(epochs_backward, samples_per_bin, vectorizer_fitted)

    def calculate_directional_sequenceness(evidence_df, sequence, lags_bins):
        n_items = len(sequence)
        sequence_map = {item: i for i, item in enumerate(sequence)}
        T_forward = np.zeros((n_items, n_items))
        for i in range(n_items - 1): T_forward[sequence_map[sequence[i]], sequence_map[sequence[i+1]]] = 1
        sequenceness_forward, sequenceness_backward = [], []
        evidence_reordered = evidence_df.loc[:, sequence].values
        for lag in lags_bins:
            if lag == 0: continue
            y_t, y_t_lag = evidence_reordered[lag:], evidence_reordered[:-lag]
            T_empirical = y_t_lag.T @ y_t / len(y_t)
            sequenceness_forward.append(np.sum(T_empirical * T_forward) - np.sum(T_empirical * T_forward.T))
            sequenceness_backward.append(np.sum(T_empirical * T_forward.T) - np.sum(T_empirical * T_forward))
        return np.array(sequenceness_forward), np.array(sequenceness_backward)

    min_lag_ms, max_lag_ms, step_ms = 10, 100, 10
    lags_ms = np.arange(min_lag_ms, max_lag_ms + step_ms, step_ms)
    lags_bins = (lags_ms / bin_width_ms).astype(int)

    post_rest_fwd_seq, _ = calculate_directional_sequenceness(evidence_df_post_rest, replay_sequence, lags_bins)
    pre_rest_fwd_seq, _ = calculate_directional_sequenceness(evidence_df_pre_rest, replay_sequence, lags_bins)
    cued_fwd_fwd_seq, cued_fwd_bwd_seq = calculate_directional_sequenceness(evidence_df_fwd, replay_sequence, lags_bins)
    cued_bwd_fwd_seq, cued_bwd_bwd_seq = calculate_directional_sequenceness(evidence_df_bwd, replay_sequence, lags_bins)
    
    n_permutations = 500
    null_sequenceness = np.zeros((n_permutations, len(lags_bins)))
    for i in range(n_permutations):
        shuffled_sequence = np.random.permutation(replay_sequence)
        shuffled_seq_vals, _ = calculate_directional_sequenceness(evidence_df_post_rest, shuffled_sequence, lags_bins)
        null_sequenceness[i, :] = shuffled_seq_vals
    threshold = np.percentile(np.max(null_sequenceness, axis=1), 95)

    plt.figure(figsize=(14, 6))
    ax1 = plt.subplot(1, 2, 1)
    ax1.plot(lags_ms, post_rest_fwd_seq, 'o-', label='Post-Learning Sequenceness')
    ax1.axhline(threshold, color='red', linestyle='--', label=f'p<0.05 Threshold ({threshold:.3f})')
    ax1.set_title(f'POST-Learning Rest'); ax1.set_xlabel('Time Lag (ms)'); ax1.set_ylabel('Forward Sequenceness'); ax1.legend(); ax1.grid(True, alpha=0.3)
    ax2 = plt.subplot(1, 2, 2)
    ax2.plot(lags_ms, pre_rest_fwd_seq, 'o-', color='green', label='Pre-Learning Sequenceness')
    ax2.axhline(threshold, color='red', linestyle='--')
    ax2.set_title(f'PRE-Learning Rest (Control)'); ax2.set_xlabel('Time Lag (ms)'); ax2.set_ylabel('Forward Sequenceness'); ax2.legend(); ax2.grid(True, alpha=0.3)
    plt.suptitle(f'{sub} - Resting-State Replay Analysis (Binned)', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(sub_folder, f'{sub}_rest_replay_detection_binned.png'), dpi=300)
    plt.show()

    plt.figure(figsize=(14, 6))
    ax1 = plt.subplot(1, 2, 1)
    ax1.plot(lags_ms, cued_fwd_fwd_seq, 'o-', label='Forward Sequenceness')
    ax1.plot(lags_ms, cued_fwd_bwd_seq, 'o-', color='orange', label='Backward Sequenceness')
    ax1.axhline(threshold, color='red', linestyle='--', label=f'p<0.05 Threshold ({threshold:.3f})')
    ax1.set_title('Condition: FORWARD-Cued Trials'); ax1.set_xlabel('Time Lag (ms)'); ax1.set_ylabel('Sequenceness Score'); ax1.legend(); ax1.grid(True, alpha=0.3); ax1.axhline(0, color='black', lw=0.5, linestyle=':')
    ax2 = plt.subplot(1, 2, 2)
    ax2.plot(lags_ms, cued_bwd_fwd_seq, 'o-', label='Forward Sequenceness')
    ax2.plot(lags_ms, cued_bwd_bwd_seq, 'o-', color='orange', label='Backward Sequenceness')
    ax2.axhline(threshold, color='red', linestyle='--')
    ax2.set_title('Condition: BACKWARD-Cued Trials'); ax2.set_xlabel('Time Lag (ms)'); ax2.set_ylabel('Sequenceness Score'); ax2.legend(); ax2.grid(True, alpha=0.3); ax2.axhline(0, color='black', lw=0.5, linestyle=':')
    plt.suptitle(f'{sub} - Cued Replay Analysis (Binned)', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(sub_folder, f'{sub}_cued_replay_detection_binned.png'), dpi=300)
    plt.show()

    # --- VISUALIZING REPLAY-TRIGGERED AVERAGES (ADAPTED FOR BINS) ---
    window_duration_ms = 200
    window_bins = int(window_duration_ms / bin_width_ms)
    min_peak_distance_bins = int(50 / bin_width_ms)
    fwd_event_windows_to_save, bwd_event_windows_to_save = [], []
    analysis_conditions = {'Forward_Cued': (evidence_df_fwd, replay_sequence[0], replay_sequence), 'Backward_Cued': (evidence_df_bwd, replay_sequence[-1], list(reversed(replay_sequence)))}

    for condition_name, (evidence_df, trigger_item, plot_sequence) in analysis_conditions.items():
        if evidence_df.empty: continue
        expected_shape = (window_bins, len(plot_sequence))
        trigger_item_evidence = evidence_df[trigger_item].values
        trigger_threshold = np.percentile(trigger_item_evidence, 95)
        trigger_indices, _ = find_peaks(trigger_item_evidence, height=trigger_threshold, distance=min_peak_distance_bins)
        
        event_windows_for_plotting = []
        for trigger_idx in trigger_indices:
            if trigger_idx + window_bins <= len(evidence_df):
                window = evidence_df.iloc[trigger_idx : trigger_idx + window_bins]
                if window.shape == expected_shape:
                    event_windows_for_plotting.append(window[plot_sequence].values)
        
        print(f"Condition '{condition_name}': Found {len(trigger_indices)} triggers, kept {len(event_windows_for_plotting)} with complete shape.")
        
        if 'Forward' in condition_name: fwd_event_windows_to_save = event_windows_for_plotting
        else: bwd_event_windows_to_save = event_windows_for_plotting

        if event_windows_for_plotting:
            averaged_event = np.mean(np.stack(event_windows_for_plotting, axis=0), axis=0)
            fig, ax = plt.subplots(figsize=(8, 6))
            # <<< HEATMAP COLOR CHANGE >>>
            # Calculate chance level for the color map limit
            chance_level_heatmap = 1 / len(plot_sequence)
            im = ax.pcolormesh(np.arange(len(plot_sequence) + 1), np.arange(window_bins + 1) * bin_width_ms, 
                               averaged_event, cmap='afmhot', vmin=chance_level_heatmap)
            ax.set_title(f"{sub} - {condition_name} Replay Average\n({len(event_windows_for_plotting)} events)", fontsize=14)
            ax.set_xlabel("Sequence Item"); ax.set_ylabel("Time from Trigger (ms)")
            ax.set_xticks(np.arange(len(plot_sequence)) + 0.5)
            ax.set_xticklabels([item.replace('.png','') for item in plot_sequence])
            ax.invert_yaxis()
            fig.colorbar(im, ax=ax, label="Classifier Probability")
            plt.tight_layout()
            plt.savefig(os.path.join(sub_folder, f'{sub}_{condition_name}_replay_triggered_average_binned.png'), dpi=300)
            plt.show()

    # --- SAVE NUMERICAL RESULTS FOR GROUP ANALYSIS ---
    # ... (permutation test to get threshold is correct) ...
    n_permutations = 500
    # ... (sequenceness calculation) ...
    
    # <<< THE DEFINITIVE FIX: Stack windows into a clean 3D float array before saving >>>
    fwd_windows_3d = np.stack(fwd_event_windows_to_save) if fwd_event_windows_to_save else np.array([])
    bwd_windows_3d = np.stack(bwd_event_windows_to_save) if bwd_event_windows_to_save else np.array([])

    results_to_save = {
        'lags_ms': lags_ms, 'post_learn_rest': post_rest_fwd_seq, 'pre_learn_rest': pre_rest_fwd_seq,
        'cued_fwd_forward_seq': cued_fwd_fwd_seq, 'cued_fwd_backward_seq': cued_fwd_bwd_seq,
        'cued_bwd_forward_seq': cued_bwd_fwd_seq, 'cued_bwd_backward_seq': cued_bwd_bwd_seq,
        'significance_threshold': threshold, 'replay_sequence_items': np.array(replay_sequence),
        # Save the clean 3D arrays, NOT the object arrays
        'forward_cued_event_windows': fwd_windows_3d,
        'backward_cued_event_windows': bwd_windows_3d
    }

    results_path = os.path.join(sub_folder, f'{sub}_sequenceness_results_binned.npz')
    np.savez(results_path, **results_to_save)
    print(f'Analysis for {sub} completed.')
    print('-' * 60)

print("\nAll analyses completed successfully!")