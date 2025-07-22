import numpy as np
import matplotlib.pyplot as plt
import os
from glob import glob
from scipy import stats
import mne # We'll use MNE for its powerful stats functions

# --- 1. SETUP ---
subjects = ['sub-027', 'sub-098', 'sub-099'] # Add all subjects
root = os.getcwd()
deriv_root = os.path.join(root, 'analyse_deriv')
group_deriv_root = os.path.join(deriv_root, 'group_level')
os.makedirs(group_deriv_root, exist_ok=True)

bin_width_ms = 10

# --- 2. DATA COLLECTION LOOP ---
all_post_rest, all_cued_fwd_fwd, all_cued_bwd_bwd = [], [], []
subject_fwd_event_windows, subject_bwd_event_windows = [], []
lags_ms = None
replay_sequence_items = None
n_sequence_items = None

print("--- Starting Data Collection ---")
for sub in subjects:
    file_path = os.path.join(deriv_root, sub, f'{sub}_sequenceness_results_binned.npz')
    
    if os.path.exists(file_path):
        print(f"Loading data for {sub}...")
        data = np.load(file_path, allow_pickle=True)
        
        if lags_ms is None: lags_ms = data['lags_ms']
        if replay_sequence_items is None: 
            replay_sequence_items = data['replay_sequence_items']
            n_sequence_items = len(replay_sequence_items)

        all_post_rest.append(data['post_learn_rest'])
        all_cued_fwd_fwd.append(data['cued_fwd_forward_seq'])
        all_cued_bwd_bwd.append(data['cued_bwd_backward_seq'])
        
        if 'forward_cued_event_windows' in data and data['forward_cued_event_windows'].size > 0:
            subject_fwd_event_windows.append(data['forward_cued_event_windows'])
        
        if 'backward_cued_event_windows' in data and data['backward_cued_event_windows'].size > 0:
            subject_bwd_event_windows.append(data['backward_cued_event_windows'])
    else:
        print(f"Warning: Could not find results file for {sub} at {file_path}")

print("--- Data Collection Finished ---")

# --- 3. DATA AGGREGATION & ANALYSIS ---
print("\n--- Aggregating and Analyzing Group Data ---")

all_post_rest = np.array(all_post_rest)
all_cued_fwd_fwd = np.array(all_cued_fwd_fwd)
all_cued_bwd_bwd = np.array(all_cued_bwd_bwd)

grand_average_fwd_windows = np.vstack(subject_fwd_event_windows) if subject_fwd_event_windows else None
grand_average_bwd_windows = np.vstack(subject_bwd_event_windows) if subject_bwd_event_windows else None

# --- 4. GROUP-LEVEL TDLM PLOTS (Quantitative) ---
def plot_group_sequenceness(data, lags, title, ax, ylabel='Sequenceness Score'):
    n_subjects = data.shape[0]
    if n_subjects < 2:
        ax.set_title(f"{title}\n(Not enough data: {n_subjects} subjects)")
        return
        
    mean_seq, sem_seq = np.mean(data, axis=0), stats.sem(data, axis=0)
    
    # <<< STATISTICAL ANALYSIS RESTORED >>>
    # This block was missing and is now added back.
    try:
        # We test against 0 (chance) and use a one-tailed test because our hypothesis is directional (sequenceness > 0)
        t_obs, clusters, cluster_p_values, H0 = mne.stats.permutation_cluster_1samp_test(
            data, n_permutations=1024, tail=1, n_jobs=-1
        )
        
        # Find clusters with p-value less than 0.05
        sig_clusters = [c[0] for i, c in enumerate(clusters) if cluster_p_values[i] < 0.05]
        
        if sig_clusters:
            # Create a single legend entry for all significant clusters
            ax.axvspan(0, 0, color='red', alpha=0.3, label='p < 0.05 (cluster-corrected)')
            # Shade the significant time lags
            for clst in sig_clusters:
                ax.axvspan(lags[clst.start], lags[clst.stop-1], color='red', alpha=0.3)
    except Exception as e:
        print(f"Could not run cluster stats for '{title}': {e}")
    # <<< END OF RESTORED BLOCK >>>

    ax.axhline(0, color='black', lw=0.5)
    ax.plot(lags, mean_seq, 'o-', label=f'Mean (N={n_subjects})')
    ax.fill_between(lags, mean_seq - sem_seq, mean_seq + sem_seq, alpha=0.2, label='SEM')
    ax.set_title(title)
    ax.set_xlabel('Time Lag (ms)')
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True, alpha=0.3)

fig_tdlm, axes_tdlm = plt.subplots(1, 3, figsize=(20, 6), sharey=True)
fig_tdlm.suptitle('Group-Level TDLM Analysis (Binned)', fontsize=16)
plot_group_sequenceness(all_post_rest, lags_ms, 'Spontaneous Replay (Post-Learn Rest)', axes_tdlm[0], ylabel='Forward Sequenceness')
plot_group_sequenceness(all_cued_fwd_fwd, lags_ms, 'Forward Replay (Forward Cue)', axes_tdlm[1], ylabel='Forward Sequenceness')
plot_group_sequenceness(all_cued_bwd_bwd, lags_ms, 'Backward Replay (Backward Cue)', axes_tdlm[2], ylabel='Backward Sequenceness')
fig_tdlm.tight_layout(rect=[0, 0.03, 1, 0.95])
fig_tdlm.savefig(os.path.join(group_deriv_root, 'group_tdlm_analysis_binned.png'), dpi=300)
plt.show()


# --- 5. GROUP-LEVEL REPLAY-TRIGGERED HEATMAPS (Directional) ---
def plot_group_heatmap(event_windows, subject_event_list, sequence_labels, title_prefix, output_filename, bin_width):
    if event_windows is None or event_windows.size == 0:
        print(f"No valid replay event windows found for '{title_prefix}'. Skipping heatmap.")
        return
    grand_averaged_event = np.mean(event_windows, axis=0)
    n_subjects_with_events = len([s for s in subject_event_list if s.size > 0])
    n_total_events = event_windows.shape[0]
    n_sequence_items = grand_averaged_event.shape[1]
    window_bins = grand_averaged_event.shape[0]

    fig_heatmap, ax_heatmap = plt.subplots(figsize=(8, 6))
    x_axis = np.arange(n_sequence_items + 1)
    y_axis = np.arange(window_bins + 1) * bin_width
    
    chance_level_heatmap = 1 / n_sequence_items
    im = ax_heatmap.pcolormesh(x_axis, y_axis, grand_averaged_event, cmap='afmhot', vmin=chance_level_heatmap)
    
    title = (f"Group {title_prefix} Replay-Triggered Average\n"
             f"(Averaged over {n_total_events} events from {n_subjects_with_events} subjects)")
    ax_heatmap.set_title(title, fontsize=14)
    ax_heatmap.set_xticks(np.arange(n_sequence_items) + 0.5)
    ax_heatmap.set_xticklabels([label.replace('.png','') for label in sequence_labels], fontsize=12)
    ax_heatmap.set_xlabel("Expected Sequence Item", fontsize=12)
    ax_heatmap.set_ylabel("Time from Trigger Onset (ms)", fontsize=12)
    ax_heatmap.invert_yaxis()
    cbar = fig_heatmap.colorbar(im, ax=ax_heatmap)
    cbar.set_label("Classifier Probability", fontsize=12)
    fig_heatmap.tight_layout()
    fig_heatmap.savefig(output_filename, dpi=300)
    plt.show()

# --- Call the plotting function for each condition ---
if replay_sequence_items is not None:
    print("\n--- Generating Group Heatmaps ---")
    plot_group_heatmap(
        grand_average_fwd_windows,
        subject_event_list=subject_fwd_event_windows,
        sequence_labels=replay_sequence_items,
        title_prefix='Forward-Cued',
        output_filename=os.path.join(group_deriv_root, 'group_forward_replay_triggered_average_binned.png'),
        bin_width=bin_width_ms
    )
    plot_group_heatmap(
        grand_average_bwd_windows,
        subject_event_list=subject_bwd_event_windows,
        sequence_labels=list(reversed(replay_sequence_items)),
        title_prefix='Backward-Cued',
        output_filename=os.path.join(group_deriv_root, 'group_backward_replay_triggered_average_binned.png'),
        bin_width=bin_width_ms
    )
else:
    print("Cannot generate heatmaps because replay_sequence_items was not loaded.")