import numpy as np
import matplotlib.pyplot as plt
import os
from glob import glob
from scipy import stats
import mne # We'll use MNE for its powerful stats functions

# --- 1. SETUP ---
subjects = ['sub-027', 'sub-099', 'sub-098'] # Add all subjects
root = os.getcwd()
deriv_root = os.path.join(root, 'analyse_deriv')

# --- 2. AGGREGATE DATA ---
all_post_rest, all_cued_fwd_fwd, all_cued_bwd_bwd, all_event_windows = [], [], [], []
lags_ms = None

print("Loading data for all subjects...")
for sub in subjects:
    file_path = os.path.join(deriv_root, sub, f'{sub}_sequenceness_results.npz')
    if os.path.exists(file_path):
        data = np.load(file_path, allow_pickle=True)
        
        if lags_ms is None:
            lags_ms = data['lags_ms']
        all_post_rest.append(data['post_learn_rest'])
        all_cued_fwd_fwd.append(data['cued_fwd_forward_seq'])
        all_cued_bwd_bwd.append(data['cued_bwd_backward_seq'])
        
        if 'replay_event_windows' in data and data['replay_event_windows'].size > 0:
            all_event_windows.append(data['replay_event_windows'])
    else:
        print(f"Warning: Could not find results file for {sub}")

all_post_rest = np.array(all_post_rest)
all_cued_fwd_fwd = np.array(all_cued_fwd_fwd)
all_cued_bwd_bwd = np.array(all_cued_bwd_bwd)

if all_event_windows:
    grand_average_windows = np.concatenate(all_event_windows, axis=0)
    print(f"Aggregated a total of {grand_average_windows.shape[0]} replay events across {len(subjects)} subjects.")
else:
    grand_average_windows = None
    print("No replay event windows found to create a group heatmap.")


# --- 3. GROUP-LEVEL TDLM PLOTS ---
# (This section can remain unchanged, as it correctly analyzes the quantitative data)
def plot_group_sequenceness(data, lags, title, ax):
    n_subjects = data.shape[0]
    if n_subjects < 2:
        ax.set_title(f"{title}\n(Not enough data)")
        return

    mean_seq, sem_seq = np.mean(data, axis=0), stats.sem(data, axis=0)
    
    try:
        _, clusters, cluster_p_values, _ = mne.stats.permutation_cluster_1samp_test(
            data, n_permutations=1024, tail=1, n_jobs=-1
        )
        for i_c, c in enumerate(clusters):
            if cluster_p_values[i_c] < 0.05:
                c = c[0]
                ax.axvspan(lags[c.start], lags[c.stop-1], color='red', alpha=0.3, 
                           label=f'p={cluster_p_values[i_c]:.3f}')
    except Exception:
        pass # Ignore stats errors if data is not suitable

    ax.axhline(0, color='black', lw=0.5)
    ax.plot(lags, mean_seq, 'o-', label=f'Mean (N={n_subjects})')
    ax.fill_between(lags, mean_seq - sem_seq, mean_seq + sem_seq, alpha=0.2, label='SEM')
    ax.set_title(title)
    ax.set_xlabel('Time Lag (ms)')
    ax.set_ylabel('Sequenceness Score')
    ax.legend()
    ax.grid(True, alpha=0.3)

fig_tdlm, axes_tdlm = plt.subplots(1, 3, figsize=(20, 6), sharey=True)
fig_tdlm.suptitle('Group-Level TDLM Analysis', fontsize=16)
plot_group_sequenceness(all_post_rest, lags_ms, 'Spontaneous Replay (Post-Learn Rest)', axes_tdlm[0])
plot_group_sequenceness(all_cued_fwd_fwd, lags_ms, 'Forward Replay (Forward Cue)', axes_tdlm[1])
axes_tdlm[2].set_ylabel('Backward Sequenceness Score')
plot_group_sequenceness(all_cued_bwd_bwd, lags_ms, 'Backward Replay (Backward Cue)', axes_tdlm[2])
fig_tdlm.tight_layout(rect=[0, 0.03, 1, 0.95])
fig_tdlm.savefig(os.path.join(deriv_root, 'group_tdlm_analysis.png'), dpi=300)
plt.show()


# --- 4. GROUP-LEVEL REPLAY-TRIGGERED HEATMAP (ORDINAL) ---

if grand_average_windows is not None:
    # Calculate the grand average across all ALIGNED events
    grand_averaged_event = np.mean(grand_average_windows, axis=0)
    
    n_sequence_items = grand_averaged_event.shape[1]
    window_samples = grand_averaged_event.shape[0]
    sfreq = 500 # Assume 500 Hz from your config
    window_duration_ms = window_samples / sfreq * 1000

    fig_heatmap, ax_heatmap = plt.subplots(figsize=(8, 10))
    
    im = ax_heatmap.imshow(grand_averaged_event, cmap='afmhot', interpolation='none', 
                           aspect='auto', origin='upper', vmin=0)
    
    # --- Customize Axes and Labels for ORDINAL positions ---
    ax_heatmap.set_title(f"Group Replay-Triggered Average\n(Averaged over {grand_average_windows.shape[0]} events from {len(subjects)} subjects)", 
                         fontsize=16, pad=20)
    
    # X-axis: Use ordinal labels
    ordinal_labels = [f'Stim {i+1}' for i in range(n_sequence_items)]
    ax_heatmap.set_xticks(np.arange(n_sequence_items))
    ax_heatmap.set_xticklabels(ordinal_labels, fontsize=12)
    ax_heatmap.xaxis.tick_top()
    ax_heatmap.xaxis.set_label_position('top')
    ax_heatmap.set_xlabel("Ordinal Position in Sequence", fontsize=14, labelpad=10)
    
    # Y-axis
    ax_heatmap.set_ylabel("Time from Peak Onset (ms)", fontsize=14)
    tick_positions = np.linspace(0, window_samples - 1, 5)
    tick_labels = np.linspace(0, window_duration_ms, 5).astype(int)
    ax_heatmap.set_yticks(tick_positions)
    ax_heatmap.set_yticklabels(tick_labels)
    
    # Colorbar
    cbar = fig_heatmap.colorbar(im, ax=ax_heatmap, shrink=0.6, pad=0.03)
    cbar.set_label("Classifier Probability", fontsize=14, labelpad=10)
    
    # Save the plot
    fig_heatmap.tight_layout()
    fig_heatmap.savefig(os.path.join(deriv_root, 'group_ordinal_replay_triggered_average.png'), dpi=300)
    plt.show()