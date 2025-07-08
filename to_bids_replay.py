import shutil
from pathlib import Path

import mne
from mne.datasets import eegbci

from mne_bids import BIDSPath, print_dir_tree, write_raw_bids
from mne_bids.stats import count_events

import os 
import glob
from pathlib import Path

import json 
import pandas as pd
import numpy as np

# Set paths
root = os.getcwd()
eeg_data_raw = os.path.join(root, 'raw_eeg')
bids_output = os.path.join(root, 'bids_output')
meta_root = os.path.join(root, 'behaviour_data')

# Find all subject folders
sub_directories = glob.glob(os.path.join(eeg_data_raw, 'sub-*'))
sub_meta_dir = glob.glob(os.path.join(meta_root, 'sub-*'))


#find the bids_output directory
bids_output = os.path.join(root, 'bids_output')





#The .json file gives infromation about the project and will be written over (if not for this code) once the code is ran
#We will create a code with the relevant information here that will make sure this file has what we want in it + will no be overwritten if the code is reran
dataset_description = {
    "Name": "EEG Replay Project",
    "BIDSVersion": "1.8.0",
    "DatasetType": "raw",
    "License": "CC0",
    "Authors": [
        "Antoine Cyr Bouchard",
        "Michel-Pierre Coll",
    ],
    "Acknowledgements": "Not sure what to acknowledge but we can add later",
    "HowToAcknowledge": "Rewrite this bit once article is submitted or being written",
    "Funding": [
        "Add this once article is submited",
        "Cash money 1_2_3"
    ],
    "EthicsApprovals": [
        "We do not have an tehics comitee yet, this will be changed when we apply for it, try and respect format"
    ]
}

#The code above is essentially what will be written in the file - if we do not want to overwrite we need to check for the files existence
bids_output = Path(bids_output)  
desc_file_path = bids_output / 'dataset_description.json'

if not desc_file_path.exists():
    with open(desc_file_path, 'w', encoding='utf-8') as f:
        # Use indent=4 for a human-readable file
        json.dump(dataset_description, f, ensure_ascii=False, indent=4)



unwanted = ['sub-098']


already_bids = []

sub_directories = [Path(sub) for sub in sub_directories]

#create list where we can append all data frame 
meta_frames = []
meta_frames_func = []
meta_frames_cued = []

#we will gain access to all behaviour file so that if meta data is needed we have access to it
sub_meta_dir= [Path(sub) for sub in sub_meta_dir]

for sub in sub_meta_dir: 
        
        if sub.name in unwanted :
            continue
        #find all files in path
        rest_state_dir = sub / 'rest_state' 
        learn_prob_dir = sub / 'learn_prob'
        func_loc_dir = sub / 'func_loc'
        cued_stim_dir = sub / 'cued_mental_stim'

        #find all csv file 
        rest_state_file = list(rest_state_dir.glob('sub-*rest_state*.csv'))
        learn_prob_file = list(learn_prob_dir.glob('sub-*learn_prob*.csv'))
        func_loc_file = list(func_loc_dir.glob('sub-*FunctionalLocalizer*.csv'))
        cued_stim_file = list(cued_stim_dir.glob('sub-*cued_mental_stim*.csv'))

        #ensure only 1 csv file for each
        assert len(rest_state_file) == 1
        assert len(learn_prob_file) == 1
        assert len(func_loc_file) == 1
        assert len(cued_stim_file) == 1
        #undo list
        func_loc_file = func_loc_file[0]
        cued_stim_file = cued_stim_file[0]
        learn_prob_file = learn_prob_file[0]
        rest_state_file = rest_state_file[0]

        #as of now we only need meta data for the learn prob file to get when the rest state starts 
        learn_prob_file = pd.read_csv(str(learn_prob_file))
        #we need to determine when the rest_instr started and when was the last trigger sent compared to text_4.started
        rest_instr_timer = learn_prob_file['rest_instr.started'].dropna().iloc[-1]
        last_text_4 = learn_prob_file['text_4.started'].dropna().iloc[-1]

        #create a new data frame with sub, rest_instr_timer and last_text_4
        meta_data = pd.DataFrame({'sub': [sub.name], 'rest_instr_timer': [rest_instr_timer], 'last_text_4': [last_text_4]})
        #concat meta_data to the meta_frames
        meta_frames.append(meta_data)

        #we also need to get the meta data for func loc 
        meta_df_func = pd.read_csv(str(func_loc_file))
        meta_clean = meta_df_func[['image_file','presented_word', 'is_match','response_correct']]
        #drow rows of meta_clean that have NaN for image_file and re index
        meta_clean = meta_clean.dropna(subset=['image_file']).reset_index(drop=True)
        new_image_file = {
            'stimuli/ciseau.png' : 'ciseau',
            'stimuli/face.png' : 'face',
            'stimuli/banane.png' : 'banane',
            'stimuli/zèbre.png' : 'zèbre',
        }
        meta_clean['image_file'] = meta_clean['image_file'].replace(new_image_file)

        #add the sub name to meta_clean
        meta_clean['sub'] = sub.name
        #append meta_clean to meta_frames_func
        meta_frames_func.append(meta_clean)

        #WE could also need meta data from the cued stim files - we could therefore add it just in case

        cued_stim_file = pd.read_csv(str(cued_stim_file))
        cued_clean_stim_file = cued_stim_file[['cue_direction', 'cue_text', 'probe_image_file']]
        cued_clean_stim_file = cued_clean_stim_file.dropna(subset=['cue_direction']).reset_index(drop=True)
        cued_clean_stim_file['sub'] = sub.name
        meta_frames_cued.append(cued_clean_stim_file)




        


huge_dataframe = pd.concat(meta_frames, ignore_index=True)

func_huge = pd.concat(meta_frames_func, ignore_index=True)

cued_huge = pd.concat(meta_frames_cued, ignore_index=True)


        


#We need a participant .tsv file that will be added to the meta data, we will append the infromation to a list that will be created here
#multiple sources indicate that this file is necessary, we can add multiple info to this file, for now only participant ID will be added
all_participants_data = [] 



for sub in sub_directories:

    #once our code is more robust and runs the pipeline without issues we will need to change the line under this one for something that checks if the sub is in the deriv path, right now ignore
    if sub.name in unwanted or sub.name in already_bids:
        continue

    

    # find file paths 
    rest_state_dir = Path(sub) / 'rest_state'
    learn_prob_dir = Path(sub) / 'learn_prob'
    func_loc_dir = Path(sub) / 'func_loc'
    cued_stim_dir = Path(sub) / 'mental_stim'

    # find all vhdr paths
    vhdr_files_rest_state = list(rest_state_dir.glob('*.vhdr'))
    vhdr_files_learn_prob = list(learn_prob_dir.glob('*.vhdr'))
    vhdr_files_func_loc = list(func_loc_dir.glob('*.vhdr'))
    vhdr_files_cued_stim = list(cued_stim_dir.glob('*.vhdr'))

    # ensure only 1 vhdr file for each
    assert len(vhdr_files_rest_state) == 1
    assert len(vhdr_files_learn_prob) == 1
    assert len(vhdr_files_func_loc) == 1
    assert len(vhdr_files_cued_stim) == 1

    # load all raw files - these files should be read without a problem
    raw_rest_state = mne.io.read_raw_brainvision(vhdr_files_rest_state[0], preload=True)
    raw_learn_prob = mne.io.read_raw_brainvision(vhdr_files_learn_prob[0], preload=True)
    raw_func_loc = mne.io.read_raw_brainvision(vhdr_files_func_loc[0], preload=True)
    raw_cued_stim = mne.io.read_raw_brainvision(vhdr_files_cued_stim[0], preload=True)




    #we need to get our events for each files + also need to write an event_id code so that our events will be read with information
    #events in our files are not in the events format but in the anotations format
    #They are also fairly messy, we will rewrite them and rename them 

    print([ann['description'] for ann in raw_rest_state.annotations])
    print([ann['description'] for ann in raw_learn_prob.annotations])
    print([ann['description'] for ann in raw_func_loc.annotations])
    print([ann['description'] for ann in raw_cued_stim.annotations])
   
    #We will define a mapping strucutre so the new labels become clean 
    description_mapping = {
        'Fix/F  1' : 'fix',
        'Stimulis/S  1' : 'stim1',
        'Blank/B  1' : 'blank',
        'Word/W  1' : 'word',
        'Kreak/K  1' : 'break',
        'Resp/R  1' : 'resp'
    }


    #create a list with raw files

    raw_list = [raw_rest_state, raw_learn_prob, raw_func_loc, raw_cued_stim]


    #we can then set a custum event code (has to be int's) that will be easier to follow
    event_id_1 = {
        'fix': 1,
        'stim1': 2,
        'blank': 3,
        'word': 4,
        'break': 5,
        'resp' : 6
    }

    # For our new post-learning rest task, there are no relevant events to map.
    event_id_postlearnrest = {}

    
    #we need to create a loop that will loop overs all files to clean the annotations 
    for raw in raw_list :

        #we will now clean and rename all anotations 
        new_descriptions = []
        new_onsets = []
        new_durations = []
        
        
        #add line_freq meta data
        LINE_FREQ = 60
        raw.info['line_freq'] = LINE_FREQ


        for ann in raw.annotations:
            cleaned = description_mapping.get(ann['description'], None)
            if cleaned is not None:
                new_descriptions.append(cleaned)
                new_onsets.append(ann['onset'])
                new_durations.append(ann['duration'])
        
        cleaned_annotations = mne.Annotations(onset=new_onsets, duration=new_durations, description=new_descriptions)
        raw.set_annotations(cleaned_annotations)


    #we should now have cleaned events with ids for each triggers we sent during the experiment :) 

    #we need to crop our learn_prob so that only the rest state is left - to do so we must first match our eeg clock to our psychopy clock
    #use the huge dataframe to get the data
    rest_instr_timer = huge_dataframe[huge_dataframe['sub'] == sub.name]['rest_instr_timer'].values[0]
    last_text_4 = huge_dataframe[huge_dataframe['sub'] == sub.name]['last_text_4'].values[0]

    

    onsets = raw_learn_prob.annotations.onset
    descriptions = np.array(raw_learn_prob.annotations.description)
    try:
        resp_onsets = onsets[descriptions == 'resp']
        last_resp_time = resp_onsets[-1]
        clock_diff = last_resp_time - last_text_4
        crop_start = rest_instr_timer - clock_diff
    except IndexError:
        crop_start = None

    if crop_start is not None:
        raw_postlearnrest = raw_learn_prob.copy().crop(tmin=crop_start, tmax=None)
    else:
        raw_postlearnrest = None


    #we can add meta data of func loc here - we start by opening the huge meta only with the sub name matching sub in this loop
    func_df_sub = func_huge[func_huge['sub'] == sub.name].reset_index(drop=True)
    #we then drop the sub.name
    func_df_sub = func_df_sub.drop(columns=['sub'])
   
    #we need the event meta data to match all events/trigger that we have in our annotations - 5 triggers per run + 1 trigger to start recording
    #n_repeats  = 5
    #new_index = np.repeat(func_df_sub.index, n_repeats)
    #df_expanded_funcloc = func_df_sub.loc[new_index]
    #df_expanded_funcloc.index = range(1, len(df_expanded_funcloc) + 1)
    #df_expanded_funcloc = df_expanded_funcloc.reset_index(drop=True)
   
   #add 1 row to func_df_sub with a NaN value
    #new_row = {'image_file': np.nan, 'presented_word': np.nan, 'is_match': np.nan, 'response_correct': np.nan}
    #create a data frame with new row   
    #new_row_df = pd.DataFrame(new_row, index=[0])
    #concat new_row with df_expanded and make sure new_row is first
    #df_expanded_funcloc = pd.concat([new_row_df, df_expanded_funcloc], ignore_index=True)
    #print first 10 rows of this new df
   
    # 1. Get all cleaned annotations from the raw file.
    annots_funcloc = raw_func_loc.annotations

    # 2. Find the indices of ONLY the 'fix' events.
    stim_indices = [i for i, desc in enumerate(annots_funcloc.description) if desc == 'stim1']
    # 3. Create a new Annotations object containing only the 'fix' events.
    annots_funcloc_stim_only = annots_funcloc[stim_indices]
    # Sanity check: ensure the number of trials in your metadata matches the number of 'fix' events
    assert len(func_df_sub) == len(annots_funcloc_stim_only)
    # 4. Create a copy of the raw object to avoid modifying the original.
    raw_func_loc_for_bids = raw_func_loc.copy()
    raw_func_loc_for_bids.set_annotations(annots_funcloc_stim_only)


    #we will create a dictionnary to descripe the extra event
    extra_event_descriptions = {
        'image_file': 'Image shown to participants',
        'presented_word': 'word presented to participants',
        'is_match' : 'Did the word and image match',
        'response_correct' : '1 is good, 0 is bad'
    }


    #we now can get the meta data from our cued stim
    cued_df_sub = cued_huge[cued_huge['sub'] == sub.name].reset_index(drop=True)
    cued_df_sub = cued_df_sub.drop(columns=['sub'])

    #we also need to match the ammounts of triggers do the event descriptions + meta data (4 triggers + 1 start recording)
    #n_repeats  = 4
    #new_index = np.repeat(cued_df_sub.index, n_repeats)
    #expanded_cued_df = cued_df_sub.loc[new_index]
    #expanded_cued_df.index = range(1, len(expanded_cued_df) + 1)
    #expanded_cued_df = expanded_cued_df.reset_index(drop=True)


    #add another row of empty nan values
    #new_row = {'cue_direction': np.nan, 'cue_text': np.nan, 'probe_image_file': np.nan}
    #new_row_df = pd.DataFrame(new_row, index=[0])
    #expanded_cued_df = pd.concat([new_row_df, expanded_cued_df], ignore_index=True)

    #repeat what we just did for func loc 
    annots_cued = raw_cued_stim.annotations
    fix_indices_cued = [i for i, desc in enumerate(annots_cued.description) if desc == 'fix']
    annots_cued_fix_only = annots_cued[fix_indices_cued]
    assert len(cued_df_sub) == len(annots_cued_fix_only)
    raw_cued_stim_for_bids = raw_cued_stim.copy()
    raw_cued_stim_for_bids.set_annotations(annots_cued_fix_only)

    #we can add the event dictionnary
    extra_event_descriptions_cued = {
        'cue_direction':'direction of mental task',
        'cue_text' : 'text shown to participants',
        'probe_image_file': 'image shown to participants'
    }




    #change sub name so that there is no - 
    folder_name = sub.name
    subject_name = sub.name.split('-')[1]


    #we can add the participants ID to our list from here
    participant_info = {'participant_id': f'sub-{subject_name}'}
    all_participants_data.append(participant_info)

    # create BIDSPath for each raw file
    bids_path_rest_state = BIDSPath(
        subject=subject_name,
        task='reststate',
        root=bids_output
    )
    bids_path_learn_prob = BIDSPath(
        subject=subject_name,
        task='learnprob',
        root=bids_output
    )
    bids_path_func_loc = BIDSPath(
        subject=subject_name,
        task='funcloc',
        root=bids_output
    )
    bids_path_cued_stim = BIDSPath(
        subject=subject_name,
        task='cuedstim',
        root=bids_output
    )
    bids_path_postlearnrest = BIDSPath(
        subject=subject_name, 
        task='postlearnrest',
        root=bids_output)


    #rewrite the eog, veog and ecg channels to match BIDS format - we could write to misc instead 
    raw_rest_state.set_channel_types({'HEOG': 'eog', 'VEOG': 'eog', 'ECG': 'ecg'})
    raw_learn_prob.set_channel_types({'HEOG': 'eog', 'VEOG': 'eog', 'ECG': 'ecg'})
    raw_func_loc.set_channel_types({'HEOG': 'eog', 'VEOG': 'eog', 'ECG': 'ecg'})
    raw_cued_stim.set_channel_types({'HEOG': 'eog', 'VEOG': 'eog', 'ECG': 'ecg'})
    raw_cued_stim_for_bids.set_channel_types({'HEOG': 'eog', 'VEOG': 'eog', 'ECG': 'ecg'})
    raw_func_loc_for_bids.set_channel_types({'HEOG': 'eog', 'VEOG': 'eog', 'ECG': 'ecg'})
    if raw_postlearnrest is not None:
        raw_postlearnrest.set_channel_types({'HEOG': 'eog', 'VEOG': 'eog', 'ECG': 'ecg'})

    # write raw files to BIDS format for files without meta data
    write_raw_bids(
        raw_rest_state,
        bids_path_rest_state,
        overwrite=True,
        allow_preload=True,
        format='BrainVision'
    )
    
    write_raw_bids(
        raw_learn_prob,
        bids_path_learn_prob,
        overwrite=True,
        allow_preload=True,
        format='BrainVision'
    )

    #meta data bids
    write_raw_bids(
        raw=raw_func_loc_for_bids,
        bids_path=bids_path_func_loc,
        event_metadata=func_df_sub,
        extra_columns_descriptions=extra_event_descriptions,
        overwrite=True,
        allow_preload=True,
        format='BrainVision'
    )

    write_raw_bids(
        raw=raw_cued_stim_for_bids,
        bids_path=bids_path_cued_stim,
        event_metadata=cued_df_sub,
        extra_columns_descriptions=extra_event_descriptions_cued,
        overwrite=True,
        allow_preload=True,
        format='BrainVision'
    )



    if raw_postlearnrest is not None:
        write_raw_bids(
            raw_postlearnrest,
            bids_path_postlearnrest,
            event_id=event_id_postlearnrest, 
            overwrite=True,
            allow_preload=True,
            format='BrainVision'
        )

    #add the subject to the already_bids list
    already_bids.append(folder_name)


#now that we have completed our loop we will add our participant file to the bids meta data 
participants_df = pd.DataFrame(all_participants_data)
participants_df = participants_df.drop_duplicates()
participants_df.to_csv(bids_output / 'participants.tsv', sep='\t', index=False)



#for some reason adding meta data has messed up the event naming scheme
#we will use numbers in our config because I do not understand how to fix it and it should work with the numbers instead of keywords like fix
#This is not ideal and we can look for a solution - if a solution is found we should apply + rename to fix the epochs 
