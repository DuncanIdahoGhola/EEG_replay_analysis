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
meta_frames_rest = []

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
        rest_instr_timer = learn_prob_file['rest_instr.stopped'].dropna().iloc[-1]
        last_text_4 = learn_prob_file['text_4.started'].dropna().iloc[-1]
        end_exp_start = learn_prob_file['end_exp.started'].dropna().iloc[-1]
        

        #create a new data frame with sub, rest_instr_timer and last_text_4
        meta_data = pd.DataFrame({'sub': [sub.name], 'rest_instr_timer': [rest_instr_timer], 'last_text_4': [last_text_4], 'end_of_exp': [end_exp_start]})
        #concat meta_data to the meta_frames
        meta_frames.append(meta_data)

        #we also need to get the meta data for func loc 
        meta_df_func = pd.read_csv(str(func_loc_file))
        #drow rows of meta_clean that have NaN for image_file and re index

        meta_df_func_clean = meta_df_func.dropna(subset=['image_file']).reset_index(drop=True)

        #meta_clean = meta_df_func[['image_file','presented_word', 'is_match','response_correct']]
        new_image_file = {
            'stimuli/ciseau.png' : 'ciseau',
            'stimuli/face.png' : 'face',
            'stimuli/banane.png' : 'banane',
            'stimuli/zèbre.png' : 'zèbre',
        }
        meta_df_func_clean['image_file'] = meta_df_func_clean['image_file'].replace(new_image_file)
        #add the sub name to meta_clean
        meta_df_func_clean['sub'] = sub.name
        #append meta_clean to meta_frames_func
        meta_frames_func.append(meta_df_func_clean)

        #WE could also need meta data from the cued stim files - we could therefore add it just in case

        cued_stim_file = pd.read_csv(str(cued_stim_file))

        #cued_clean_stim_file = cued_stim_file[['cue_direction', 'cue_text', 'probe_image_file']]
        cued_clean_stim_file = cued_stim_file.dropna(subset=['cue_direction']).reset_index(drop=True)
        cued_clean_stim_file['sub'] = sub.name
        meta_frames_cued.append(cued_clean_stim_file)



        #we will get the meta data from the rest state to crop it to exaclty 300 seconds
        rest_state_file = pd.read_csv(str(rest_state_file))
        rest_state_started = rest_state_file['rest_state.started'].dropna().iloc[-1]
        rest_state_stopped = rest_state_file['rest_state.stopped'].dropna().iloc[-1]

        rest_state_meta = pd.DataFrame({'sub': [sub.name], 'rest_state_started': [rest_state_started], 'rest_state_stopped': [rest_state_stopped]})
        meta_frames_rest.append(rest_state_meta)

        






        


huge_dataframe = pd.concat(meta_frames, ignore_index=True)

func_huge = pd.concat(meta_frames_func, ignore_index=True)

cued_huge = pd.concat(meta_frames_cued, ignore_index=True)

rest_huge = pd.concat(meta_frames_rest, ignore_index=True)


        


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
    end_of_exp = huge_dataframe[huge_dataframe['sub'] == sub.name]['end_of_exp'].values[0]
    
    onsets = raw_learn_prob.annotations.onset
    descriptions = np.array(raw_learn_prob.annotations.description)
    if 'resp' in descriptions:
        resp_onsets = onsets[descriptions == 'resp']
        last_resp_time = resp_onsets[-1]
        clock_diff = last_resp_time - last_text_4
        crop_start = rest_instr_timer - clock_diff
        crop_end = end_of_exp - clock_diff
    else:
        crop_start = rest_instr_timer


    if crop_start is not None:
        raw_postlearnrest = raw_learn_prob.copy().crop(tmin=crop_start, tmax=crop_end)
        #we also need to remove anotations that are not relevant to the rest state
        raw_postlearnrest.set_annotations(None)
    else:
        raw_postlearnrest = None



    #we should also crop our rest state to ensure that it lasts 300seconds
    rest_state_start = rest_huge[rest_huge['sub'] == sub.name]['rest_state_started'].values[0]
    rest_state_end = rest_huge[rest_huge['sub'] == sub.name]['rest_state_stopped'].values[0]

    #find onsets to find another cloc diff

    onsets_rest = raw_rest_state.annotations.onset
    descriptions_rest = np.array(raw_rest_state.annotations.description)

    if 'fix' in descriptions_rest:
        fix_onsets = onsets_rest[descriptions_rest == 'fix']
        clock_diff_rest = rest_state_start - fix_onsets[0]
        crop_start_rest = rest_state_start - clock_diff_rest
        crop_end_rest = rest_state_end - clock_diff_rest
    else:
        crop_start_rest = rest_state_start
        crop_end_rest = rest_state_end

    if crop_start_rest is not None:
        raw_rest_state = raw_rest_state.copy().crop(tmin=crop_start_rest, tmax=crop_end_rest)
        #we need to remove annotations that are not relevant to the rest state
        raw_rest_state.set_annotations(None)
    else:
        raw_rest_state = None

    

    #we can add meta data of func loc here - we start by opening the huge meta only with the sub name matching sub in this loop
    func_df_sub = func_huge[func_huge['sub'] == sub.name].reset_index(drop=True)
    #we then drop the sub.name
    func_df_sub = func_df_sub.drop(columns=['sub'])
   
    #we need the event meta data to match all events/trigger that we have in our annotations - 5 triggers per run + 1 trigger to start recording
    n_repeats  = 5
    new_index = np.repeat(func_df_sub.index, n_repeats)
    df_expanded_funcloc = func_df_sub.loc[new_index]
   
   #add 1 row to func_df_sub with a NaN value
    #new_row = {'image_file': np.nan, 'presented_word': np.nan, 'is_match': np.nan, 'response_correct': np.nan}
    #create a data frame with new row   
    #new_row_df = pd.DataFrame(new_row, index=[0])
    #concat new_row with df_expanded and make sure new_row is first
    #df_expanded_funcloc = pd.concat([new_row_df, df_expanded_funcloc], ignore_index=True)
    #print first 10 rows of this new df
   
    #we will create a dictionnary to descripe the extra event
    extra_event_descriptions = {
        # == Core Trial & Condition Information ==
        'image_file': 'The relative path to the image file presented on this trial.',
        'presented_word': 'The word stimulus presented on this trial.',
        'is_match': 'The pre-defined condition of the trial, specifying if the image and word were a "match" or "mismatch".',
        'corrAns': 'The correct answer key for the trial (e.g., "1" for mismatch, "2" for match).',
        'blank_duration': 'Planned duration in seconds of the blank screen that appears after the word stimulus.',
        'iti_duration': 'Planned duration in seconds of the inter-trial interval (ITI) that follows the trial.',

        # == Participant Response Data ==
        'key_resp.keys': 'The key pressed by the participant to respond to the stimulus.',
        'key_resp.corr': 'Indicates if the participant\'s response was correct (1) or incorrect (0).',
        'key_resp.rt': 'Reaction time of the participant\'s response, in seconds, from the start of the response window.',
        'key_resp.duration': 'Duration in seconds for which the response key was held down.',
        'finish_time.keys': 'Key pressed by the experimenter to pause or end the experiment prematurely (e.g., "p").',
        'finish_time.rt': 'Reaction time for the key press to pause or end the experiment.',
        'finish_time.duration': 'Duration the key was held down to pause or end the experiment.',
        'instr_resp.keys': 'The key pressed by the participant to advance past the instructions.',
        'instr_resp.rt': 'Time taken in seconds for the participant to advance past the instructions.',
        'instr_resp.duration': 'Duration the instruction-advancement key was held down.',
        'key_resp_3.keys': 'The key pressed to trigger the start of the experiment trials (e.g., scanner trigger or spacebar).',
        'key_resp_3.rt': 'Time taken in seconds to press the trigger key after the prompt appeared.',
        'key_resp_3.duration': 'Duration the trigger key was held down.',
        'key_resp_2.keys': 'The key pressed by the participant to dismiss the final "end of experiment" screen.',

        # == Custom BIDS-related Variables (Likely created during processing) ==
        'image_presented': 'The relative path to the image file presented in the trial (BIDS-compliant name).',
        'word_presented': 'The word presented in the trial (BIDS-compliant name).',
        'is_match_condition': 'The specified condition for the trial: match or mismatch (BIDS-compliant name).',
        'response_given': 'The key ("1" or "2") pressed by the participant as their response (BIDS-compliant name).',
        'response_correct': 'A binary indicator of whether the response was correct (1) or incorrect (0) (BIDS-compliant name).',
        'response_rt': 'The reaction time of the participant\'s response for the trial, in seconds (BIDS-compliant name).',

        # == PsychoPy Loop & Trial Counters ==
        'thisN': 'The overall trial number within the entire experiment, 0-indexed (PsychoPy variable).',
        'thisTrialN': 'The trial number within the current loop or block, 0-indexed (PsychoPy variable).',
        'thisRepN': 'The repetition (block) number of the current loop, 0-indexed (PsychoPy variable).',
        'inside_loop_trials.thisRepN': 'Repetition number of the main trial loop (PsychoPy internal log).',
        'inside_loop_trials.thisTrialN': 'Trial number within the current loop repetition (PsychoPy internal log).',
        'inside_loop_trials.thisN': 'Overall trial number within the main trial loop (PsychoPy internal log).',
        'inside_loop_trials.thisIndex': 'The 0-indexed number of the trial from the conditions file (PsychoPy internal log).',
        'inside_loop_trials.key_resp.keys': 'Key pressed, logged internally by the trial loop object.',
        'inside_loop_trials.key_resp.corr': 'Correctness of response, logged internally by the trial loop object.',
        'inside_loop_trials.key_resp.rt': 'Reaction time, logged internally by the trial loop object.',
        'inside_loop_trials.key_resp.duration': 'Key press duration, logged internally by the trial loop object.',
        'inside_loop_trials.finish_time.keys': 'Experimenter key press to pause/end, logged internally by the trial loop object.',
        'inside_loop_trials.finish_time.rt': 'Reaction time of experimenter pause/end key, logged internally by the trial loop object.',
        'inside_loop_trials.finish_time.duration': 'Duration of experimenter pause/end key, logged internally by the trial loop object.',

        # == Timestamps & Durations (System-level) ==
        'thisRow.t': 'Time elapsed in seconds since the start of the experiment when the current row of data was logged.',
        'setup_experiment.started': 'Timestamp for the start of the initial experiment setup routine.',
        'setup_experiment.stopped': 'Timestamp for the end of the initial experiment setup routine.',
        'instr_text.started': 'Timestamp for the start of the instruction text display.',
        'instr_resp.started': 'Timestamp for the start of the response period for the instructions.',
        'start_eeg.started': 'Timestamp for the start of the "waiting for scanner/trigger" routine.',
        'start_eeg.stopped': 'Timestamp for the end of the "waiting for scanner/trigger" routine.',
        'text_2.started': 'Timestamp for the onset of the "waiting for trigger" text prompt.',
        'key_resp_3.started': 'Timestamp for when the program started listening for the trigger key.',
        'load_images.started': 'Timestamp for the start of the image pre-loading routine (e.g., during a break).',
        'load_images.stopped': 'Timestamp for the end of the image pre-loading routine.',
        'text_3.started': 'Timestamp for the onset of the text component shown during a break.',
        'text_3.stopped': 'Timestamp for the offset of the text component shown during a break.',
        'preload_current_image_2.started': 'Timestamp for the start of the component responsible for pre-loading images during a break.',
        'fixation.started': 'Timestamp for the onset of the primary pre-stimulus fixation cross.',
        'fixation.stopped': 'Timestamp for the offset of the primary pre-stimulus fixation cross.',
        'fixation_2.started': 'Timestamp for the onset of a secondary fixation cross (e.g., during a break).',
        'fixation_2.stopped': 'Timestamp for the offset of a secondary fixation cross.',
        'trials_image.started': 'Timestamp for the start of the routine responsible for showing the image stimulus.',
        'trials_image.stopped': 'Timestamp for the end of the routine responsible for showing the image stimulus.',
        'stim_image.started': 'Timestamp for the precise onset of the image stimulus component.',
        'stim_image.stopped': 'Timestamp for the precise offset of the image stimulus component.',
        'trials_word.started': 'Timestamp for the start of the routine responsible for showing the word stimulus.',
        'trials_word.stopped': 'Timestamp for the end of the routine responsible for showing the word stimulus.',
        'stim_word.started': 'Timestamp for the precise onset of the word stimulus component.',
        'stim_word.stopped': 'Timestamp for the precise offset of the word stimulus component.',
        'blank.started': 'Timestamp for the start of the routine containing the post-stimulus blank screen.',
        'blank.stopped': 'Timestamp for the end of the routine containing the post-stimulus blank screen.',
        'blank_interval.started': 'Timestamp for the onset of the post-stimulus blank screen component.',
        'blank_interval.stopped': 'Timestamp for the offset of the post-stimulus blank screen component.',
        'response.started': 'Timestamp for the start of the routine containing the response window.',
        'response.stopped': 'Timestamp for the end of the routine containing the response window.',
        'key_resp.started': 'Timestamp for the start of the response collection window.',
        'text.started': 'Timestamp for the onset of any text component within the response routine.',
        'feedback_routine.started': 'Timestamp for the start of the feedback display routine.',
        'feedback_routine.stopped': 'Timestamp for the end of the feedback display routine.',
        'iti_routine.started': 'Timestamp for the start of the inter-trial interval (ITI) routine.',
        'iti_routine.stopped': 'Timestamp for the end of the ITI routine.',
        'iti_fixation.started': 'Timestamp for the onset of the fixation cross during the ITI.',
        'iti_fixation.stopped': 'Timestamp for the offset of the fixation cross during the ITI.',
        'break_2.started': 'Timestamp for the start of a break routine within the experiment.',
        'break_2.stopped': 'Timestamp for the end of a break routine.',
        'timer_text.started': 'Timestamp for the onset of the break countdown timer text.',
        'finish_time.started': 'Timestamp for when the program started listening for a pause/end key press.',
        'end.started': 'Timestamp for the start of the final "end of experiment" routine.',
        'end.stopped': 'Timestamp for the end of the final "end of experiment" routine.',
        'end_text.started': 'Timestamp for the onset of the final "thank you" or "end" text.',
        'end_text.stopped': 'Timestamp for the offset of the final "thank you" or "end" text.',
        'key_resp_2.started': 'Timestamp for when the program started listening for a key press to end the experiment.',
        'key_resp_2.stopped': 'Timestamp for when the participant pressed a key to end the experiment.',

        # == Feedback-related columns ==
        'feedback_message_to_show': 'The content of the feedback message prepared for display (may not have been shown).',
        'show_feedback_flag_value': 'A boolean flag indicating if feedback was scheduled to be shown for the trial.',

        # == General Experiment & Session Metadata ==
        'notes': 'Any manual notes recorded by the experimenter during the session.',
        'participant': 'The participant identifier.',
        'session': 'The session number or identifier.',
        'date': 'The date and time the experiment was run, in YYYY-MM-DD_HHhMM.SS.ms format.',
        'expName': 'The name of the PsychoPy experiment.',
        'psychopyVersion': 'The version of PsychoPy used to run the experiment.',
        'frameRate': 'The measured frame rate of the monitor in Hz during the experiment.',
        'match_key': 'The specific keyboard key assigned to "match" responses.',
        'mismatch_key': 'The specific keyboard key assigned to "mismatch" responses.',
        'allowed_keys_list': 'A list of keyboard keys that were accepted as valid participant responses.',
        'expStart': 'The start date and time of the experiment session with timezone information.',
        'event_type' : "which even took place during the experiment",
        
    }
    
    #add even type to the func_loc_meta data
    import itertools

    labels = ['fix', 'stim1', 'word', 'blank', 'resp']
    df_expanded_funcloc['event_type'] = list(itertools.islice(
        itertools.cycle(labels), len(df_expanded_funcloc)))
        
    #drop the unadmed:103 column from the data frame 
    if 'Unnamed: 103' in df_expanded_funcloc.columns:
        df_expanded_funcloc = df_expanded_funcloc.drop(columns=['Unnamed: 103'])
    df_expanded_funcloc = df_expanded_funcloc.reset_index(drop=True)
    #it is important to reset_index so it matches index in the orignal eeg files 
    #WE should now have all the meta data that we will add to func_loc


    ############################################################################
    # These lines are optional, they were used to check if all stims were aligned 
    #############################################################################
    #change check_up to true if you want to check for alignment
    check_up_neccessary = False
    if check_up_neccessary == True :
        check_up = func_df_sub['stim_image.started']
        stim = np.array(raw_func_loc.annotations.description) == 'stim1'
        stim_onsets = raw_func_loc.annotations.onset[stim]
        stim_seconds = stim_onsets.tolist()

        check_up_2 = func_df_sub['fixation_2.started']
        fix = np.array(raw_func_loc.annotations.description) == 'fix'
        fix_onsets = raw_func_loc.annotations.onset[fix]
        fix_seconds = fix_onsets.tolist()


        diff_seconds_fix = check_up_2 - fix_seconds
        diff_seconds_stim = check_up - stim_seconds 

        check_up_2 = check_up_2 - diff_seconds_fix
        check_up = check_up - diff_seconds_stim
        #add all these variables to 1 big data frame
        new_df = pd.DataFrame({
            'fix_psychopy' : check_up_2,
            'fix_eeg': fix_seconds,
            'stim_eeg': stim_seconds,
            'stim_psychopy' : check_up,
        })
        if not np.allclose(new_df['fix_psychopy'], new_df['fix_eeg']):
            print(f"Fixation times do not match for {sub.name}!")
        else : 
            print(f"Fixation times match for {sub.name}!")
        

    #we now can get the meta data from our cued stim
    cued_df_sub = cued_huge[cued_huge['sub'] == sub.name].reset_index(drop=True)
    cued_df_sub = cued_df_sub.drop(columns=['sub'])

    #we also need to match the ammounts of triggers do the event descriptions + meta data (4 triggers + 1 start recording)
    n_repeats  = 4
    new_index = np.repeat(cued_df_sub.index, n_repeats)
    expanded_cued_df = cued_df_sub.loc[new_index]
    

    #add the even type to our df - start with the label strategy we used earlier
    labels = ['word','fix','stim1','resp',]
    expanded_cued_df['event_type'] = list(itertools.islice(
        itertools.cycle(labels), len(expanded_cued_df)))

    #drop the unnamed:93
    if 'Unnamed: 93' in expanded_cued_df.columns:
        expanded_cued_df = expanded_cued_df.drop(columns=['Unnamed: 93'])
    #reset index 
    expanded_cued_df = expanded_cued_df.reset_index(drop=True)
    

    #we can add the event dictionnary
    extra_event_descriptions_cued = {
    # == Custom Column for BIDS ==
    'event_type': 'The type of event/trigger from the EEG annotations (e.g., fix, cue, probe).',

    # == Core Trial & Condition Information ==
    'block': 'The block number for the current trial.',
    'cue_direction': 'The direction of the cued mental replay task (e.g., "forward", "backward").',
    'cue_text': 'The text content of the cue presented to the participant (e.g., "1 ->", "<- 4").',
    'probe_image_file': 'The image file shown as a probe after the mental replay period.',
    'correct_response': 'The correct key press expected for the probe image ("1" for mismatch, "2" for match).',

    # == Participant Response Data ==
    'key_resp_probe.keys': 'The key pressed by the participant in response to the probe image.',
    'key_resp_probe.corr': 'Indicates if the probe response was correct (1) or incorrect (0).',
    'key_resp_probe.rt': 'Reaction time in seconds for the response to the probe image.',
    'key_resp_probe.duration': 'Duration in seconds for which the probe response key was held down.',
    'rating_vividness.response': 'The vividness rating provided by the participant (e.g., on a scale of 1-4).',
    'rating_vividness.rt': 'Time taken in seconds to provide the vividness rating, measured from the start of the rating scale display.',
    'key_resp_rating.keys': 'The key used to confirm and submit the vividness rating.',
    'key_resp_rating.rt': 'Time taken in seconds to confirm the rating after it was selected.',
    'key_resp_rating.duration': 'Duration the rating confirmation key was held down.',
    'key_resp_break.keys': 'The key pressed to continue after a block break.',
    'key_resp_break.rt': 'Time taken in seconds to press the key to continue after a break.',
    'key_resp_break.duration': 'Duration the continue-after-break key was held down.',
    'quit_instr.keys': 'The key pressed to exit the initial instruction screen.',
    'quit_instr.rt': 'Time taken to exit the initial instruction screen.',
    'quit_instr.duration': 'Duration the instruction exit key was held down.',
    'exit_eeg_text.keys': 'The key pressed to proceed from the "waiting for trigger" screen.',
    'exit_eeg_text.rt': 'Time taken to proceed from the "waiting for trigger" screen.',
    'exit_eeg_text.duration': 'Duration the "waiting for trigger" exit key was held down.',
    'exit_end.keys': 'The key pressed to finally exit the experiment after the "end" text.',
    'exit_end.rt': 'Time taken to exit the final screen.',
    'exit_end.duration': 'Duration the final exit key was held down.',

    # == PsychoPy Loop & Trial Counters ==
    'thisN': 'The overall trial number within the entire experiment, 0-indexed.',
    'thisTrialN': 'The trial number within the current loop or block, 0-indexed.',
    'thisRepN': 'The repetition (block) number of the current loop, 0-indexed.',
    'trials_loop.thisRepN': 'Repetition number of the main trial loop (PsychoPy internal log).',
    'trials_loop.thisTrialN': 'Trial number within the current loop repetition (PsychoPy internal log).',
    'trials_loop.thisN': 'Overall trial number within the main trial loop (PsychoPy internal log).',
    'trials_loop.thisIndex': 'The 0-indexed number of the trial from the conditions file (PsychoPy internal log).',
    'trials_loop.key_resp_probe.keys': 'Key pressed for the probe, logged internally by the trial loop.',
    'trials_loop.key_resp_probe.corr': 'Correctness of probe response, logged internally by the trial loop.',
    'trials_loop.key_resp_probe.rt': 'Reaction time for the probe, logged internally by the trial loop.',
    'trials_loop.key_resp_probe.duration': 'Key press duration for the probe, logged internally by the trial loop.',
    'trials_loop.rating_vividness.response': 'Vividness rating response, logged internally by the trial loop.',
    'trials_loop.rating_vividness.rt': 'Vividness rating reaction time, logged internally by the trial loop.',
    'trials_loop.key_resp_rating.keys': 'Key press for rating confirmation, logged internally by the trial loop.',
    'trials_loop.key_resp_rating.rt': 'Reaction time for rating confirmation, logged internally by the trial loop.',
    'trials_loop.key_resp_rating.duration': 'Key press duration for rating confirmation, logged internally by the trial loop.',
    'trials_loop.key_resp_break.keys': 'Key press to end break, logged internally by the trial loop.',
    'trials_loop.key_resp_break.rt': 'Reaction time to end break, logged internally by the trial loop.',
    'trials_loop.key_resp_break.duration': 'Key press duration to end break, logged internally by the trial loop.',

    # == Timestamps & Durations (System-level) ==
    'thisRow.t': 'Time elapsed in seconds since the start of the experiment when this data row was logged.',
    'instr_task.started': 'Timestamp for the start of the main instruction routine.',
    'instr_task.stopped': 'Timestamp for the end of the main instruction routine.',
    'task_instr_text.started': 'Timestamp for the onset of the task instruction text component.',
    'quit_instr.started': 'Timestamp for when the program started listening for a key to exit instructions.',
    'start_eeg.started': 'Timestamp for the start of the "waiting for EEG/scanner trigger" routine.',
    'start_eeg.stopped': 'Timestamp for the end of the "waiting for EEG/scanner trigger" routine.',
    'EEG_start_text.started': 'Timestamp for the onset of the "starting EEG" text prompt.',
    'exit_eeg_text.started': 'Timestamp for when the program started listening for a key to exit the trigger wait screen.',
    'image_load.started': 'Timestamp for the start of the image pre-loading routine.',
    'image_load.stopped': 'Timestamp for the end of the image pre-loading routine.',
    'image_load_text.started': 'Timestamp for the onset of the "loading images" text.',
    'imag_loader.started': 'Timestamp for the start of the specific image loader component.',
    'trial.started': 'Timestamp for the start of a single trial routine.',
    'trial.stopped': 'Timestamp for the end of a single trial routine.',
    'cue_stim.started': 'Timestamp for the onset of the cue stimulus (text).',
    'cue_stim.stopped': 'Timestamp for the offset of the cue stimulus (text).',
    'fixation_stim.started': 'Timestamp for the onset of the fixation cross during the mental replay period.',
    'fixation_stim.stopped': 'Timestamp for the offset of the fixation cross.',
    'probe_image.started': 'Timestamp for the onset of the probe image.',
    'probe_image.stopped': 'Timestamp for the offset of the probe image.',
    'key_resp_probe.started': 'Timestamp for when the program started listening for a probe response.',
    'resp_instr.started': 'Timestamp for the onset of the response instruction text.',
    'rating.started': 'Timestamp for the start of the rating routine.',
    'rating.stopped': 'Timestamp for the end of the rating routine.',
    'rating_vividness.started': 'Timestamp for the onset of the vividness rating scale component.',
    'text_rating_question.started': 'Timestamp for the onset of the rating question text.',
    'key_resp_rating.started': 'Timestamp for when the program started listening for the rating confirmation key.',
    'block_break.started': 'Timestamp for the start of a block break routine.',
    'block_break.stopped': 'Timestamp for the end of a block break routine.',
    'block_break_text.started': 'Timestamp for the onset of the break text.',
    'block_break_text.stopped': 'Timestamp for the offset of the break text.',
    'key_resp_break.started': 'Timestamp for when the program started listening for a key to end the break.',
    'end_task.started': 'Timestamp for the start of the final "end of task" routine.',
    'end_task.stopped': 'Timestamp for the end of the final "end of task" routine.',
    'end_text.started': 'Timestamp for the onset of the final "end" text.',
    'exit_end.started': 'Timestamp for when the program started listening for the final exit key.',
    
    # == General Experiment & Session Metadata ==
    'notes': 'Any manual notes recorded by the experimenter during the session.',
    'participant': 'The participant identifier.',
    'session': 'The session number or identifier.',
    'date': 'The date and time the experiment was run.',
    'expName': 'The name of the PsychoPy experiment.',
    'psychopyVersion': 'The version of PsychoPy used to run the experiment.',
    'frameRate': 'The measured frame rate of the monitor in Hz.',
    'sequence_good_key': 'The key assigned to indicate the probe image matched the cued sequence.',
    'sequence_bad_key': 'The key assigned to indicate the probe image did not match the cued sequence.',
    'allowed_keys_list': 'A list of keyboard keys accepted as valid participant responses.',
    'expStart': 'The start date and time of the experiment session with timezone information.'
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
    #raw_cued_stim_for_bids.set_channel_types({'HEOG': 'eog', 'VEOG': 'eog', 'ECG': 'ecg'})
    #raw_func_loc_for_bids.set_channel_types({'HEOG': 'eog', 'VEOG': 'eog', 'ECG': 'ecg'})
    if raw_postlearnrest is not None:
        raw_postlearnrest.set_channel_types({'HEOG': 'eog', 'VEOG': 'eog', 'ECG': 'ecg'})

    # write raw files to BIDS format for files without meta data


    #we will also pass an empty dictionnary for the raw_rest_state so that no events are kept

    rest_state_dic = {}
    write_raw_bids(
        raw_rest_state,
        bids_path_rest_state,
        event_id=rest_state_dic,
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
        raw=raw_func_loc,
        bids_path=bids_path_func_loc,
        event_metadata=df_expanded_funcloc,
        extra_columns_descriptions=extra_event_descriptions,
        event_id=event_id_1,
        overwrite=True,
        allow_preload=True,
        format='BrainVision'
    )

    event_id_cued = {
    'word': 1,
    'fix': 2,
    'stim1': 3,
    'resp': 4
    }

    write_raw_bids(
        raw=raw_cued_stim,
        bids_path=bids_path_cued_stim,
        event_metadata=expanded_cued_df,
        extra_columns_descriptions=extra_event_descriptions_cued,
        event_id=event_id_cued,
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



