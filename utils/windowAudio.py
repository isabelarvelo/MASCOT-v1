import librosa
import os
from pathlib import Path
from IPython.display import Audio
import numpy as np
import pandas as pd

def time_in_window(row, clip_start, clip_end, rate=16000):
    beginning_in_window = max(row['start_time'], clip_start/rate)
    end_in_window = min(row['end_time'], clip_end/rate)
    return end_in_window - beginning_in_window

def get_windows_from_file(audio_path, label_path, window_size=30, step=15, rate=16000, add_labels=False):
    label_df = pd.read_table(label_path, names=['start_time', 'end_time', 'labels'])
    clip_name = audio_path.split('/')[-1].split('_')[0]
    clips = []
    audio, sr = librosa.load(path=audio_path, sr=rate)
    counter = 0    
    for i in np.arange(0, len(audio), rate*step):
        start_time = int(i)
        end_time = int(i+(window_size*rate))
        clip = audio[start_time:end_time]
        if add_labels:
            labels = label_df[((start_time/rate <= label_df['end_time']) & (start_time/rate >= label_df['start_time'])) | ((end_time/rate <= label_df['end_time']) & (end_time/rate >= label_df['start_time'])) | ((start_time/rate <= label_df['start_time']) & (end_time/rate >= label_df['end_time']))]  
            labels_with_duration = labels.copy()
            labels_with_duration.loc[:, 'duration'] = labels.apply(lambda row: time_in_window(row, start_time, end_time), axis=1)
            label = labels_with_duration.sort_values(by='duration')['labels'].iloc[0]        
            clips.append({'index': counter, 'id':clip_name, 'start_time': start_time/rate, 'end_time': end_time/rate, 'clip': clip, 'labels': label, })
        else:
            clips.append({'index': counter, 'id':clip_name, 'start_time': start_time/rate, 'end_time': end_time/rate, 'clip': clip,})
        counter += 1    
    return clips

def get_windows_from_clip(audio, label_path, window_size=0.02, step=0.02, rate=16000, add_labels=True):
    label_df = pd.read_table(label_path, names=['start_time', 'end_time', 'labels'])
    counter = 0 
    clips = []
    clip_name = label_path.split('/')[-1].split('_')[0]
    for i in np.arange(0, len(audio), rate*step):
        start_time = int(i)
        end_time = int(i+(window_size*rate))
        clip = audio[start_time:end_time]
        if add_labels:
            labels = label_df[((start_time/rate <= label_df['end_time']) & (start_time/rate >= label_df['start_time'])) | ((end_time/rate <= label_df['end_time']) & (end_time/rate >= label_df['start_time'])) | ((start_time/rate <= label_df['start_time']) & (end_time/rate >= label_df['end_time']))]  
            labels_with_duration = labels.copy()
            labels_with_duration.loc[:, 'duration'] = labels.apply(lambda row: time_in_window(row, start_time, end_time), axis=1)
            label = labels_with_duration.sort_values(by='duration')['labels'].iloc[0]        
            clips.append({'index': counter, 'id':clip_name, 'start_time': start_time/rate, 'end_time': end_time/rate, 'clip': clip, 'labels': label, })
        else:
            clips.append({'index': counter, 'id':clip_name, 'start_time': start_time/rate, 'end_time': end_time/rate, 'clip': clip,})
        counter += 1    
    return clips