import torch
from torch import nn
import torch.nn.functional as F
from transformers import WhisperProcessor, WhisperForAudioClassification
import transformers
import pandas as pd
import ast
import librosa
import numpy as np


PATH = '/home/jovyan/data/wise/MASCoT-CP1.0'
transformers.utils.logging.set_verbosity_error()

class AudioTokenClassifier:
    def __init__(self, model_path):
        self.code_dict = {'ist': 0, 'st': 1, 'aotr': 2, 'sotr': 3, 'rep': 4, 'red': 5, 'gprs': 6, 'bsp': 7, 'aaff': 8, 'acorr': 9, 'sv': 10, 'neu': 11, 'uni': 12, 'sil': 13, 'rv': 14, 'sot': 15}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        self.processor = WhisperProcessor.from_pretrained("openai/whisper-base")
        base_model = WhisperForAudioClassification.from_pretrained("openai/whisper-base")
        class NeuralNetwork(nn.Module):
            def __init__(self, num_labels):
                super().__init__()
                self.encoder = base_model.encoder
                self.projector = nn.Linear(512, 256)
                self.classifier = nn.Linear(256, num_labels)
        
            def forward(self, x):
                x = F.relu(self.encoder(x).last_hidden_state)
                x = F.relu(self.projector(x))
                x = F.sigmoid(self.classifier(x))
                return x
        self.model = NeuralNetwork(num_labels=16).to(self.device)
        self.model.load_state_dict(torch.load(self.model_path, weights_only=True, map_location=torch.device(self.device)))

    def get_clips_from_file(self, audio_path, window_size=30, step=15, rate=16000, add_labels=False):
        clip_name = audio_path.split('/')[-1].split('.')[0]
        clips = []
        audio, sr = librosa.load(path=audio_path, sr=rate)
        counter = 0    
        # Iterate over steps in each clip
        for i in np.arange(0, len(audio), rate*step):
            start_time = int(i)
            end_time = int(i+(window_size*rate))
            if end_time < len(audio):
                features = self.model(self.processor(audio[start_time:end_time], sampling_rate=16000, return_tensors='pt').input_features.to(self.device))
                clips.append({'index': counter, 'id':clip_name, 'start_time': start_time/rate, 'end_time': end_time/rate, 'features': features})
            counter += 1    
        return clips

    def get_predictions(self, audio_path):
        clips = self.get_clips_from_file(audio_path)
        features = [x['features'] for x in clips]
        results = {'rater1':[], 'rater2':[None]*750}
        # stack the two lists side by side
        for i, clip in enumerate(features):
            res = list(clip[0].cpu().detach().numpy())
            if i == 0:
                results['rater2'] = res[:750]
            if i%2 == 0:
                results['rater1'] = results['rater1'] + res[:750]
                results['rater1'] = results['rater1'] + res[750:]
            else:
                results['rater2'] = results['rater2'] + res[:750]
                results['rater2'] = results['rater2'] + res[750:]
        # fill in the shorter list
        if len(results['rater1']) > len(results['rater2']):
            results['rater2'] = results['rater2'] + results['rater1'][-750:]
        else:
            results['rater1'] = results['rater1'] + results['rater2'][-750:]
        df = pd.DataFrame(results)
        # average the two lists together
        def get_average(row):
            return np.mean(np.array([row['rater1'], row['rater2']]), axis=0)
        logits = df.apply(lambda row: get_average(row), axis=1)
        # make a list of timestamps
        time_list = []
        x = 0
        while len(time_list) < df.shape[0]:
            time_list.append(round(x, 2))
            x += 0.02
        # predict most likely cp
        preds = logits.apply(lambda x: np.argmax(x))
        # predict whether teacher talk
        teacher_talk = ~preds.isin([10, 13])
        # return df
        return pd.DataFrame({'time':time_list, 'logits':logits, 'preds':preds, 'teacher_talk':teacher_talk})

    def diarize(self, audio_path):
        prediction_df = self.get_predictions(audio_path)
        prediction_df['prev_label'] = prediction_df['teacher_talk'].shift()
        prediction_df['is_start'] = prediction_df['teacher_talk'] != prediction_df['prev_label']
        # Get the start indices
        start_times = prediction_df[prediction_df['is_start']][['time', 'teacher_talk']]
        start_times_list = start_times['time'].to_list()
        # # Add the last index as the end of the final run
        end_times = start_times_list[1:] + [start_times_list[-1] + 1]
        start_times['end_times']=end_times
        start_times = start_times[['time', 'end_times', 'teacher_talk']]
        start_times.columns = ['start_time', 'end_time', 'teacher_talk']
        return start_times