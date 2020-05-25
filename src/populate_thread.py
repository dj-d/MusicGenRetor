from threading import Thread
from os import path, walk, listdir
import os
import pandas as pd
from src.audio_features import AudioFeatures


class PopulateThread(Thread):
    def __init__(self, id, data_path, directory, index, data_frame):
        Thread.__init__(self)

        self.id = id

        self.data_path = data_path
        self.directory = directory

        self.index = index
        self.data_frame = data_frame

    def load_song(self, path):
        audio_features = AudioFeatures(path)

        title = os.path.basename(path)
        bpm = audio_features.get_bpm()
        zero_crossing_rate = audio_features.get_zero_crossing_rate()
        audio_time_series = audio_features.get_audio_time_series()

        song = pd.Series([title, bpm, zero_crossing_rate, audio_time_series], index=self.index)
        self.data_frame = self.data_frame.append(song, ignore_index=True)

        self.id += 1

    def run(self):
        subdir_path = path.join(self.data_path, self.directory)

        walk(subdir_path)

        print('\n---------- ' + self.directory + ' ----------\n')

        files = listdir(subdir_path)

        for file in files:
            print(file)

            file_path = path.join(subdir_path, file)
            self.load_song(file_path)
