import librosa
import matplotlib.pyplot as plt
import librosa.display


class AudioFeatures:
    def __init__(self, audio_path):
        self.x, self.sr = librosa.load(audio_path)

    def get_waveplot(self):
        pass

    def get_spectogram(self):
        pass

    def get_zero_crossing_rate(self):
        pass

    def get_spectral_centroid(self):
        pass

    def get_spectral_rolloff(self):
        pass

    def get_mfcc(self):
        pass

    def get_chroma_features(self):
        pass
