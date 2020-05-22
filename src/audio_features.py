import matplotlib.pyplot as plt
from librosa import load, display, stft, amplitude_to_db, zero_crossings, feature, frames_to_time
from sklearn import preprocessing


class AudioFeatures:
    def __init__(self, audio_path):
        self.x, self.sr = load(audio_path)

    def plot_audio(self):
        plt.figure(figsize=(14, 5))
        display.waveplot(self.x, sr=self.sr)
        plt.show()

    def plot_spectogram(self):
        X = stft(self.x)
        Xdb = amplitude_to_db(abs(X))
        plt.figure(figsize=(14, 5))
        display.specshow(Xdb, sr=self.sr, x_axis='time', y_axis='hz')
        plt.show()

    def get_zero_crossing_rate(self):
        zero_crossing = zero_crossings(self.x, pad=False)

        return sum(zero_crossing)

    @staticmethod
    def normalize(x, axis=0):
        return preprocessing.minmax_scale(x, axis=axis)

    def plot_spectral_centroid(self):
        spectral_centroids = feature.spectral_centroid(self.x, sr=self.sr)[0]

        frames = range(len(spectral_centroids))
        t = frames_to_time(frames)

        display.waveplot(self.x, sr=self.sr, alpha=0.4)
        plt.plot(t, self.normalize(spectral_centroids), color='r')
        plt.show()

    def plot_spectral_rolloff(self):
        spectral_rolloff = feature.spectral_rolloff(self.x + 0.01, sr=self.sr)[0]

        frames = range(len(spectral_rolloff))
        t = frames_to_time(frames)

        display.waveplot(self.x, sr=self.sr, alpha=0.4)
        plt.plot(t, self.normalize(spectral_rolloff), color='r')
        plt.show()

    def plot_mfcc(self):
        mfccs = feature.mfcc(self.x, sr=self.sr)

        display.specshow(mfccs, sr=self.sr, x_axis='time')
        plt.show()

    def plot_perform_mfcc(self):
        mfccs = feature.mfcc(self.x, sr=self.sr)
        mfccs = preprocessing.scale(mfccs, axis=1)

        display.specshow(mfccs, sr=self.sr, x_axis='time')
        plt.show()

    def plot_chroma_features(self):
        hop_lenght = 512

        chrmomagram = feature.chroma_stft(self.x, sr=self.sr, hop_length=hop_lenght)

        plt.figure(figsize=(14, 5))
        display.specshow(chrmomagram, x_axis='time', y_axis='chroma', hop_length=hop_lenght, cmap='coolwarm')
        plt.show()
