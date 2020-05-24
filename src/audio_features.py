import matplotlib.pyplot as plt

from librosa import load, stft, amplitude_to_db, zero_crossings, frames_to_time
from librosa.beat import tempo
from librosa.display import waveplot, specshow
from librosa.feature import spectral_centroid, spectral_rolloff, mfcc, chroma_stft
from librosa.onset import onset_strength
from librosa.util import example_audio_file

from sklearn.preprocessing import minmax_scale, scale

import warnings


class AudioFeatures:
    """
    Provide methods for audio visualization and audio info
    """

    def __init__(self, audio_path=example_audio_file()):
        """
        Class constructor

        :param audio_path: path of audio file
        """

        warnings.filterwarnings('ignore')

        self.y, self.sr = load(path=audio_path)

    def get_audio_time_series(self):
        return self.y

    def plot_audio(self):
        """
        Plot audio
        """

        plt.figure(figsize=(14, 5))
        waveplot(self.y, sr=self.sr)
        plt.show()

    # TODO: To improve
    def plot_zoomed_audio(self, i, j):
        """
        Plot audio of in a range [i, j]

        :param i: start point
        :param j: end point
        """

        plt.figure(figsize=(14, 5))
        plt.plot(self.y[i:j])
        plt.grid()
        plt.show()

    def plot_spectrogram(self):
        """
        Plot spectrogram of frequencies of audio
        """

        x_db = amplitude_to_db(abs(stft(self.y)))

        plt.figure(figsize=(14, 5))
        specshow(x_db, sr=self.sr, x_axis='time', y_axis='hz')
        plt.show()

    def get_zero_crossing_rate(self):
        """
        Return the number of times the signal changes sign

        :return: Integer
        """

        zero_crossing = zero_crossings(self.y, pad=False)

        return sum(zero_crossing)

    # TODO: To improve
    def get_zoomed_zero_crossing_rate(self, i, j):
        """
        Return number of time the signal changes sign in a range [i, j]

        :param i: start point
        :param j: end point
        :return: Integer
        """
        zero_crossing = zero_crossings(self.y[i:j], pad=False)

        return sum(zero_crossing)

    # TODO: Add DOC
    @staticmethod
    def normalize(x, axis=0):
        """


        :param x: The data
        :param axis:
        :return:
        """

        return minmax_scale(x, axis=axis)

    def plot_spectral_centroid(self):
        """
        Plot weighted average of the frequencies present in the sound
        """

        spectral_centroids = spectral_centroid(self.y, sr=self.sr)[0]

        frames = range(len(spectral_centroids))
        t = frames_to_time(frames)

        waveplot(self.y, sr=self.sr)
        plt.plot(t, self.normalize(spectral_centroids), color='r')
        plt.show()

    # TODO: Add DOC
    def plot_spectral_rolloff(self):
        """

        """

        spectral_rolloffs = spectral_rolloff(self.y + 0.01, sr=self.sr)[0]

        frames = range(len(spectral_rolloffs))
        t = frames_to_time(frames)

        waveplot(self.y, sr=self.sr)
        plt.plot(t, self.normalize(spectral_rolloffs), color='r')
        plt.show()

    def plot_mfcc(self):
        """
        MFCC - Mel Frequency Cepstral Coefficients

        Plot small set of features that concisely describe the overall shape of a spectral envelope (curve tangent to a family of curves)
        """

        mfccs = mfcc(self.y, sr=self.sr)

        specshow(mfccs, sr=self.sr, x_axis='time')
        plt.show()

    # TODO: Add DOC
    def plot_perform_mfcc(self):
        """

        """

        mfccs = mfcc(self.y, sr=self.sr)
        mfccs = scale(mfccs, axis=1)

        specshow(mfccs, sr=self.sr, x_axis='time')
        plt.show()

    def plot_chroma_frequencies(self):
        """
        Plot audio where the entire spectrum is projected on 12 bins representing the 12 semitones of the musical octave
        """

        hop_length = 512

        chroma_gram = chroma_stft(self.y, sr=self.sr, hop_length=hop_length)

        plt.figure(figsize=(14, 5))
        specshow(chroma_gram, x_axis='time', y_axis='chroma', hop_length=hop_length, cmap='coolwarm')
        plt.show()

    # TODO: To improve the precision
    def get_bpm(self):
        """
        Get BPM (Beats Per Minute)

        :return: Integer
        """

        onset_env = onset_strength(self.y, sr=self.sr)

        return int(tempo(onset_envelope=onset_env, sr=self.sr)[0])
