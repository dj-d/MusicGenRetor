import warnings

import matplotlib.pyplot as plt
from librosa import load, stft, amplitude_to_db, zero_crossings, frames_to_time
from librosa.beat import tempo
from librosa.display import waveplot, specshow
from librosa.feature import spectral_centroid, spectral_bandwidth, spectral_rolloff, mfcc, chroma_stft
from librosa.onset import onset_strength
from sklearn.preprocessing import minmax_scale, scale


class AudioFeatures:
    """
    Provide methods for audio visualization and audio info
    """

    def __init__(self, audio_path=None):
        """
        Class constructor

        :param audio_path: path of audio file
        """

        warnings.filterwarnings('ignore')

        if audio_path is not None:
            self.y, self.sr = load(path=audio_path)

    def select_series(self, y):
        """

        :param y:
        :return:
        """

        if y is not None:
            return y
        else:
            return self.y

    def select_sr(self, sr):
        """

        :param sr:
        :return:
        """

        if sr is not None:
            return sr
        else:
            return self.sr

    def get_audio_time_series(self):
        """
        Get all audio points

        :return: Numpy Array
        """

        return self.y

    def plot_audio(self, outside_series=None, outside_sr=None):
        """
        Plot audio

        :param outside_series:
        :param outside_sr:
        :return:
        """

        y = self.select_series(outside_series)
        sr = self.select_sr(outside_sr)

        plt.figure(figsize=(14, 5))
        waveplot(y, sr=sr)
        plt.show()

    def plot_zoomed_audio(self, i, j, outside_series=None):
        """
        Plot audio of in a range [i, j]

        :param outside_series:
        :param i: start point
        :param j: end point
        """

        y = self.select_series(outside_series)

        plt.figure(figsize=(14, 5))
        plt.plot(y[i:j])
        plt.grid()
        plt.show()

    def get_spectrogram_db(self, outside_series=None):
        y = self.select_series(outside_series)

        return amplitude_to_db(abs(stft(y)))

    def plot_spectrogram(self, outside_series=None, outside_sr=None):
        """
        Plot spectrogram of frequencies of audio

        :param outside_series:
        :param outside_sr:
        :return:
        """

        sr = self.select_sr(outside_sr)

        x_db = self.get_spectrogram_db(outside_series)

        plt.figure(figsize=(14, 5))
        specshow(x_db, sr=sr, x_axis='time', y_axis='hz')
        plt.show()

    def get_zero_crossing_rate(self, outside_series=None):
        """
        Return the number of times the signal changes sign

        :return: Integer
        """

        y = self.select_series(outside_series)

        zero_crossing = zero_crossings(y, pad=False)

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

    def get_spectral_centroid(self, outside_series=None, outside_sr=None):
        """

        :return:
        """

        y = self.select_series(outside_series)
        sr = self.select_sr(outside_sr)

        return spectral_centroid(y, sr=sr)[0]

    def plot_spectral_centroid(self, outside_series=None, outside_sr=None):
        """
        Plot weighted average of the frequencies present in the sound

        :param outside_series:
        :param outside_sr:
        :return:
        """

        y = self.select_series(outside_series)
        sr = self.select_sr(outside_sr)

        spectral_centroids = self.get_spectral_centroid(outside_series, outside_sr)

        frames = range(len(spectral_centroids))
        t = frames_to_time(frames)

        waveplot(y, sr=sr)
        plt.plot(t, self.normalize(spectral_centroids), color='r')
        plt.show()

    def get_spectral_bandwidth(self, outside_series=None, outside_sr=None):
        """

        :param outside_series:
        :param outside_sr:
        :return:
        """

        y = self.select_series(outside_series)
        sr = self.select_sr(outside_sr)

        return spectral_bandwidth(y, sr=sr)

    def get_spectral_rolloff(self, outside_series=None, outside_sr=None):
        y = self.select_series(outside_series)
        sr = self.select_sr(outside_sr)

        return spectral_rolloff(y + 0.01, sr=sr)[0]

    # TODO: Add DOC
    def plot_spectral_rolloff(self, outside_series=None, outside_sr=None):
        """

        :param outside_series:
        :param outside_sr:
        :return:
        """

        y = self.select_series(outside_series)
        sr = self.select_sr(outside_sr)

        spectral_rolloffs = self.get_spectral_rolloff(y, sr)

        frames = range(len(spectral_rolloffs))
        t = frames_to_time(frames)

        waveplot(y, sr=sr)
        plt.plot(t, self.normalize(spectral_rolloffs), color='r')
        plt.show()

    def get_mfcc(self, outside_series=None, outside_sr=None):
        y = self.select_series(outside_series)
        sr = self.select_sr(outside_sr)

        return mfcc(y, sr=sr)

    def plot_mfcc(self, outside_series=None, outside_sr=None):
        """
        MFCC - Mel Frequency Cepstral Coefficients

        Plot small set of features that concisely describe the overall shape of a spectral envelope (curve tangent to a family of curves)

        :param outside_series:
        :param outside_sr:
        :return:
        """

        y = self.select_series(outside_series)
        sr = self.select_sr(outside_sr)

        mfccs = self.get_mfcc(y, sr)

        specshow(mfccs, sr=sr, x_axis='time')
        plt.show()

    def get_perform_mfcc(self, outside_series=None, outside_sr=None):
        y = self.select_series(outside_series)
        sr = self.select_sr(outside_sr)

        mfccs = mfcc(y, sr=sr)

        return scale(mfccs, axis=1)

    # TODO: Add DOC
    def plot_perform_mfcc(self, outside_series=None, outside_sr=None):
        """

        :param outside_series:
        :param outside_sr:
        :return:
        """

        y = self.select_series(outside_series)
        sr = self.select_sr(outside_sr)

        mfccs = self.get_perform_mfcc(y, sr)

        specshow(mfccs, sr=sr, x_axis='time')
        plt.show()

    def plot_perform_mfcc_by_values(self, mfccs=None, outside_sr=None):
        """

        :param mfccs:
        :param outside_sr:
        :return:
        """

        mfccs = scale(mfccs, axis=1)
        sr = self.select_sr(outside_sr)

        specshow(mfccs, sr=sr, x_axis='time')
        plt.show()

    def get_chroma_frequencies(self, outside_series=None, outside_sr=None):
        y = self.select_series(outside_series)
        sr = self.select_sr(outside_sr)

        hop_length = 512

        return hop_length, chroma_stft(y, sr=sr, hop_length=hop_length)

    def plot_chroma_frequencies(self, outside_series=None, outside_sr=None):
        """
        Plot audio where the entire spectrum is projected on 12 bins representing the 12 semitones of the musical octave

        :param outside_series:
        :param outside_sr:
        :return:
        """

        y = self.select_series(outside_series)
        sr = self.select_sr(outside_sr)

        get_chroma_frequencies = self.get_chroma_frequencies(y, sr)

        hop_length = get_chroma_frequencies[0]

        chroma_gram = get_chroma_frequencies[1]

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
