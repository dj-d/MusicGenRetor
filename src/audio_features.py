import matplotlib.pyplot as plt
from librosa import load, display, stft, amplitude_to_db, zero_crossings, feature, frames_to_time, util, onset, beat
from sklearn import preprocessing


class AudioFeatures:
    def __init__(self, audio_path=util.example_audio_file()):
        self.x, self.sr = load(audio_path)

    def plot_audio(self):
        """
        Plot audio
        """

        plt.figure(figsize=(14, 5))
        display.waveplot(self.x, sr=self.sr)
        plt.show()

    def plot_spectogram(self):
        """
        Plot spectogram of frequencies of audio
        """

        x_db = amplitude_to_db(abs(stft(self.x)))

        plt.figure(figsize=(14, 5))
        display.specshow(x_db, sr=self.sr, x_axis='time', y_axis='hz')
        plt.show()

    def get_zero_crossing_rate(self):
        """
        Return the number of times the signal changes sign

        :return: Integer
        """

        zero_crossing = zero_crossings(self.x, pad=False)

        return sum(zero_crossing)

    @staticmethod
    def normalize(x, axis=0):
        """


        :param x:
        :param axis:
        :return:
        """

        return preprocessing.minmax_scale(x, axis=axis)

    def plot_spectral_centroid(self):
        """
        Plot weighted average of the frequencies present in the sound
        """

        spectral_centroids = feature.spectral_centroid(self.x, sr=self.sr)[0]

        frames = range(len(spectral_centroids))
        t = frames_to_time(frames)

        display.waveplot(self.x, sr=self.sr, alpha=0.4)
        plt.plot(t, self.normalize(spectral_centroids), color='r')
        plt.show()

    def plot_spectral_rolloff(self):
        """

        """

        spectral_rolloff = feature.spectral_rolloff(self.x + 0.01, sr=self.sr)[0]

        frames = range(len(spectral_rolloff))
        t = frames_to_time(frames)

        display.waveplot(self.x, sr=self.sr, alpha=0.4)
        plt.plot(t, self.normalize(spectral_rolloff), color='r')
        plt.show()

    def plot_mfcc(self):
        """
        MFCC - Mel Frequency Cepstral Coefficients

        Plot small set of features that concisely describe the overall shape of a spectral envelope (curve tangent to a family of curves)
        """

        mfccs = feature.mfcc(self.x, sr=self.sr)

        display.specshow(mfccs, sr=self.sr, x_axis='time')
        plt.show()

    def plot_perform_mfcc(self):
        """

        """

        mfccs = feature.mfcc(self.x, sr=self.sr)
        mfccs = preprocessing.scale(mfccs, axis=1)

        display.specshow(mfccs, sr=self.sr, x_axis='time')
        plt.show()

    def plot_chroma_features(self):
        """
        Plot audio where the entire spectrum is projected on 12 bins representing the 12 semitones of the musical octave
        """

        hop_lenght = 512

        chrmomagram = feature.chroma_stft(self.x, sr=self.sr, hop_length=hop_lenght)

        plt.figure(figsize=(14, 5))
        display.specshow(chrmomagram, x_axis='time', y_axis='chroma', hop_length=hop_lenght, cmap='coolwarm')
        plt.show()

    def get_bpm(self):
        """
        Get BPM (Beats Per Minute)

        :return: Integer
        """

        onset_env = onset.onset_strength(self.x, sr=self.sr)

        return beat.tempo(onset_envelope=onset_env, sr=self.sr)