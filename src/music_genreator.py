from src.dataset_creator import DatasetsCreator
from src.testing import Testing
from src.training import Training


class MusicGenRator:

    def __init__(self):
        DatasetsCreator()
        self.training = Training()
        # self.training.generate_models()
        # self.training.plot_models()
        self.testing = Testing()

    def genrate_song(self, song_path):
        print('Testing a song:\t' + song_path)
        print(self.testing.compare_song_by_path(song_path))


# Testing single song
# test_song_path = \
# '/home/dj-d/Desktop/Repository/GitHub/MusicGenRetor/songs/Jazz/Mela_-_01_-_For_Such_a_Thing_to_Land.mp3'
music_genrator = MusicGenRator()
