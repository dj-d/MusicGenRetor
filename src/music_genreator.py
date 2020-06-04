from src.dataset_creator import DatasetsCreator
from src.testing import Testing
from src.training import Training


class MusicGenRator:

    def __init__(self, song_path):
        DatasetsCreator()
        training = Training()
        training.generate_models()
        testing = Testing()
        training.plot_models()

        print('Testing a song:\t' + song_path)
        print(testing.compare_song_by_path(song_path))


# Testing single song
test_song_path = '/home/zappaboy/Desktop/Projects/MusicGenRetor/songs/Jazz/Mela_-_01_-_For_Such_a_Thing_to_Land.mp3'
music_genrator = MusicGenRator(test_song_path)
