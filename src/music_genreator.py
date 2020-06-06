from src.dataset_creator import DatasetsCreator
from src.random_forest import RandomForest


class MusicGenRator:

    def __init__(self):
        DatasetsCreator()
        # training = Training()
        # training.generate_models()
        # testing = Testing()
        # training.plot_models()
        random_forest = RandomForest()
        # random_forest.analyzing()
        random_forest.perform_test()

    def genrate_song(self, song_path):
        print('Testing a song:\t' + song_path)
        # print(self.testing.compare_song_by_path(song_path))


# Testing single song
# test_song_path = '/home/zappaboy/Desktop/Projects/MusicGenRetor/songs/Jazz/Mela_-_01_-_For_Such_a_Thing_to_Land.mp3'
music_genrator = MusicGenRator()
# music_genrator.genrate_song(test_song_path)
