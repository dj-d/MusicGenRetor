from src.logic import Logic

logic = Logic()

logic.generate_models()
logic.plot_models()
song_path = '/home/zappaboy/Desktop/Projects/MusicGenRetor/songs/Jazz/Mela_-_01_-_For_Such_a_Thing_to_Land.mp3'
results = logic.compare_song(song_path)

print(results)
