from os import path, walk, listdir, system
from pydub import AudioSegment

script_path = path.dirname(path.realpath(__file__))
dir_path = path.abspath(script_path + "/..") + "/old"

for root, directories, files in walk(dir_path):
    for directory in directories:
        subdir_path = path.join(dir_path, directory)
        walk(subdir_path)

        print('-------------' + directory + '-------------')

        files = listdir(subdir_path)

        for file in files:
            dst = file.replace('mp3', 'wav')

            sound = AudioSegment.from_mp3(file)
            sound.export(dst, format='wav')
