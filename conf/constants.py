SONGS_PATH = ""

DATASET_DIRECTORY = "Datasets" + "/"

TRAINING_DATASET_PATH = DATASET_DIRECTORY + "Training" + "/"
TESTING_DATASET_PATH = DATASET_DIRECTORY + "Testing" + "/"

MODELS_PATH = "Models" + "/"

DATASET_BASE_NAME = "Dataset"

# ATTRS = ['Title', 'bpm', 'zero_crossing_rate', 'audio_time_series', 'genre']
ATTRS = ['Title', 'bpm', 'spectrogram', 'zero_crossing_rate', 'spectral_centroid', 'spectral_bandwidth',
         'spectral_rolloff', 'mfcc', 'perform_mfcc', 'chroma_frequencies', 'genre']

GENRES = ['Blues', 'Electronic', 'Classical', 'Pop', 'Rock', 'Jazz']

SR = 22100

ROWS = 20
COLUMNS = 49
