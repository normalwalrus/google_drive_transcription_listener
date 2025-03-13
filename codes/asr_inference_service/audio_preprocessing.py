import logging
import os
import tempfile

import librosa
from moviepy.video.io.VideoFileClip import VideoFileClip

logging.basicConfig(
    format="%(levelname)s | %(asctime)s | %(message)s", level=logging.INFO
)


def resample_audio_filepath(audio_filepath, desired_sr):
    """
    Using Librosa, converts an audio file to mono and 
    converts the samplerate of the audio to the desired samplerate
    """

    logging.info("Audio preprocessing started : To %s SR", desired_sr)

    y, sr = librosa.load(audio_filepath)
    logging.info("Audio loaded ")
    y_desired = librosa.resample(y, orig_sr=sr, target_sr=desired_sr)
    y_mono = librosa.to_mono(y_desired)

    logging.info("Audio preprocessed, Shape : %s", y_mono.shape)

    return y_mono


def resample_audio_array(y, original_sr, desired_sr):
    """
    Using Librosa, converts an audio array to mono and 
    converts the samplerate of the audio to the desired samplerate
    """

    logging.info("Audio preprocessing starte : %s SR to %s SR", original_sr, desired_sr)

    y_mono = librosa.to_mono(y)
    # print(y_mono)
    y_desired = librosa.resample(y_mono, orig_sr=original_sr, target_sr=desired_sr)
    # print(y_desired)
    logging.info("Resample done")

    logging.info("Audio preprocessed, Shape : %s", y_desired.shape)

    return y_desired


def get_numpy_array_from_mp4(mp4_bytes):
    """
    Gets a numpy array from Byte class from an mp4 file
    """

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
        temp_file.write(mp4_bytes)
        temp_file_path = temp_file.name

    video_clip = VideoFileClip(temp_file_path)

    if not video_clip.audio:
        return {"error": "No audio found in the MP4 file"}

    audio_clip = video_clip.audio
    audio_array = audio_clip.to_soundarray()
    sample_rate = audio_clip.fps

    audio_clip.close()
    video_clip.close()

    os.remove(temp_file_path)

    audio_array = audio_array.T
    print(audio_array)

    return audio_array, sample_rate
