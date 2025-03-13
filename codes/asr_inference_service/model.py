"""ASR Inference Model Class"""

import logging
from time import perf_counter

import librosa
import numpy as np
import torch

from codes.asr_inference_service.asr_model import FasterWhisperASR, WhisperASR
from codes.asr_inference_service.diarizer import PyannoteDiarizer

from moviepy.video.io.VideoFileClip import VideoFileClip

logging.basicConfig(
    format="%(levelname)s | %(asctime)s | %(message)s", level=logging.INFO
)

logging.getLogger("nemo_logger").setLevel(logging.ERROR)


class ASRModelForInference:
    """Base class for ASR model for inference"""

    def __init__(
        self,
        model_dir: str,
        sample_rate: int = 16000,
        device: str = "cpu",
        timestamp_format: str = "seconds",
        min_segment_length=0.5,
        min_silence_length=0,
    ):
        """
        Inputs:
            model_dir (str): path to model directory
            sample_rate (int): the target sample rate in which the model accepts
        """

        device = (
            device
            if device in ["cuda", "cpu"]
            else "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.device_number = [0] if device == "cuda" else 1
        self.accelerator = "gpu" if device == "cuda" else "cpu"

        self.asr_model = WhisperASR(model_dir, sample_rate, device)
        # self.asr_model = FasterWhisperASR(model_dir, sample_rate, device)

        self.timestamp_format = (
            timestamp_format
            if timestamp_format in ["minutes", "seconds","hour-minute-second"]
            else "seconds"
        )
        logging.info("Running on device: %s", device)
        self.target_sr = sample_rate

        self.diar_model = PyannoteDiarizer(
            device=device,
            min_segment_length=min_segment_length,
            min_silence_length=min_silence_length,
        )

    def load_audio(self, audio_filepath: str) -> np.ndarray:
        """Method to load an audio filepath to generate a waveform, it automatically
        standardises the waveform to the target sample rate and channel

        Inputs:
            audio_filepath (str): path to the audio file

        Returns:
            waveform (np.ndarray) of shape (T,)
        """
        waveform, _ = librosa.load(audio_filepath, sr=self.target_sr, mono=True)

        return waveform

    def infer(self, filepath: str):
        """
        Infer from filepath method
        """

        ar, sr = self.load_audio(filepath)
        transcription = self.asr_model.infer(ar, sr)

        return transcription

    def diar_inference(self, filepath: str):
        """
        Method to call vad methods and using segments 
        of speech to transcribe using the infer method

        Inputs:
            waveform (np.ndarray): Takes in waveform of shape (T,)
            input_sr (int): Sample rate of input waveform

        Returns:
            final_transcription (str): transcription with timestamps attached to it
        """
        diarizer_start = perf_counter()
        logging.info("Diarization Model triggered.")

        segments = self.diar_model.diarize(filepath)
        waveform = self.load_audio(filepath)

        diarizer_end = perf_counter()
        logging.info(
            "Diarization Model Done. Elapsed time: %s",
            diarizer_end - diarizer_start,
        )

        final_transcription = ""

        for x in range(len(segments)):
            start_time = segments["start_time"][x]
            end_time = segments["end_time"][x]

            start_frame = int(start_time * self.target_sr)
            end_frame = int(end_time * self.target_sr)

            split_audio = waveform[start_frame:end_frame]

            transcription = self.asr_model.infer(split_audio, self.target_sr)

            if self.timestamp_format == "minutes":
                start_time = start_time / 60
                end_time = end_time / 60
                segment_string = f"[{start_time:.2f} - {end_time:.2f}] [{segments['speaker'][x]}] : {transcription}\n\n"
            elif self.timestamp_format == "hour-minute-second":
                start_time = int(start_time)
                end_time = int(end_time)
                
                start_hours = start_time // 3600
                start_minutes = (start_time % 3600) // 60
                start_seconds = start_time % 60
                
                end_hours = end_time // 3600
                end_minutes = (end_time % 3600) // 60
                end_seconds = end_time % 60
                
                segment_string = f"[{start_hours:02d}:{start_minutes:02d}:{start_seconds:02d} - {end_hours:02d}:{end_minutes:02d}:{end_seconds:02d}] [{segments['speaker'][x]}] : {transcription}\n\n"       
            else:
                segment_string = f"[{start_time:.2f} - {end_time:.2f}] [{segments['speaker'][x]}] : {transcription}\n\n"


            final_transcription = "".join([final_transcription, segment_string])

        return final_transcription
        
        


if __name__ == "__main__":
    pass
