# from nemo.collections.asr.models.msdd_models import NeuralDiarizer
# from nemo.utils import nemo_logging
import logging

import pandas as pd
import torch
from pyannote.audio import Pipeline

logger_nemo = logging.getLogger("nemo_logger")
logger_nemo.disabled = True

# class NemoDiarizer:
#     '''
#     Nemo Diar inference class
#     '''
#     def __init__(self,
#                  model_path: str,
#                  device: Union[str, List[int]],
#                  accelerator: str):

#         self.map_location = torch.device(f'cuda:{device[0]}' if accelerator == 'gpu' else 'cpu')
#         self.diar_model = NeuralDiarizer.from_pretrained(model_path).to(self.map_location)

#     def diarize(self, audio_path: str) -> pd.DataFrame:
#         '''
#         Diarize from audio_filepath to pandas dataframe with format:

#         ['start_time', 'end_time', 'speaker', 'text']
#         '''
#         annotation = self.diar_model(audio_path, num_workers=0, batch_size=16)
#         rttm=annotation.to_rttm()
#         df = pd.DataFrame(columns=['start_time', 'end_time', 'speaker', 'text'])
#         lines = rttm.splitlines()
#         if len(lines) == 0:
#             df.loc[0] = 0, 0, 'No speaker found'
#             return df
#         start_time, duration, prev_speaker = float(lines[0].split()[3]), float(lines[0].split()[4]), lines[0].split()[7]
#         end_time = float(start_time) + float(duration)
#         df.loc[0] = start_time, end_time, prev_speaker, ''

#         for line in lines[1:]:
#             split = line.split()
#             start_time, duration, cur_speaker = float(split[3]), float(split[4]), split[7]
#             end_time = float(start_time) + float(duration)
#             if cur_speaker == prev_speaker:
#                 df.loc[df.index[-1], 'end_time'] = end_time
#             else:
#                 df.loc[len(df)] = start_time, end_time, cur_speaker, ''
#             prev_speaker = cur_speaker

#         return df


class PyannoteDiarizer:
    '''
    Pyannote diarizer implementation
    '''

    def __init__(
        self, device: str, min_segment_length: float, min_silence_length: float
    ):

        device = (
            device
            if device in ["cuda", "cpu"]
            else "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.device = torch.device(device)
        logging.info("Running on device: %s", device)

        self.min_segment_length = min_segment_length
        logging.info("Minimum Segment Length: %s", self.min_segment_length)

        self.min_silence_length = min_silence_length
        logging.info("Minimum Silence Length: %s", self.min_silence_length)

        self.diarizer = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1").to(
            self.device
        )

        logging.info("Pyannote model loaded!")

    def diarize_into_string(self, audio_filepath: str) -> str:
        """
        Diarize from audio_filepath to string with format:

        start={}s stop={}s speaker_{} \n
        """

        logging.info("Diarization started")
        diarization = self.diarizer(audio_filepath)
        simple_text = ""

        for turn, _, cur_speaker in diarization.itertracks(yield_label=True):
            simple_text += (
                f"start={turn.start:.3f}s stop={turn.end:.3f}s speaker_{cur_speaker} \n"
            )

        return simple_text

    def diarize(self, audio_filepath: str) -> pd.DataFrame:
        """
        Diarize from audio_filepath to pandas dataframe with format:

        ['start_time', 'end_time', 'speaker', 'text']
        """

        logging.info("Diarization started")
        diarization = self.diarizer(audio_filepath)
        df_diarized = pd.DataFrame(columns=["start_time", "end_time", "speaker", "text"])
        prev_speaker = "None"

        for turn, _, speaker in diarization.itertracks(yield_label=True):

            start_time, stop_time, cur_speaker = (
                round(turn.start, 3),
                round(turn.end, 3),
                speaker,
            )
            duration = stop_time - start_time

            if duration < self.min_segment_length:
                continue

            if cur_speaker == prev_speaker:

                if start_time - prev_stoptime > self.min_silence_length:

                    df_diarized.loc[len(df_diarized)] = start_time, stop_time, cur_speaker, ""

                df_diarized.loc[df_diarized.index[-1], "end_time"] = stop_time

            else:
                df_diarized.loc[len(df_diarized)] = start_time, stop_time, cur_speaker, ""

            prev_speaker = cur_speaker
            prev_stoptime = stop_time

        return df_diarized