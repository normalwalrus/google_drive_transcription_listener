import os
import tempfile
import time
from codes.asr_inference_service.audio_preprocessing import resample_audio_array
from codes.asr_inference_service.model import ASRModelForInference
from codes.google_doc_utils.utils import (audio_from_mp4, authenticate,
                                         check_if_file_in_folder,
                                         download_file, write_text_to_txt,
                                         extract_root_folder_id, read_txt_file,
                                         get_all_audio_files, append_text_to_txt,
                                         get_all_mp4_files, upload_txt_file, update_txt_file)
from codes.google_doc_utils.error_handling import get_status_message

import librosa
import soundfile as sf
from dotenv import load_dotenv

load_dotenv()

DRIVE_URL = '#'
OUTPUTS_URL = '#'
LOGS_URL = '#'
AUDIO_VIDEO_URL = '#'

LOCAL_DOWNLOAD_FOLDER = 'downloads'
LOCAL_OUTPUT_FOLDER = 'outputs'
LOCAL_LOGS_TXT_FILE = 'logs'
SAMPLE_RATE = 16000
ROOT_FOLDER_ID = extract_root_folder_id(DRIVE_URL)
AUDIO_VIDEO_FOLDER_ID = extract_root_folder_id(AUDIO_VIDEO_URL)
UPLOAD_OUTPUTS_FOLDER_ID = extract_root_folder_id(OUTPUTS_URL)
UPLOAD_LOGS_FOLDER_ID = extract_root_folder_id(LOGS_URL)

STATUS_TXT_FILE = 'status.txt'
ARCHIVE_STATUS_TXT_FILE = 'archive_status.txt'
STATUS_TXT_ID = '#'
ARCHIVE_STATUS_TXT_ID = '#'

OVERALL_STATUS_TXT_FILE = os.path.join(LOCAL_LOGS_TXT_FILE, STATUS_TXT_FILE)

model = ASRModelForInference(
    model_dir=os.getenv("PRETRAINED_MODEL_DIR"),
    sample_rate=int(os.getenv("SAMPLE_RATE")),
    device=os.getenv("DEVICE"),
    timestamp_format=os.getenv("TIMESTAMPS_FORMAT"),
    min_segment_length=float(os.getenv("MIN_SEGMENT_LENGTH")),
    min_silence_length=float(os.getenv("MIN_SILENCE_LENGTH")),
)

service = authenticate()

def handle_statuses(filename: str ='none', step:str  ='', status_filepath = '', status_txt_id = STATUS_TXT_ID):
    ''' Update and upload status.txt '''
    archive_status_filepath = os.path.join(LOCAL_LOGS_TXT_FILE, ARCHIVE_STATUS_TXT_FILE)
    status_message = get_status_message(filename, step=step)

    if step == 'started_up' or step == 'reset':
        # Archiving old status messages
        old_messages = read_txt_file(status_filepath)
        append_text_to_txt(old_messages, archive_status_filepath)
        update_txt_file( archive_status_filepath, ARCHIVE_STATUS_TXT_ID, service)
        # Upload current status messages and overwrites old ones
        write_text_to_txt(status_message, status_filepath)
        update_txt_file(status_filepath, STATUS_TXT_ID, service)
    
    else:
        append_text_to_txt(status_message, status_filepath)
        update_txt_file(status_filepath, status_txt_id, service)
    
    return
    
def audio_loop(list_of_audio_in_drive: list):
    ''' 
    Audio loop where checking of:
    1. new mp3/wav files in drive
    2. downloading the file
    3. resampling + rechannleing + diarization + transcription
    4. uploading the transcription txt to drive
    '''
    for audio in list_of_audio_in_drive:
        if not check_if_file_in_folder(audio['name'], LOCAL_DOWNLOAD_FOLDER):
            
            pre, _ = os.path.splitext(audio['name'])
            audio_status_filepath = os.path.join(LOCAL_LOGS_TXT_FILE, pre+'_status.txt')
            write_text_to_txt(f'Transcription process of {pre}:\n\n', audio_status_filepath)
            status_txt_id = upload_txt_file(audio_status_filepath, UPLOAD_LOGS_FOLDER_ID, service)
            
            try:
                handle_statuses(audio['name'], step = 'downloading', status_filepath=audio_status_filepath, status_txt_id=status_txt_id)
                output_filepath = download_file(audio['id'], LOCAL_DOWNLOAD_FOLDER, service)
                handle_statuses(audio['name'], step = 'downloaded', status_filepath=audio_status_filepath, status_txt_id=status_txt_id)
                
                # Transcription 
                data, samplerate = librosa.load(output_filepath)
                y = resample_audio_array(data, samplerate, SAMPLE_RATE)
                with tempfile.NamedTemporaryFile(delete=True, suffix=".wav") as temp_file:
                    sf.write(temp_file, y, SAMPLE_RATE)
                    temp_filepath = temp_file.name
                    transcriptions = model.diar_inference(filepath=temp_filepath)
                    handle_statuses(audio['name'], step = 'transcribed', status_filepath=audio_status_filepath, status_txt_id=status_txt_id)
                
                #
                output_txt_path = os.path.join(LOCAL_OUTPUT_FOLDER, pre + '.txt')
                write_text_to_txt(transcriptions, output_txt_path)
                upload_txt_file(output_txt_path, UPLOAD_OUTPUTS_FOLDER_ID, service)
                handle_statuses(output_txt_path, step = 'uploaded', status_filepath=audio_status_filepath, status_txt_id=status_txt_id)
                handle_statuses(output_txt_path, step = 'uploaded', status_filepath=OVERALL_STATUS_TXT_FILE, status_txt_id=STATUS_TXT_ID)
                
            except:
                handle_statuses(audio['name'], step = 'error', status_filepath=pre, status_txt_id=status_txt_id)
                
def video_loop(list_of_mp4_in_drive: list):
    ''' 
    Video loop where checking of:
    1. new mp4 files in drive
    2. downloading the file
    3. resampling + rechannleing + diarization + transcription
    4. uploading the transcription txt to drive
    '''
    for video in list_of_mp4_in_drive:
            if not check_if_file_in_folder(video['name'], LOCAL_DOWNLOAD_FOLDER):
                pre, _ = os.path.splitext(video['name'])
                video_status_filepath = os.path.join(LOCAL_LOGS_TXT_FILE, pre+'_status.txt')
                write_text_to_txt(f'Transcription process of {pre}:\n\n', video_status_filepath)
                status_txt_id = upload_txt_file(video_status_filepath, UPLOAD_LOGS_FOLDER_ID, service)
                
                try:
                    handle_statuses(video['name'], step = 'downloading', status_filepath=video_status_filepath, status_txt_id=status_txt_id)
                    output_filepath = download_file(video['id'], LOCAL_DOWNLOAD_FOLDER, service)
                    handle_statuses(video['name'], step = 'downloaded', status_filepath=video_status_filepath, status_txt_id=status_txt_id)
                    
                    numpy_audio_array, sample_rate = audio_from_mp4(output_filepath)
                    numpy_audio_array = resample_audio_array(numpy_audio_array, sample_rate, SAMPLE_RATE)
                    
                    with tempfile.NamedTemporaryFile(delete=True, suffix=".wav") as temp_file:
                        sf.write(temp_file, numpy_audio_array, SAMPLE_RATE)
                        temp_file_path = temp_file.name
                        transcription = model.diar_inference(temp_file_path)
                        handle_statuses(video['name'], step = 'transcribed', status_filepath=video_status_filepath, status_txt_id=status_txt_id)
                    
                    output_txt_path = os.path.join(LOCAL_OUTPUT_FOLDER, pre + '.txt')
                    write_text_to_txt(transcription, output_txt_path)
                    upload_txt_file(output_txt_path, UPLOAD_OUTPUTS_FOLDER_ID, service)
                    handle_statuses(output_txt_path, step = 'uploaded', status_filepath=video_status_filepath, status_txt_id=status_txt_id)
                    handle_statuses(output_txt_path, step = 'uploaded', status_filepath=OVERALL_STATUS_TXT_FILE, status_txt_id=STATUS_TXT_ID)
                except:
                    handle_statuses(video['name'], step = 'error', status_filepath=video_status_filepath, status_txt_id=status_txt_id)

def list_files_in_folder(service, folder_id):
    query = f"'{folder_id}' in parents"
    results = service.files().list(q=query, fields="files(id, name, mimeType)").execute()
    files = results.get('files', [])
    
    if not files:
        print("No files found.")
    else:
        for file in files:
            print(f"Name: {file['name']}, MIME Type: {file['mimeType']}")

def main():

    #list_files_in_folder(service, ROOT_FOLDER_ID)
    
    list_of_audio_in_drive = get_all_audio_files(AUDIO_VIDEO_FOLDER_ID, service)
    list_of_mp4_in_drive = get_all_mp4_files(AUDIO_VIDEO_FOLDER_ID, service)
    
    # print(list_of_audio_in_drive)
    # print(list_of_mp4_in_drive)
    
    audio_loop(list_of_audio_in_drive)
    video_loop(list_of_mp4_in_drive)


if __name__ == '__main__':
    
    # Reset the status
    handle_statuses('', 'started_up', OVERALL_STATUS_TXT_FILE)
    count = 0
    sleepy_time = 10
    status_update_interval_in_sec = 600
    
    while (True):
    
        main()
        time.sleep(sleepy_time)
        count+=1
        
        # Heartbeat status update
        if (sleepy_time * count) >= status_update_interval_in_sec:
            count=0
            handle_statuses(status_filepath=OVERALL_STATUS_TXT_FILE)