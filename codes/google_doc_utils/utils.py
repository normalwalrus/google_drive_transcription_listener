import os
import io
import re

from google.auth.transport.requests import Request # type: ignore
from google.oauth2.credentials import Credentials # type: ignore
from google_auth_oauthlib.flow import InstalledAppFlow # type: ignore
from googleapiclient.discovery import build # type: ignore
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload # type: ignore

from moviepy.video.io.VideoFileClip import VideoFileClip

# If modifying these SCOPES, delete the file token.json.
SCOPES = ['https://www.googleapis.com/auth/drive']

def authenticate():
    """Authenticate the user and return the service object."""
    creds = None
    credentials_path = 'credentials.json'
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(credentials_path, SCOPES)
            creds = flow.run_local_server(port=0)
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    service = build('drive', 'v3', credentials=creds)
    return service

def extract_root_folder_id(drive_url):
    """Extract the folder ID from a Google Drive folder URL."""
    match = re.search(r'/folders/([a-zA-Z0-9_-]+)', drive_url)
    if not match:
        raise ValueError('Invalid Google Drive folder URL')
    return match.group(1)

def get_all_audio_files(folder_id, service):
    """Extract name and ids of all wav and mp3 (mpeg) files in folder id"""

    query = f"'{folder_id}' in parents and (mimeType='audio/wav' or mimeType='audio/mpeg' or mimeType = 'audio/x-wav')"
    results = service.files().list(q=query, fields="files(id, name)").execute()
    files = results.get('files', [])

    list_of_wav = []
    for file in files:
        wav_entry = {'name':file['name'], 'id':file['id']}
        list_of_wav.append(wav_entry)
    
    return list_of_wav

def get_all_mp4_files(folder_id, service):
    """Extract name and ids of all mp4 files in folder id"""

    query = f"'{folder_id}' in parents and mimeType='video/mp4'"
    results = service.files().list(q=query, fields="files(id, name)").execute()
    files = results.get('files', [])

    list_of_mp4 = []
    for file in files:
        wav_entry = {'name':file['name'], 'id':file['id']}
        list_of_mp4.append(wav_entry)
    
    return list_of_mp4

def download_file(file_id, output_folder, service):
    ''' Download file from Google Drive'''
    
    file_metadata = service.files().get(fileId=file_id).execute()
    file_name = file_metadata['name']
    print(file_metadata)
    
    request = service.files().get_media(fileId = file_id)
    
    file_path = os.path.join(output_folder, file_name)
    os.makedirs(output_folder, exist_ok=True)
    
    with io.FileIO(file_path, 'wb') as fh:
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            status, done = downloader.next_chunk()
            print(f"Download {int(status.progress() * 100)}% complete.")
            print("done",done)
            
    return file_path
            
def check_if_file_in_folder(filename, folder_path):
    ''' check if a file is in a folder , return True if in, return False if not '''
    
    onlyfiles = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    
    if (filename in onlyfiles):
        #print(f"{filename} is in {folder_path}")
        return True
    else:
        #print(f"{filename} is not in {folder_path}")
        return False
    
def write_text_to_txt(text, output_text_filepath):
    ''' Write a string into a txt file '''

    with open(output_text_filepath, "w") as f:
        
        f.write(text)
        
def append_text_to_txt(text, output_text_filepath):
    ''' Append a string into a txt file '''

    with open(output_text_filepath, "a") as f:
        
        f.write(text)
        
def audio_from_mp4(mp4_filepath):
    ''' Read an mp4's audio file into a numpy array'''
    
    video_clip = VideoFileClip(mp4_filepath)

    if not video_clip.audio:
        return {"error": "No audio found in the MP4 file"}

    audio_clip = video_clip.audio
    audio_array = audio_clip.to_soundarray()
    sample_rate = audio_clip.fps

    audio_clip.close()
    video_clip.close()

    os.remove(mp4_filepath)

    audio_array = audio_array.T

    return audio_array, sample_rate

def audio_from_mp4(mp4_filepath):
    ''' Read an mp4's audio file into a numpy array'''
    
    video_clip = VideoFileClip(mp4_filepath)

    if not video_clip.audio:
        return {"error": "No audio found in the MP4 file"}

    audio_clip = video_clip.audio
    audio_array = audio_clip.to_soundarray()
    sample_rate = audio_clip.fps

    audio_clip.close()
    video_clip.close()

    audio_array = audio_array.T

    return audio_array, sample_rate

def upload_txt_file(file_path, folder_id, service):
    """Uploads a TXT file to a specific Google Drive folder."""
    
    file_metadata = {
        "name": os.path.basename(file_path),  # Keep the original file name
        "parents": [folder_id]  # Upload to the specified folder
    }

    media = MediaFileUpload(file_path, mimetype="text/plain")

    uploaded_file = service.files().create(
        body=file_metadata,
        media_body=media,
        fields="id, name"
    ).execute()

    print(f"Uploaded {uploaded_file['name']} with ID: {uploaded_file['id']}")
    return 

def update_txt_file(file_path, folder_id, file_id, service):
    """Uploads a TXT file to a specific Google Drive folder."""
    
    file_metadata = {
        "name": os.path.basename(file_path),  # Keep the original file name
    }

    media = MediaFileUpload(file_path, mimetype="text/plain")

    uploaded_file = service.files().update(
        body=file_metadata,
        media_body=media,
        fields="id, name",
        fileId = file_id,
    ).execute()

    print(f"Uploaded {uploaded_file['name']} with ID: {uploaded_file['id']}")
    return 

def read_txt_file(txt_filepath):
    ''' Read contents of a txt file '''
    with open(txt_filepath, "r") as f:
        
        return f.read()