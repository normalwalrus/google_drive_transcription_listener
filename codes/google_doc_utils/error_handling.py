import datetime

def get_status_message(filename, step='downloading'):
    ''' Generate status message at each step '''
    
    current_time = datetime.datetime.now()
    
    if step == 'downloading':
        
        status_message = f'{current_time} : {filename} is being downloaded \n'
    
    elif step == 'downloaded':
        
        status_message = f'{current_time} : {filename} has been downloaded, transcription in progress \n'
    
    elif step == 'transcribed':
        
        status_message = f'{current_time} : {filename} has been transcribed, text file being uploaded to Outputs folder \n'
    
    elif step == 'uploaded':
        
        status_message = f'{current_time} : {filename} has been uploaded! \n\n'
    
    elif step == 'started_up':
        
        status_message = f'{current_time} : DH Transcription Service started up! \n\n'
    
    elif step == 'error':
        
        status_message = f'{current_time} : DH Transcription Service has faced an error with {filename} and has shut down! \n\n'
    
    else:
        
        status_message = f'{current_time} : DH Transcription Service is online and running! \n\n'
        
    return status_message