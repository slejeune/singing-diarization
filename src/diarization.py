from pyannote.audio import Pipeline
import torch
import os

class Diarization:
    
    def __init__(self, access_token:str) -> None:
        
        # To gain access token:
        # 1. visit hf.co/pyannote/speaker-diarization and accept user conditions
        # 2. visit hf.co/pyannote/segmentation and accept user conditions
        # 3. visit hf.co/settings/tokens to create an access token
        self.access_token = access_token
    
    def run(self, input_path:str):
        '''
        Apply diarization on one or multiple audio file(s).
        
        Args:
            input_path: location of the input file or folder
        '''
        
        # Create device for running on GPU
        device = torch.device('mps')

        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1", use_auth_token=self.access_token)
        pipeline.to(device)
        
        if "." in input_path: # A single file
            
            # Apply the pipeline to an audio file
            diarization = pipeline(input_path)

            # dump the diarization output to disk using RTTM format
            with open("output/"+input_path.split("/")[-1].split(".")[0]+".rttm", "w") as rttm: # Adapt this to be what you need
                diarization.write_rttm(rttm)
                
        else: # A folder
            print("Warning: you are (probably) analyzing multiple files in a folder! This may take a while.")
            
            for filename in os.listdir(input_path):
                
                # Apply the pipeline to an audio file
                diarization = pipeline(input_path+filename)

                # dump the diarization output to disk using RTTM format
                with open("output/"+filename.split("/")[-1].split(".")[0]+".rttm", "w") as rttm: # Adapt this to be what you need
                    diarization.write_rttm(rttm)
        
        # TODO: get different type of output / check what you can do with the pipeline
