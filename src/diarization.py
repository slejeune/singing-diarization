from pyannote.audio import Pipeline
import torch

class Diarization:
    
    def __init__(self, access_token:str) -> None:
        
        # To gain access token:
        # 1. visit hf.co/pyannote/speaker-diarization and accept user conditions
        # 2. visit hf.co/pyannote/segmentation and accept user conditions
        # 3. visit hf.co/settings/tokens to create an access token
        self.access_token = access_token
    
    def run(self, input_name:str, output_name:str):
        '''
        Add later.
        '''
        
        # Create device for running on GPU
        device = torch.device('mps')

        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1", use_auth_token=self.access_token)
        pipeline.to(device)
        
        # Apply the pipeline to an audio file
        diarization = pipeline(input_name)

        # dump the diarization output to disk using RTTM format
        with open("output/"+output_name+".rttm", "w") as rttm: # Adapt this to be what you need
            diarization.write_rttm(rttm)
        
        # TODO: get different type of output / check what you can do with the pipeline
