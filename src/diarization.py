from pyannote.audio import Pipeline
from pyannote.database.util import load_rttm
from pyannote.core import notebook

import torch
import matplotlib.pyplot as plt

class Diarization:
    
    def __init__(self, access_token:str) -> None:
        
        # To gain access token:
        # 1. visit hf.co/pyannote/speaker-diarization and accept user conditions
        # 2. visit hf.co/pyannote/segmentation and accept user conditions
        # 3. visit hf.co/settings/tokens to create an access token
        self.access_token = access_token
    
    def run(self, input_path:str) -> None:
        '''
        Apply diarization on one audio file.
        Also generates a RTTM file & a timeline graph per audio file.
        
        Args:
            input_path: location of the input file
        '''
        
        # Create a pipeline
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1", use_auth_token=self.access_token)
            
        # Apply the pipeline to an audio file
        diarization = pipeline(input_path)

        file_name = input_path.split("/")[-1].split(".")[0]
        output_path = "output/"+file_name+".rttm"
        
        # Dump the diarization output to disk using RTTM format
        with open(output_path, "w") as rttm:
            diarization.write_rttm(rttm)
        
    def plot(self, rttm_path:str, save:bool=False) -> None:
        
        file_name = rttm_path.split("/")[-1].split(".")[0]
        
        # Plot and save a graph of the RTTM results
        reference = load_rttm(rttm_path)[file_name]            
        fig, ax = plt.subplots(figsize=(10,5))
        notebook.plot_annotation(reference, ax=ax, time=True, legend=True)
        ax.set_title(file_name)
        plt.show()
        
        if save:
            fig.savefig('plots/'+file_name+'.png')