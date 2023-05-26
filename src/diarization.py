from pyannote.audio import Pipeline
from pyannote.database.util import load_rttm
from pyannote.core import notebook

import torch
import os
import matplotlib.pyplot as plt

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
        Also generates a RTTM file & a timeline graph per audio file.
        
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

            file_name = input_path.split("/")[-1].split(".")[0]
            output_path = "output/"+file_name+".rttm"
            
            # Dump the diarization output to disk using RTTM format
            with open(output_path, "w") as rttm:
                diarization.write_rttm(rttm)
                
            reference = load_rttm(output_path)[file_name]            
            fig, ax = plt.subplots()
            notebook.plot_annotation(reference, ax=ax, time=True, legend=True)
            ax.set_title(file_name, pad=20)
            fig.savefig('plots/'+file_name+'.png')
            
        else: # A folder
            print("Warning: you are (probably) analyzing multiple files in a folder! This may take a while.")
            
            for file in os.listdir(input_path):
                
                file_name = file.split("/")[-1].split(".")[0]
                output_path = "output/"+file_name+".rttm"
                
                # Ignore hidden files
                if(file_name == ""):
                    continue
                
                # Apply the pipeline to an audio file
                diarization = pipeline(input_path+file)

                # Dump the diarization output to disk using RTTM format
                with open(output_path, "w") as rttm:
                    diarization.write_rttm(rttm)

                reference = load_rttm(output_path)[file_name]            
                fig, ax = plt.subplots()
                notebook.plot_annotation(reference, ax=ax, time=True, legend=True)
                ax.set_title(file_name, pad=20)
                fig.savefig('plots/'+file_name+'.png')