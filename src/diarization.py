from pyannote.audio import Pipeline


class Diarization:
    
    def __init__(self, access_token:str) -> None:
        
        # To gain access token:
        # 1. visit hf.co/pyannote/speaker-diarization and accept user conditions
        # 2. visit hf.co/pyannote/segmentation and accept user conditions
        # 3. visit hf.co/settings/tokens to create an access token
        self.access_token = access_token
    
    def run(self, audio_file:str):
        '''
        Add later.
        '''

        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1", use_auth_token=self.access_token)

        # apply the pipeline to an audio file
        diarization = pipeline(audio_file)

        # dump the diarization output to disk using RTTM format
        with open("test.rttm", "w") as rttm:
            diarization.write_rttm(rttm)
            
        # TODO: get images of output
        # TODO: change the output name to be customizable from main
