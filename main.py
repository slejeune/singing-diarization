from src import *

import time
from beepy import beep

def main():
    with open('access_token.txt') as f:
        access_token = f.readlines()[0]
        
    input_name = "audio/test.mp3"
    output_name = "test"
    
    diarization = Diarization(access_token)
    evaluation = Evaluation()
    
    # TODO: Make it possible to throw an entire folder in there and loop through all the files

    tic = time.perf_counter()
    diarization.run(input_name, output_name)
    toc = time.perf_counter()
    print(f"Runtime diarization: {toc - tic:0.2f} seconds")
    
    # Make a beep to alert that execution is done
    beep(sound="ping")
    
if __name__ == '__main__':
    main()
