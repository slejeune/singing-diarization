from src import *

import time
from os import listdir
from os.path import isfile, join
from beepy import beep

def main():
    with open('access_token.txt') as f:
        access_token = f.readlines()[0]
        
    input_path = "input/singing/"
    
    diarization = Diarization(access_token)
    evaluation = Evaluation()

    tic = time.perf_counter()
    if "." in input_path:   # Single file
        diarization.run(input_path)
    else:                   # Folder
        print("Warning: you are analyzing a folder! This may take a while.")
        for input_file in [f for f in listdir(input_path) if isfile(join(input_path, f))]:
            # Ignore hidden files
            if(input_file.split("/")[-1].split(".")[0] == ""):
                continue
            diarization.run(input_path+input_file)
    toc = time.perf_counter()
    print(f"Runtime diarization: {toc - tic:0.2f} seconds")
    
    # print(evaluation.evaluate("labels/speaking/Bdb001.rttm", "output/Bdb001.rttm"))
    # print(evaluation.report())
    
    # Make a beep to alert that execution is done
    beep(sound="ping")
    
if __name__ == '__main__':
    main()
