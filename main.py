from src import *

import time
from os import listdir
from os.path import isfile, join
from beepy import beep

def main():
    with open('access_token.txt') as f:
        access_token = f.readlines()[0]
        
    input_path_singing = "input/singing/"
    input_path_speech = "input/speaking/"
    
    diarization = Diarization(access_token)
    evaluation = Evaluation()

    for input_path in [input_path_speech, input_path_singing]:
        
        file_name = input_path.split("/")[-1].split(".")[0]
        
        tic = time.perf_counter()
        if "." in input_path:   # Single file
            diarization.run(input_path)
            evaluation.evaluate("labels/"+input_path.split("/")[-2]+"/"+file_name+".rttm",
                                "output/"+file_name+".rttm")
        else:                   # Folder
            print("Warning: you are analyzing a folder! Depending on your device this may take a while.")
            for input_file in [f for f in listdir(input_path) if isfile(join(input_path, f))]:
                # Ignore hidden files
                if(input_file.split("/")[-1].split(".")[0] == ""):
                    continue
                diarization.run(input_path+input_file)
                evaluation.evaluate("labels/"+input_path.split("/")[-2]+"/"+input_file.split(".")[0]+".rttm",
                                    "output/"+input_file.split(".")[0]+".rttm")
        toc = time.perf_counter()
        print(f"Runtime: {toc - tic:0.2f} seconds")
    
    report = evaluation.report()
    evaluation.plot(report)
    
    # Make a beep to alert that execution is done
    beep(sound="ping")
    
if __name__ == '__main__':
    main()
