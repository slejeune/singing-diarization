from src import *

import time
from beepy import beep

def main():
    with open('access_token.txt') as f:
        access_token = f.readlines()[0]
        
    input_path = "audio/"
    
    diarization = Diarization(access_token)
    evaluation = Evaluation()

    tic = time.perf_counter()
    diarization.run(input_path)
    toc = time.perf_counter()
    print(f"Runtime diarization: {toc - tic:0.2f} seconds")
    
    # Make a beep to alert that execution is done
    beep(sound="ping")
    
if __name__ == '__main__':
    main()
