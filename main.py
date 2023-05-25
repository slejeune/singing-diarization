from src import *
import time

def main():
    with open('access_token.txt') as f:
        access_token = f.readlines()[0]
    
    diarization = Diarization(access_token)
    evaluation = Evaluation()
    
    tic = time.perf_counter()
    diarization.run("audio/test.mp3")
    toc = time.perf_counter()
    print(f"Runtime diarization: {((toc - tic)/60):0.2f} minutes")
    
    
if __name__ == '__main__':
    main()
