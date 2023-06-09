
# Singing Diarization in a Speaker Diarization System

This repository uses `pyannote.audio`'s speaker diarization pipeline to compare singing and speaking input. This project is part of the course Automatic Speech Recognition (Radboud University, 2022/2023).

# Set up

To properly function, this repository requires the following folder structure:
```
.
├── access_token.txt
├── input/
│   ├── singing
│   └── speaking
├── labels/
│   ├── singing
│   └── speaking
├── output
└── plots
```
Furthermore, the `access_token.txt` file required for the pipeline is created  by pasting the access token obtained from [Huggingface](https://huggingface.co/pyannote/speaker-diarization) into an empty `txt` file.