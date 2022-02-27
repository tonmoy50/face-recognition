# Face Recognition
A simple OpenCV based facial recognizer that can detect and identify faces

## Setup

Run "conda install -c conda-forge dlib" to install dlib to your conda python environment and for the rest of the requirement run 'pip install -r requirements.txt'

## Structure

- Put harrcascade files to src/ directory
- Create folder named 'dataset', 'models' in the project root directory

## Functionality

- `capture_faces.py` file is used to create dataset based on user id
- `trainer.py` file is used to train model based on images present in dataset folder. Run this after running previous mentioned python file
- `recognizer.py` is used to run realtime recognizer. It will show confidence level and name of the identified face/object