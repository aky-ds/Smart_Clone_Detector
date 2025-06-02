import os
from pathlib import Path
# Define the root directory of the project
list_of_files=[
    'setup.py',
    'README.md',
    'Requirements.txt',
    'models/models.txt',
    'experiments/experiments.txt',
    'data/data.txt',
    'Detector.py',
    'templates/index.html',
    'templates/result.html',
    'static/css/style.css',
    'static/uploads/upload.txt',
]

for file in list_of_files:
    file_path = Path(file)
    file_dir,file_name = os.path.split(file_path)
    if file_dir!='':
        os.makedirs(file_dir, exist_ok=True)
    if not os.path.exists(file_path):
        with open(file_path,'w') as f:
            pass