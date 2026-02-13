import glob
import os
import random
import re
import numpy as np

def get_audio_files(folder_path, extensions=('.mp3', '.wav', '.flac', '.aac')):
    """Gets a list of audio files from a folder."""
    if not os.path.exists(folder_path):
        return []
    return [
        file for file in glob.glob(os.path.join(folder_path, '*'))
        if file.lower().endswith(extensions)
    ]

def categorize_files_by_gender(file_list):
    """Categorizes files into male and female lists."""
    male_list = []
    female_list = []
    
    for file in file_list:
        lower_file = file.lower()
        if "female" in lower_file:
            female_list.append(file)
        elif "male" in lower_file:
            male_list.append(file)
            
    return male_list, female_list

def create_balanced_dataset(male_list, female_list, count_per_gender=190, seed=42):
    """Creates a balanced dataset of male and female files."""
    random.seed(seed)
    
    selected_male = random.sample(male_list, min(len(male_list), count_per_gender))
    selected_female = random.sample(female_list, min(len(female_list), count_per_gender))
    
    dataset = selected_male + selected_female
    random.shuffle(dataset)
    
    return dataset

def split_train_test(dataset, train_ratio=0.75):
    """Splits the dataset into train and test sets."""
    threshold = int(train_ratio * len(dataset))
    return dataset[:threshold], dataset[threshold:]

def extract_and_select_students(filenames, max_students=9, files_per_student=7):
    """Selects students with a specific number of files."""
    id_to_files = {}

    # Store filenames for each student ID
    for file in filenames:
        pattern = r'[/_](\d{9})[/_]'  # Match 9-digit student IDs between / or _
        match = re.findall(pattern, file)
        if match:
            student_id = match[0]
            if student_id not in id_to_files:
                id_to_files[student_id] = []
            id_to_files[student_id].append(file)

    # Shuffle student IDs to randomize selection
    shuffled_ids = list(id_to_files.keys())
    random.shuffle(shuffled_ids)

    # Select students who have enough files
    selected_students = {}
    for student_id in shuffled_ids:
        if len(id_to_files[student_id]) == files_per_student:
            selected_students[student_id] = id_to_files[student_id]
            if len(selected_students) == max_students:
                break 
                
    return selected_students
