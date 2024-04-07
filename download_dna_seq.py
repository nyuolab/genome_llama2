import os
import yaml
import subprocess
import zipfile
import glob
import shutil

# Function to recursively find files with specific patterns
def find_files(directory, pattern):
    for root, dirs, files in os.walk(directory):
        for basename in files:
            if glob.fnmatch.fnmatch(basename, pattern):
                filename = os.path.join(root, basename)
                yield filename

# Load the YAML file
yaml_file_path = 'genome_train_tokenizer.yml'  # Adjust this to your YAML file's path
with open(yaml_file_path) as file:
    data = yaml.safe_load(file)  # Using safe_load for security

# Ensure a single 'genome' directory exists in the current working directory
genome_dir = os.path.join(os.getcwd(), 'genome_train_tokenizer_raw_data')
os.makedirs(genome_dir, exist_ok=True)

# Process each genome item in the YAML file
for item in data['genome']:
    dir_name = item['name'].replace(' ', '_').replace('.', '')
    os.makedirs(dir_name, exist_ok=True)

    zip_path = os.path.join(dir_name, 'download.zip')
    curl_command = f"curl -o \"{zip_path}\" \"{item['command']}\""
    subprocess.run(curl_command, shell=True)

    # Extract the ZIP file
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(dir_name)

    # Find and move the desired files to the 'genome' directory
    for file_pattern in ["GCF*", "GCA*"]:
        for file in find_files(dir_name, file_pattern):
            shutil.move(file, genome_dir)
            print(f"Moved {file} to {genome_dir}")

    # Cleanup: Remove the directory after extracting and moving the desired files
    shutil.rmtree(dir_name)

print("Process complete. All desired files have been moved to the 'genome' folder.")