from multiprocessing import Pool, freeze_support
import os
import subprocess

dir_path = 'E:\\UIUC\\Spring 2023\\CS 527\\FreeFuzz\\Scrape'

def run_file(file_name):
    if file_name.endswith('.py') and not file_name.startswith('paddle_distributed'):
        command = f'python "{os.path.join(dir_path, file_name)}"'
        print(f"Running {file_name}")
        result = subprocess.run(command, shell=True, capture_output=True)
        if result.returncode != 0:
            print(f"{file_name} produced an error:")
            print(result.stderr.decode())

if __name__ == '__main__':
    freeze_support()
    p = Pool(5)
    file_list = os.listdir(dir_path)
    with p:
        p.map(run_file, file_list)