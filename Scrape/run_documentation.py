# import os

# # set the directory path where your python files are located
# dir_path = 'Scrape/api_documentation_code'

# # get a list of all the files in the directory
# file_list = os.listdir(dir_path)

# # iterate over each file in the list
# executed = []
# skipped = []
# for file_name in file_list:

#     # check if the file is a Python file
#     if file_name.endswith('.py'):
#         if not file_name.startswith('paddle_fluid'):
#         # build the command to run the file using the Python interpreter
#             command = f'python {os.path.join(dir_path, file_name)}'

#             # execute the command using the os module
#             result = os.system(command)

#             # if the result is non-zero, the command failed
#             if result != 0:
#                 skipped.append(file_name)

# # print the list of error files
# if skipped:
#     print('The following files produced an error:')
#     for file_name in skipped:
#         print(file_name)
# else:
#     print('All files ran successfully!')
        

import csv

# note: If you use 'b' for the mode, you will get a TypeError
# under Python3. You can just use 'w' for Python 3


import os
import subprocess

# set the directory path where your python files are located
dir_path = 'Scrape/api_documentation_code'

# get a list of all the files in the directory
file_list = os.listdir(dir_path)

# create an empty list to store the names of the files that produced an error
error_files = []
lines_list = open("Scrape/error_documentation_wo_hijack.txt").read().splitlines()
# print(lines_list)
# iterate over each file in the list
for file_name in file_list:

    # check if the file is a Python file
    if file_name in lines_list and file_name.endswith('.py') and not file_name.startswith('paddle_fluid'):

        # build the command to run the file using the Python interpreter
        command = f'python {os.path.join(dir_path, file_name)}'

        # execute the command using the subprocess module
        result = subprocess.run(command, shell=True, capture_output=True)

        # if the result has a non-zero return code, the command failed
        if result.returncode != 0:
            error_files.append((file_name,result.stderr.decode()))
            print(f"{file_name} produced an error:")
            print(result.stderr.decode())

with open('error_1.txt', 'w') as fp:
    fp.write('\n'.join('%s %s' % x for x in error_files))
# print the list of error files



with open('errors.csv','wb') as out:
    # csv_out=csv.writer(out)
    # csv_out.writerow(['name','num'])
    # for row in error_files:
    #     csv_out.writerow(row)
    out.write('\n'.join('%s , %s' % x for x in error_files))

if error_files:
    print('The following files produced an error:')
    for file_name,b in error_files:
        print(file_name,b)
else:
    print('All files ran successfully!')
