import os
import pandas as pd

folder_path = 'names/'

for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)
    names = []
    try:
        infile = open(file_path, 'r')
        all_names = infile.readlines()
        for line in all_names:
            name = line.split(',')
            if name not in names:
                names.append(name[0])
        print(f'Have retrieved names from {file_path}')
    except Exception as e:
        print('Unable to access file and retrieve name')

# yob1880 = 'names/yob1880.txt'

# names = []
# infile = open(yob1880, 'r')
# all_names = infile.readlines()
# for line in all_names:
#     name = line.split(',')
#     if name not in names:
#         names.append(name[0])
        
df = pd.DataFrame(names)
df.to_csv('list_of_names.csv', index=False)