import os
import json

path = "C:\\Users\\Moyez\\Desktop\\Code\\Python\\Mnet"

files  = []

for r, d, f in os.walk(path):
    for file in f:
        if (file.find('.ipynb') != -1) and (file.find('checkpoint') == -1):
            files.append(((os.path.join(r, file)), file))



for file, name in files:
    print(file)
    print(name[:-6])

def parse_file(file_json):
    file_lines = []
    for i in range(len(file_json['cells'])):
        cell = file_json['cells'][i]

        if 'source' in cell:
            for line in cell['source']:
                file_lines.append(line)
        file_lines.append('\n')
        file_lines.append('\n')

    return file_lines


for file, name in files:
    f = open(file, "rb")
    contents = f.read()
    file = contents.decode()
    file_json = json.loads(file)
    
    parsed_file = parse_file(file_json)
    f = open("C:\\Users\\Moyez\\Desktop\\Code\\Python\\M-NET\\{}.py".format(name[:-6]), "w")
    for line in parsed_file:
        f.write(line)
    f.close()







