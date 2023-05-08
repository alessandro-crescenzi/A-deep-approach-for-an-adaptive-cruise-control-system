import json
import re
from tqdm import tqdm

if __name__ == '__main__':
    # pattern = re.compile(r'(?P<timestamp>\S+\s\S+),(?P<B>\S+),(?P<A>\S+),(?P<velocity>\d+)km/h,(?P<latitude>\f+),'
    #                      r'(?P<longitude>\f+),(?P<zero>\d+),(?P<filename>\S+)')
    pattern = re.compile(r'(?P<timestamp>\d+\-\d+\-\d+\s\d+\:\d+\:\d+),(?P<B>\S),(?P<A>\S),(?P<velocity>\d+)km/h,'
                         r'(?P<latitude>\d+\.\d+),(?P<longitude>\d+\.\d+),(?P<zero>\d+),(?P<filename>\S+)')

    f = open('GPSData000001.txt', 'r')

    data = {}

    for line in tqdm(f):
        res = pattern.match(line)
        if res is not None:
            datum = res.groupdict()
            try:
                datum['velocity'] = int(datum['velocity'])
                datum['latitude'] = float(datum['latitude'])
                datum['longitude'] = float(datum['longitude'])
            except ValueError:
                continue
            try:
                data[datum['filename']].append({'timestamp': datum['timestamp'],
                                                'velocity': datum['velocity'],
                                                'latitude': datum['latitude'],
                                                'longitude': datum['longitude']})
            except KeyError:
                data[datum['filename']] = []
                data[datum['filename']].append({'timestamp': datum['timestamp'],
                                                'velocity': datum['velocity'],
                                                'latitude': datum['latitude'],
                                                'longitude': datum['longitude']})

    with open('GPSData.json', 'w') as outfile:
        json.dump(data, outfile)

