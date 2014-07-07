#! /import/bc2/home/nimwegen/GROUP/local/bin/python

if __name__ == '__main__':
    from sys import argv
    data_filename = argv[1]
    import numpy as np

    min_temp = 36
    convert = lambda x: '2' if x=="no" else '1'
    with open(data_filename) as file_handler:
        for line in file_handler:
            record = line.split()
            temp = str(int(np.ceil(float(record[0].replace(',', '.')))) - min_temp + 1)
            print '\t'.join([
                convert(record[-2]),
                temp,
                convert(record[-1]),
                convert(record[1]),
                convert(record[2]),
                convert(record[4]),
                convert(record[5]),
                ])    
