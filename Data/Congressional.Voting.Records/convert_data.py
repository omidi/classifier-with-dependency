#! /import/bc2/home/nimwegen/GROUP/local/bin/python

yes_no = lambda x: '1' if x=='y' else '2'
party = lambda x: '1' if x=='democrat' else '2'

def substitution(seqs, index):
    result = []
    for s in seqs:
        s[index] = 'y'
        result.append([i for i in s])
        s[index] = 'n'
        result.append([i for i in s])
    return result


if __name__ == '__main__':
    from sys import argv
    data_filename = argv[1]
    import numpy as np
    import re

    for i in xrange(1,17):
        print i, '\t', 2
    exit()
    
    with open(data_filename) as file_handler:
        for line in file_handler:
            record = line.rstrip().split(',')
            resulted_vector = [[i for i in record]]
            while True:
                try:
                    index = record.index('?')
                    record[index] = 'X'
                    resulted_vector = substitution(resulted_vector, index)
                except ValueError:
                    break
            for record in resulted_vector:
                print party(record[0]) + '\t' + \
                      '\t'.join(map(yes_no, record[1:]))            
            
            
