#! /import/bc2/home/nimwegen/GROUP/local/bin/python

age =  lambda x: str((int(x.split('-')[0]) / 10) + 1)
menopause = {'lt40':'1', 'ge40':'2', 'premeno':'3'}
tumor_size = lambda x: str((int(x.split('-')[0]) / 5) + 1)
inv_nodes = lambda x: str((int(x.split('-')[0]) / 3) + 1)
yes_no = lambda x: '1' if x=='yes' else '2'
left_right = lambda x: '1' if x=='left' else '2'
breast_quad = {'left_up':'1', 'left_low':'2', 'right_up':'3', 'right_low':'4', 'central':'5'}
events = lambda x: '1' if x=='no-recurrence-events' else '2'


def print_record(record):
    print '\t'.join([
        events(record[0]),
        age(record[1]),
        menopause[record[2]],
        tumor_size(record[3]),
        inv_nodes(record[4]),
        yes_no(record[5]),
        record[6],
        left_right(record[7]),
        breast_quad[record[8]],
        yes_no(record[9])
        ])
        

if __name__ == '__main__':
    from sys import argv
    data_filename = argv[1]
    import numpy as np
    import re

    with open(data_filename) as file_handler:
        for line in file_handler:
            record = line.split(',')
            if re.search('\?', line):
                if record.index('?') < 8:
                    line2 = line.replace('?','yes')
                    record = line.split(',')
                    print_record(record)
                    line2 = line.replace('?','no')
                    record = line.split(',')
                    print_record(record)
            else:
                print_record(record)
            
