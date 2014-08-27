#! /import/bc2/home/nimwegen/GROUP/local/bin/python

def arguments():
    import argparse
    parser = argparse.ArgumentParser(description="""By providing a data file, this program
    splits the data into n adjoint training/test set and run the classification algorithm
    on them, as a result it reports AUC for each of the runs. """)

    parser.add_argument('-i', dest='input_file', action='store', required=True, \
                        help="""The data file that at each row contains a TAB seperated
                        data where the first column represents the class ID. The following
                        fields are reserved for the different features.""")
    parser.add_argument('-f', dest='feature_file', action='store', required=True, \
                        help="""A file that contains the length of feature vector for
                        each of the features. For example, if a feature accepts 'yes' or 'no'
                        as its values, then the feature vector length or dimension is 2.
                        At each line we have two values that are seperated by TAB. First
                        one is the feature ID (just a number) and second is the feature length.""")
    # parser.add_argument('-n', dest='number_of_subsets', action='store', required=True, type=int, \
    #                     help="""This number represents that how many subset must be selected for the
    #                     training purpose""")
    parser.add_argument('-p', dest='percentage_of_training', action='store', required=True, type=float, \
                        help="""The percentage of items in the training set, which is going to be the
                        similar number for each of the classes.""")
    parser.add_argument('-n', dest='number_of_tests', action='store', required=True, type=int, default=1, \
                        help="""How many times should the test to be done.""")    
    parser.add_argument('-o', dest='destination_dir', action='store', required=False, default='.', \
                        help="""Output directory that by default is at the current working directory.""")
    args = parser.parse_args()
    return args


def load_data(data_file):
    data = {}
    number_of_data_items = 0
    with open(data_file) as file_handler:
        for row in csv.reader(file_handler, delimiter='\t'):
            class_id = row[0]            
            data.setdefault(class_id, [])
            data[class_id].append(row[1:])
            number_of_data_items += 1
    return data, number_of_data_items


def load_data_list(data_file):
    data = []
    number_of_data_items = 0
    with open(data_file) as inf:        
        for line in inf:  data.append(line)
    return data


def running_classification_algorithm(dest_dir, featureFile, num_of_classes):
    trainingFile = os.path.join(dest_dir, 'trainingSet')
    testFile = os.path.join(dest_dir, 'testSet')
    program = '/import/bc2/home/nimwegen/omidi/Classification/Program2/bin/DWT_classification'
    cmd = [program,
           featureFile,
           str(num_of_classes),
           trainingFile,
           testFile,]
    print cmd
    proc = Popen(cmd, stdout=PIPE)
    result = []
    convert = lambda v: [int(v[0]), int(v[1]), float(v[2]), float(v[3])]
    for row in csv.reader(proc.stdout, delimiter='\t'):
        # print '\t'.join(row).rstrip()
        result.append( convert( row ) )
    return result


if __name__ == '__main__':
    import numpy as np
    from subprocess import Popen, PIPE
    import csv, random, os
    from analyze_results import *
    
    args = arguments()
    data, number_of_data_items = load_data(args.input_file)
    data_list = load_data_list(args.input_file)
    number_of_data_items = len(data_list)

    if args.destination_dir != '.':
        try:
            os.mkdir(args.destination_dir)
        except OSError:
            print "The directory %s is already exist!" % args.destination_dir
            print
            None
            
    for c in data.keys():
        random.shuffle(data[c])
    
    number_of_classes = len(data.keys())
    number_of_elements_in_training = int(args.percentage_of_training * number_of_data_items)
    number_of_elements_in_training_per_class = int(number_of_elements_in_training / number_of_classes)
    for i in xrange(args.number_of_tests):
        # The trainign set file
        # it has equal number of items for each class
        # and whenever we're making a new Training file the data is randomized
        # with open(os.path.join(args.destination_dir, 'trainingSet'), 'w') as training_file, \
        #          open(os.path.join(args.destination_dir, 'testSet'), 'w') as test_file:                 
        #      # for class_id in data.keys(): # to make sure that an equal number of cases for each class to be in the trainingSet
        #      #     j = 0  # to keep count of the number of items per class in the training set
        #      #     random.shuffle(data[class_id])  # always shuffle the input data to make sure data is 'quite' random!
        #      #     while j < number_of_elements_in_training_per_class:
        #      #         try:
        #      #             training_file.write('%s\t%s\n' % (class_id, '\t'.join(data[class_id][j])))
        #      #             j += 1
        #      #         except IndexError:
        #      #             break
        #      #     # now write up the rest of data in the testSet file
        #      #     for k in xrange(j, len(data[class_id])):
        #      #         test_file.write('%s\t%s\n' % (class_id, '\t'.join(data[class_id][k])))
        #      random.shuffle(data_list)
        #      for k in xrange(number_of_elements_in_training):
        #          training_file.write(data_list[k])
        #      for s in xrange(number_of_elements_in_training, len(data_list)):
        #          test_file.write(data_list[s])
             
        results = running_classification_algorithm(args.destination_dir, args.feature_file, number_of_classes)        
        with open(os.path.join(args.destination_dir, 'results.%d' % (i+1)), 'w') as result_file:
            for row in results:
                result_file.write('\t'.join(map(str, row)) + '\n')
        specificity_sensitivity(results)
        loss_function(results)
