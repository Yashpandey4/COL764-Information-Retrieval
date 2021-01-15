"""
    This file merges two set of recommendations:
    1) recommendations.csv
    2) string_matching_recommendations.csv
    into one file:
    1) coninuations.csv
"""

import sys
import csv

def error():
    if len(sys.argv) != 3:
        print("ERROR: commandline arguments expected")
        exit(1)

if __name__ == '__main__':
    error()
    csv1 = sys.argv[1]
    csv2 = sys.argv[2]

    with open('continuations.csv', 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(['team_info', 'dwijesh', 'iddwijesh@gmail.com'])
        print('combining recommendations.csv...')
        with open('recommendations.csv', 'r') as file1:
            content1 = csv.reader(file1, delimiter=',')
            writer.writerows(content1)
        print('combining string_matching_recommendations.csv...')
        with open('string_matching_recommendations.csv', 'r') as file2:
            content2 = csv.reader(file2, delimiter=',')
            writer.writerows(content2)
