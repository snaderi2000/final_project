import csv 

with open('SMSSpamCollection.csv', newline='') as csvfile:
    data = csv.reader(csvfile, delimiter = '\t')
    for row in data:
        print(row[1])

