import os
import csv

IMAGENET_PATH = '/Users/felixsu/Home/school/Sp18/CS194/project/data/nips'

labels_path = os.path.join(IMAGENET_PATH, 'dev_dataset.csv')
out_path = os.path.join(IMAGENET_PATH, 'aligned_dev_dataset.csv')

with open(labels_path, 'r') as f:
	reader = csv.reader(f)
	rows = list(reader)
	for i in range(1, len(rows)):
		rows[i][6] = str(int(rows[i][6]) - 1)
		rows[i][7] = str(int(rows[i][7]) - 1)

with open('aligned_dev_dataset.csv', 'w') as f:
	writer = csv.writer(f)
	writer.writerows(rows)