import os
import csv

IMAGENET_PATH = '/home/felixsu/project/data/nips'

labels_path = os.path.join(IMAGENET_PATH, 'unaligned_final_dataset.csv')
out_path = os.path.join(IMAGENET_PATH, 'final_dataset.csv')

with open(labels_path, 'r') as f:
	reader = csv.reader(f)
	rows = list(reader)
	for i in range(1, len(rows)):
		rows[i][6] = str(int(rows[i][6]) - 1)
		rows[i][7] = str(int(rows[i][7]) - 1)

with open(out_path, 'w') as f:
	writer = csv.writer(f)
	writer.writerows(rows)
