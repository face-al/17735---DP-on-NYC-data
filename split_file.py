"""
# show csv content
import pandas as pd
file = 'trip_data_1.csv'
df = pd.read_csv(file)
print(len(df))
print(df.head(1000))
"""

# split file
file = open('trip_data_1.csv', 'r')
header = file.readline()
csvfile = file.readlines()
filename = 1
batch_size = 1000
num_batch = 1
for i in range(num_batch * batch_size):
    if i % batch_size == 0:
        open(str(filename) + '.csv', 'w+').writelines(header)
        open(str(filename) + '.csv', 'a+').writelines(csvfile[i:i+batch_size])
        filename += 1