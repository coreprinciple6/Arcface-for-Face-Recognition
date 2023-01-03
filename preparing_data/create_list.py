import pandas as pd

col_list = ['first_image', 'second_image', 'is_same']
df = pd.read_csv("pairs.csv", usecols=col_list)
r_name = []
r_same = []


with open('same2.csv', 'w') as b:
    for line in df['is_same']:
        if line:
            l='True' #True
        else:
            l='False'

        b.write(l)
        b.write('\n')

# with open('same.csv', 'w') as b:
#     for line in df['is_same']:
#         if line:
#             l=1 #True
#         else:
#             l=0

#         b.write(str(l))
#         b.write('\n')

with open('names.txt', 'w') as b:
    for i,line in enumerate(df['first_image']):
        b.write(line)
        b.write('\n')
        b.write(df['second_image'][i])
        b.write('\n')