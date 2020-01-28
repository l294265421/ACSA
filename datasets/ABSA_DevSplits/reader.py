# -*- coding: utf-8 -*-

"""

Authors: liyuncong
Date:    2019/10/12 15:22
"""

import pickle


original = "./dataset/Restaurants_category.pkl"
destination = original + ".new"

content = ''
outsize = 0
with open(original, 'rb') as infile:
    content = infile.read()
with open(destination, 'wb') as output:
    for line in content.splitlines():
        outsize += len(line) + 1
        output.write(line + str.encode('\n'))
with open(destination, mode='rb') as in_file:
    data = pickle.load(in_file, encoding='utf-8')
    print()
