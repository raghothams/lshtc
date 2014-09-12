
from collections import OrderedDict

# take a list with key, value, key, value ...
arr = ["k1", "v1", "k2", "v21", "k3", "v31", "k1", "v12", "k3", "v32", "k2", "v22"]

# get an iterator for the list
it = iter(arr)

# zip arr with arr
arr1 = zip(it,it)

# get an ordered dictionary
res = OrderedDict()


# get one item at a time from arr1
for k, v in arr1:

# create an array with v as element
    if k not in res:
        res[k] = [v]

# push v to existing array
    else:
        res[k].append(v)

# create a dict for final output
final_res = {}

# iterate over res
for k,v in res.items():

# create map with k, v
    final_res[k] = v

print final_res

