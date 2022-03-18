#!/usr/bin/env python3

from os import listdir
from os.path import isfile, join
import re

def sort_predicate(value):
    nums = re.findall(r'\d+', value)
    return int(nums[0])

def list_files(dir, sorted=True):
    onlyfiles = [dir + f for f in listdir(dir) if isfile(join(dir, f))]
    if sorted:
        onlyfiles.sort(key=sort_predicate)
    return onlyfiles