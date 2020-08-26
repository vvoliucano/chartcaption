# from generate_data import *
import os
import sys
print("import from generate_data")

current_url = os.path.dirname(__file__)
parent_url = os.path.abspath(os.path.join(current_url, os.pardir))

print(current_url)
print(parent_url)

sys.path.append(current_url)
sys.path.append(parent_url)