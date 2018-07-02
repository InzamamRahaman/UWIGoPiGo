from __future__ import print_function
from UWIGoPiGo import *

init_eyes()
for i in range(0,10):
	snap()
	result, score = see()
	print(result,score)
