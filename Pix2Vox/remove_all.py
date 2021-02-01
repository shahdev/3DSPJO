import os

import sys

sources = ['car', 'chair', 'tab', 'desk', 'mail', 'case']
targets = ['car', 'chair', 'tab', 'desk', 'mail', 'case', 'ones']

for src in sources:
	for targ in targets:
		if src == targ:
			continue
		directory = '/'+src+'_'+targ+'/'
		if not os.path.exists(directory):
			os.makedirs(directory)
		try:
			os.system("cp %s_%s/prediction_18999* %s"%(src, targ, directory))
		except:
			continue