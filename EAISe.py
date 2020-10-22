import random
import sys
import gzip
import re
from nltk.tokenize import sent_tokenize, word_tokenize



random.seed(42)

inputfile = sys.argv[1] #input file with sentences to be labelled

print("ag,dg,fr,hp,sd,sp")

nrc = {}
wordnet_joy = set()
wordnet_anger = set()
wordnet_disgust = set()
wordnet_fear = set()
wordnet_sadness = set()
wordnet_surprise = set()


with open("lexicons/nrc.txt") as n:
	for line in n:
		line = line.rstrip()
		sp = line.split("\t")
		if sp[0] in nrc:
			nrc[sp[0]].append(sp[2])
		else:
			nrc[sp[0]] = [sp[2]]



#wordnet, 6 sets

with open("lexicons/anger") as w1:
	for line in w1:
		if "_" not in line:
			wordnet_anger.add(line.rstrip())


with open("lexicons/fear") as w2:
	for line in w2:
		if "_" not in line:
			wordnet_fear.add(line.rstrip())

with open("lexicons/sadness") as w3:
	for line in w3:
		if "_" not in line:
			wordnet_disgust.add(line.rstrip())

with open("lexicons/joy") as w4:
	for line in w4:
		if "_" not in line:
			wordnet_joy.add(line.rstrip())

with open("lexicons/positive") as w5:
	for line in w5:
		if "_" not in line:
			wordnet_sadness.add(line.rstrip())

with open("lexicons/negative") as w6:
	for line in w6:
		if "_" not in line:
			wordnet_surprise.add(line.rstrip())



#nrc affect intensity: anger,fear,sadness,joy,positive,negative



with open(inputfile) as infile:
	for line in infile:

		#6/6 classes
		nrc_ag = 0
		nrc_fe = 0
		nrc_jo = 0
		nrc_sa = 0
		nrc_pos = 0
        nrc_neg = 0
	


		tokenized_text = word_tokenize(line.lower())

		
		for word in tokenized_text:
			
			#nrc
			if word in nrc:
				nrc_ag = nrc_ag + int(nrc[word][0])
				nrc_fe = nrc_fe + int(nrc[word][2])
				nrc_jo = nrc_jo + int(nrc[word][3])
				nrc_sa = nrc_sa + int(nrc[word][4])
				nrc_dg = nrc_pos + int(nrc[word][5])
				nrc_sp = nrc_neg + int(nrc[word][6])

			

		dict_nrc = {"ag":nrc_ag, "pos":nrc_pos, "fr":nrc_fe, "hp":nrc_jo, "sd":nrc_sa, "neg":nrc_neg}
		
		
		#random label in case of max value tie
		nrc_label = random.choice([key for key in dict_nrc if dict_nrc[key]==max(dict_nrc.values())])
		

		print (line.strip(),"\t",','.join(str(x) for x in dict_wd.values()),",",','.join(str(x) for x in dict_nrc.values()))

