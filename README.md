# sequence_representations

------------------------------DATA on DRIVE---------------------------------------------------------------------------
contains 2 databases:
	data12000: 
		12 052 sequences: uniref database (UniProt Knowledgebase with 35% similarity)
		Max sequence length: 1500
		Min sequence length: 20


	data160000:
		160 026 sequences: intersection of uniref database (UniProt Knowledgebase with 50% identity, 21 mil sequences) and
 Swissprot (500 000 sequences)
		Max sequence length: 34348
		Min sequence length: 6

both folders include 4 more txt files containing labeled 3gram sequences: perm1, perm2, perm3, perm4
	perm1 -> original permutation
	perm2 -> permutation in all windows of size 5
	perm3 -> permutation of random 3grams in windows of size 5
	perm4 -> permutation of every third window of size 5

(for data160000 only perm4 for now, perm1,2,3 are on drive)


------------------------------NGRAM REPRESENTATIONS-------------------------------------------------------------
contains models for 3gram representations:
3grams_vecdim100_ctx25.dict -> dictionary of 3grams
3grams_vecdim100_ctx25.word2vec -> word2vec model 
Glove/results -> Glove model



------------------------------SEQUENCE LEARNING------------------------------------------------------------------
sequence_representations.py -> script for generating learning
ngram_representations.py -> script for getting ngram representations
sequence_results: folder for log and model results (both log and model should containg folders Glove and word2vec)


------------------------------EXECUTION------------------------------------------------------------------
(in folder sequence_learning)
python sequence_representations.py ../data/data160000/perm4 ../ngram_representations 3grams_vecdim100_ctx25 100 100 --nb_epoch 50 --Glove g --log_to_file l --activation tanh

(without --log_to_file argument all results are printed to the console)
(other possible parameters are in the script)

