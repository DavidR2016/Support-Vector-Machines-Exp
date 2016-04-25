#!/bin/bash
set -eu

#Radial Basis function Kernel
MAIN_DIRECTORY=/mnt/data2/home2/davidr/SVM1 #link to the main directory
SVM_SCRIPTS=/mnt/data2/home2/davidr/SVM1/scripts 
GRID_SEARCH=/mnt/data2/home2/davidr/SVM1/libsvm-3.14/tools #SCript to run grid search
#LIBLINEAR=/home/pelumi/Documents/liblinear-1.93
TEST_TRAIN=0
ESTIMATE_C_VALUE=1
FRESH_SETUP=1
PREDICT_STAT=0
for package_name in "original" #"comparing_words" "spell-check"
do
	for size in "12K" #"4K" #"2K" "4K" "8K" "10K" "12K"
	do
		LINK=$MAIN_DIRECTORY/${package_name}/NCHLT/${size}/SVM_RBF
		
		for fold in 1 2 3 4 #1 #2 3 4
		do
			for ngram in 3 #4 5
			do 
				NGRAM_LINK=$LINK/fold_${fold}/ngram_${ngram}
				
				if [ $FRESH_SETUP -eq 1 ]; then
					rm -r -f $NGRAM_LINK
					
					mkdir -p $NGRAM_LINK/computation
					mkdir -p $NGRAM_LINK/result
					
					#Create n-gram tokens
					for lang in "ss" "afr" "zul" "eng"
					do
						LANG_FOLD_DIR=$MAIN_DIRECTORY/${package_name}/NCHLT/${size}/cross_validate_${lang}/fold_${fold}
						
						#Create n-gram tokens across all train set
						perl $SVM_SCRIPTS/text_normalization.pl $LANG_FOLD_DIR/train_${fold} ${ngram} "" ${lang} 1 0 >> $NGRAM_LINK/computation/all_train_ngrams.txt
					done
					
					#Sort estracted tokens with their frequency preceeding each token item
					sort $NGRAM_LINK/computation/all_train_ngrams.txt | uniq -c | sed -e 's/^[ ]*//' > $NGRAM_LINK/computation/sorted_all_train_ngrams.txt

					#Create feature vector
					for lang in "ss" "afr" "zul" "eng"
					do
						LANG_FOLD_DIR=$MAIN_DIRECTORY/${package_name}/NCHLT/${size}/cross_validate_${lang}/fold_${fold}
						
						#Create feature vectors used for training and testing
						perl $SVM_SCRIPTS/text_normalization.pl $LANG_FOLD_DIR/train_${fold} ${ngram} $NGRAM_LINK/computation/sorted_all_train_ngrams.txt ${lang} 0 1 >> $NGRAM_LINK/computation/train.txt
						
						#Create feature vectors used for testing
						perl $SVM_SCRIPTS/text_normalization.pl $LANG_FOLD_DIR/test_${fold} ${ngram} $NGRAM_LINK/computation/sorted_all_train_ngrams.txt ${lang} 0 1 >> $NGRAM_LINK/computation/test_all.txt
						
						#Create feature vectors for each languaeg specific data. This will be used to estimate precision and recall per fold.
						perl $SVM_SCRIPTS/text_normalization.pl $LANG_FOLD_DIR/test_${fold} ${ngram} $NGRAM_LINK/computation/sorted_all_train_ngrams.txt ${lang} 0 1 > $NGRAM_LINK/computation/test_${lang}.txt
					done
					
					#Create a range values based on the train data and use it to on our test data.
					svm-scale -l 0 -u 1 -s $NGRAM_LINK/computation/range.txt $NGRAM_LINK/computation/train.txt > $NGRAM_LINK/computation/train_norm.txt
					
					#Apply our range values on test set
					svm-scale -r $NGRAM_LINK/computation/range.txt $NGRAM_LINK/computation/test_all.txt > $NGRAM_LINK/computation/test_all.data
					for lang in "ss" "afr" "zul" "eng"
					do
						svm-scale -r $NGRAM_LINK/computation/range.txt $NGRAM_LINK/computation/test_${lang}.txt > $NGRAM_LINK/computation/test_${lang}_scale
					done
					
					#Final data is train.data
					rl -o $NGRAM_LINK/computation/train.data $NGRAM_LINK/computation/train_norm.txt
					
					#Remove previous files to free up space
					#rm -f $NGRAM_LINK/computation/train_norm.txt $NGRAM_LINK/computation/train.txt $NGRAM_LINK/computation/sorted_all_train_ngrams.txt $NGRAM_LINK/computation/all_train_ngrams.txt $NGRAM_LINK/computation/test_all.txt
					for lang in "ss" "afr" "zul" "eng"
					do
						rm -f $NGRAM_LINK/computation/test_${lang}.txt
					done
				fi
					
				if [ $ESTIMATE_C_VALUE -eq 1 ]; then
					count=$(wc -l < "$NGRAM_LINK/computation/sorted_all_train_ngrams.txt")
					one=1
					result=$(echo "$one/$count" | bc -l)
					log=$(echo "l($result)/l(2)" | bc -l)
					cd $GRID_SEARCH
					python $GRID_SEARCH/grid.py -log2c -13.2877,13.2877,1.6609 -log2g ${log},${log},0 -v 3 -m 300 $NGRAM_LINK/computation/train.data  > $NGRAM_LINK/result/result_${ngram}
					#python $GRID_SEARCH/grid.py -log2c 13.2877,13.2877,0.0 -log2g $log,$log,0 -v 2 -m 400 $NGRAM_LINK/computation/train.data > $NGRAM_LINK/result/result_${ngram}

				fi
				
				#Train and ouput a model. Test validation set on the output model. 
				if [ $TEST_TRAIN -eq 1 ]; then
					svm-train -c 2 -t 2 -g 0.1767767 -s 0 -m 400 $NGRAM_LINK/computation/train.data $NGRAM_LINK/computation/train.data.model
					svm-predict $NGRAM_LINK/computation/test_all.data $NGRAM_LINK/computation/train.data.model $NGRAM_LINK/result/predict.txt > $NGRAM_LINK/result/result.txt
				fi
					
				if [ $PREDICT_STAT -eq 1 ]; then
					rm -f $NGRAM_LINK/result/statistics.txt $NGRAM_LINK/result/accuracy_for_all_lang
					
					echo -e "\t SS \t AF \t ZUL \t EN \t TOTAL\n" >> $NGRAM_LINK/result/statistics.txt
					
					for lang in "ss" "afr" "zul" "eng"
					do
						echo -e "\nIdentification accuracy for ${lang} is: \n" >> $NGRAM_LINK/result/accuracy_for_all_lang
						
						svm-predict $NGRAM_LINK/computation/test_${lang}_scale $NGRAM_LINK/computation/train.data.model $NGRAM_LINK/result/predict_${lang} >> $NGRAM_LINK/result/accuracy_for_all_lang
						
						$SVM_SCRIPTS/estimate_total_result.pl $NGRAM_LINK/result/predict_${lang} ${lang} >> $NGRAM_LINK/result/statistics.txt
					done
				fi			
			done
		done
	done
done
