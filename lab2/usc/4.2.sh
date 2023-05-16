#!/bin/bash
source ./path.sh

#4.2.1
touch data/local/dict/silence_phones.txt &&  echo "sil" > data/local/dict/silence_phones.txt
touch data/local/dict/optional_silence.txt &&  echo "sil" > data/local/dict/optional_silence.txt

./create_nonsilent_vocab.sh

./create_phonem_lexicon.sh ./data/local/dict/silence_phones.txt ./data/local/dict/nonsilence_phones.txt 

./create_lms.sh

touch ./data/local/dict/extra_questions.txt

#4.2.2
./lm_build.sh 

#4.2.3
compile-lm ./data/local/lm_tmp/train_unigram.ilm.gz -t=yes /dev/stdout | grep -v unk | gzip -c > ./data/local/nist_lm/lm_phone_ug.arpa.gz
compile-lm ./data/local/lm_tmp/train_bigram.ilm.gz -t=yes /dev/stdout | grep -v unk | gzip -c > ./data/local/nist_lm/lm_phone_bg.arpa.gz

compile-lm ./data/local/lm_tmp/test_unigram.ilm.gz -t=yes /dev/stdout | grep -v unk | gzip -c > ./data/local/nist_lm/test_ug.arpa.gz
compile-lm ./data/local/lm_tmp/test_bigram.ilm.gz -t=yes /dev/stdout | grep -v unk | gzip -c > ./data/local/nist_lm/test_bg.arpa.gz

compile-lm ./data/local/lm_tmp/dev_unigram.ilm.gz -t=yes /dev/stdout | grep -v unk | gzip -c > ./data/local/nist_lm/dev_ug.arpa.gz
compile-lm ./data/local/lm_tmp/dev_bigram.ilm.gz -t=yes /dev/stdout | grep -v unk | gzip -c > ./data/local/nist_lm/dev_bg.arpa.gz

#4.2.4
prepare_lang.sh ./data/local/dict "<oov>" ./data/local/lang ./data/lang

#4.2.5
bash sort.sh

#4.2.6
./utils/utt2spk_to_spk2utt.pl ./data/train/utt2spk > ./data/train/spk2utt
./utils/utt2spk_to_spk2utt.pl ./data/test/utt2spk > ./data/test/spk2utt
./utils/utt2spk_to_spk2utt.pl ./data/dev/utt2spk > ./data/dev/spk2utt

#4.2.7
wget https://raw.githubusercontent.com/slp-ntua/slp-labs/master/lab2/timit_format_data.sh

bash timit_format_data.sh

#question 1
bash comp_perplexity