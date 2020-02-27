mkdir data/conceptnet

mkdir -p data/conceptnet

wget https://ttic.uchicago.edu/~kgimpel/comsense_resources/train100k.txt.gz -P data/conceptnet/
wget https://ttic.uchicago.edu/~kgimpel/comsense_resources/dev1.txt.gz -P data/conceptnet/
wget https://ttic.uchicago.edu/~kgimpel/comsense_resources/dev2.txt.gz -P data/conceptnet/
wget https://ttic.uchicago.edu/~kgimpel/comsense_resources/test.txt.gz -P data/conceptnet/

gunzip train100k.txt.gz
gunzip dev1.txt.gz
gunzip dev2.txt.gz
gunzip test.txt.gz