mkdir tmp
mkdir -p data/embeddings
mkdir -p data/gazetteers/
cd tmp && wget http://ml.nec-labs.com/senna/senna-v3.0.tgz && tar -zxvf senna-v3.0.tgz
cd ..
paste tmp/senna/hash/words.lst tmp/senna/embeddings/embeddings.txt | expand -t 1 > data/embeddings/senna.w2v
cp tmp/senna/hash/ner.loc.lst data/gazetteers/
cp tmp/senna/hash/ner.misc.lst data/gazetteers/
cp tmp/senna/hash/ner.org.lst data/gazetteers/
cp tmp/senna/hash/ner.per.lst data/gazetteers/
rm -rf tmp
