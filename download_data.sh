#!/bin/sh

echo "=== Acquiring datasets ==="
echo "---"

mkdir -p data
cd data

if [[ ! -d 'wikitext-2' ]]; then
    echo "- Downloading WikiText-2 (WT2)"
    curl https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip -o wikitext-2-v1.zip
    unzip -q wikitext-2-v1.zip
    cd wikitext-2
    mv wiki.train.tokens train.txt
    mv wiki.valid.tokens valid.txt
    mv wiki.test.tokens test.txt
    cd ..
fi

echo "Finished loading data."
echo "Saved 3 files in the directory /data/wikitext-2/"
echo "Files are named (train, test, valid).txt"
echo " ~~~~~ Happy Training! ~~~~~ "
