#!/bin/bash

# download
python download_speech_corpus.py downloader_conf/vcc2018.yml

# change directory names
mv wav vcc2018
cd vcc2018/wav
find . -type d -name "VCC2*" | while read f; do mv $f $(echo $f | sed 's/VCC2//'); done
cd ../../
