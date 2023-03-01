#!/bin/bash


wget --continue http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz 

mkdir gsc_raw_data;
tar -xvzf speech_commands_v0.01.tar.gz -C gsc_raw_data

rm speech_commands_v0.01.tar.gz
