#!/bin/bash
sudo apt-get update
sudo apt-get install -y python3-pip

# Install necessary Python libraries
pip3 install pyspark
pip3 install elephas
pip3 install tensorflow
pip3 install scikit-learn
pip3 install pandas
pip3 install numpy
pip3 install Flask
pip3 install google-cloud-storage
pip3 install protobuf==3.20.*
