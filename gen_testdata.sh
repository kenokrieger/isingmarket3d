#! /usr/bin/bash
for i in {0..10}
do
cd testdata/test$i
echo "Generating data for test$i"
../../build/ising3d
../../plot_magnetisation.py
cd ../..
done
