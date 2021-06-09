#! /usr/bin/bash
for i in {0..20}
do
cd ~/repos/isingmarket3d/testdata/test$i
echo "Generating data for test$i"
../../build/ising3d
done
