#! /usr/bin/bash
for i in {0..20}
do
cd testdata/test$i
echo "Generating data for test$i"
rm -r reference_data/
../../build/ising3d
cd ../..
done
