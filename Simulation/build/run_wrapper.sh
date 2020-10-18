#!/bin/bash
count=1
while [ $count -le 4 ]; do
    ./QSimulation -c ../input/SimConfig.json -flush 1280
    mv ./RPRootOut_0000.root outdata/rp_out_$count.root
    mv ./TruthRootOut_0000.root outdata/truth_out_$count.root

    count=$(($count + 1))
done
