#!/bin/bash
for attack_type in \ dag \ spatial_dag \ dag_foreground \ spatial_dag_foreground ; do
    cd $attack_type/weight_1/
    echo $attack_type
    for d in */ ; do
        cd $d
        for subd in */ ; do
            echo $subd 
            cat $subd/log.txt | grep IOU | tail -2
        done
        cd ..
        echo $d
    done
    cd ../../
done
