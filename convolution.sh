#!/bin/bash

ary_size=4096
iteration=100
affinity=$1
priority=$2
log_name=$3

# ./convolution $ary_size $iteration $affinity $priority
./convolution $ary_size $iteration $affinity $priority $log_name
