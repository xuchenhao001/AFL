#!/bin/bash
user=$1
addr=$2
test_name=$3

FIRST_CHAR=$(echo $test_name | cut -c1-1)
FOLLOWING_CHAR=$(echo $test_name | cut -c2-)
PS_NAME="[${FIRST_CHAR}]${FOLLOWING_CHAR}.py"

ssh ${user}@${addr} "kill -9 \$(ps -ef|grep '$PS_NAME'|awk '{print \$2}')"

