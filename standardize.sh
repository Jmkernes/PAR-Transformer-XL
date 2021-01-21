#!/bin/bash

echo " Standardizing the dataset "

cat $1 | tr '[:upper:]' '[:lower:]' | sed 's/[0-9][0-9]*/NUM/g'

