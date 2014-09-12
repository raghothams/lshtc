#!/bin/bash

file_name=$1
no_of_files=$2
file_length=$3

offset_start=1

for i in $(seq 1 $2)
do
	offset_end=$(( $offset_start + $file_length - 1 ))
	echo $offset_start $offset_end

	split_file_name=$file_name"."$i
	sed -n "$offset_start,$offset_end""p;" $file_name > $split_file_name

	offset_start=$(( 1 + $offset_end ))

done

