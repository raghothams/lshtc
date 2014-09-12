#!/bin/bash

file_name=$1
file_length=$2

file_lines=$(cat $file_name | wc -l)
echo $file_lines
no_of_lines=$((file_lines / file_length))
echo $no_of_lines
offset_start=1

for i in $(seq 1 $no_of_lines)
do
	offset_end=$(( $offset_start + $file_length - 1 ))
	echo $offset_start $offset_end

	split_file_name=$file_name"."$i
	sed -n "$offset_start,$offset_end""p;" $file_name > $split_file_name

	offset_start=$(( 1 + $offset_end ))

done

