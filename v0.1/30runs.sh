#!/bin/sh

for RUN in {1..30}
do
	python3 "$1" $RUN
done

