echo -n "" > out.csv
for n in 3 4 5 6 7 8
do
	echo -n "$n "
	for i in {1..5}
	do
		./prog $((10 ** $n)) >> out.csv
	done
done
