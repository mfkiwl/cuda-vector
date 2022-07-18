for i in {1..10}
do
	echo -n "$i "
	nvcc -O3 -DNUM_BLOCKS=$((1 << i)) -arch sm_75 main.cu -o prog -lcuda
	for j in {1..10}
	do
		./prog 4 >> ../data/NB_data.csv
	done
done
