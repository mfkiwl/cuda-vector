for i in {0..9}
do
	echo -n "$i "
	for j in {1..5}
	do
		nvcc -O3 -DNUM_BLOCKS=$((1 << i)) -arch sm_75 main.cu -o prog -lcuda
		./prog 2 4 >> ../data/NB_data.csv
	done
done
