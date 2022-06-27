nvcc -O3 -arch sm_75 main.cu -o prog -lcuda

for st in {0..3}
do
	for op in {0..2}
	do
		for i in {1..2}
		do
			if ! [[ $st -eq 0 && $op -eq 0 ]]
			then
				./prog $st $op >> ../data/data.csv
			fi
		done
	done
done
