rm ../data/data_phases.csv
nvcc -O3 -arch sm_75 main.cu -o prog -lcuda

for st in 12 13
do
	for k in 1 3 10
	do
		for q in 1 3 10 30 100 300 1000
		do
			for i in {1..10}
			do
				./prog $st $k $q >> ../data/data_phases.csv
			done
		done
	done
done
