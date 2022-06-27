nvcc -O3 -arch sm_75 main.cu -o prog -lcuda

for st in {0..2}
do
	for i in {1..30}
	do
		./prog $st >> ../data/data.csv
	done
done
