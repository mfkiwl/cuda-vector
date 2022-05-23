all:
	nvcc -O3 -arch sm_70 main.cu -o prog -I../Ouroboros/include --expt-relaxed-constexpr

test:
	nvcc -O3 -arch sm_70 test.cu -o prog

debug:
	nvcc -g -G -arch sm_70 -Xcompiler -rdynamic -lineinfo main.cu -o prog
