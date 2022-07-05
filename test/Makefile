all:
	nvcc  -arch sm_75 main.cu -o prog -I../stdgpu/src/

test:
	nvcc -O3 -arch sm_70 test.cu -o prog

debug:
	nvcc -g -G -arch sm_70 -Xcompiler -rdynamic -lineinfo main.cu -o prog
