# Load CUDA using the following command
# module load cuda
#
CC = nvcc
CFLAGS = -O3 -arch=compute_35 -code=sm_35
NVCCFLAGS = -O3 -arch=compute_35 -code=sm_35
LIBS = 

TARGETS = serial gpu autograder

all:	$(TARGETS)

serial: serial.o common-serial.o
	$(CC) -o $@ $(LIBS) serial.o common-serial.o
gpu: gpu.o common.o
	$(CC) -o $@ $(NVCCLIBS) $(CFLAGS) gpu.o common.o
autograder: autograder.o common.o
	$(CC) -o $@ $(LIBS) autograder.o common.o

serial.o: serial.cu common-serial.h
	$(CC) -c $(CFLAGS) serial.cu
autograder.o: autograder.cu common.h
	$(CC) -c $(CFLAGS) autograder.cu
gpu.o: gpu.cu common.h
	$(CC) -c $(NVCCFLAGS) gpu.cu
common.o: common.cu common.h
	$(CC) -c $(CFLAGS) common.cu
common-serial.o:  common-serial.cpp common-serial.h
	$(CC) -c $(CFLAGS) common-serial.cpp

clean:
	rm -f *.o $(TARGETS) *.stdout *.txt
