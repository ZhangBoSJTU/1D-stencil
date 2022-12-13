all:01_gpu_vector_add.x 02_gpu_vector_add.x 03/stensil.x

01_gpu_vector_add.x:01/gpu_vector_add.o
	nvcc -o 01/01_gpu_vector_add.x 01/gpu_vector_add.o

01/gpu_vector_add.o:01/gpu_vector_add.cu
	nvcc -c 01/gpu_vector_add.cu -o 01/gpu_vector_add.o

02_gpu_vector_add.x:02/gpu_vector_add.o
	nvcc -o 02/02_gpu_vector_add.x 02/gpu_vector_add.o

02/gpu_vector_add.o:02/gpu_vector_add.cu
	nvcc -c 02/gpu_vector_add.cu -o 02/gpu_vector_add.o

03/stensil.x:03/stensil.o
	nvcc -o 03/stensil.x 03/stensil.o -O0

03/stensil.o:03/stensil.cu
	nvcc -c 03/stensil.cu -o 03/stensil.o -O0

clean:
	rm -rf 01/*.o 01/*.x */*.o */*.x