TARGET = convolution

$(TARGET): convolution.c timer.c
	if [ ! -d '/usr/include/CL' ]; then	cp -r CL/ /usr/include/; fi
	gcc -o $(TARGET) convolution.c timer.c -lOpenCL -lm

clean :
	rm test
