#define _GNU_SOURCE

#include <sys/time.h>
#include <stdint.h>
#include <sched.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <signal.h>
#include <CL/cl.h>
#include <time.h>
#include "timer.h"

#define PROGRAM_FILE "convolution.cl"
#define TOTAL_RESPONSE_TIME 0

static void set_affinity(int affinity);
static void set_scheduler(int priority);
void create_matrix(float* matrix, int ary_size);
void print_matrix(float* matrix, int ary_size);
cl_device_id create_device();
cl_program build_program(cl_context ctx, cl_device_id dev, const char* filename);
void cl_error_check(const char* text, cl_int err);

int main(int argc, char *argv[]) {

   /* Parse arguments */
   int ary_size; // Size of row/col in matrix
   int affinity;
   int priority;

   if(argc != 4){
      printf("Usage: ./convolution <ary_size> <affinity> <priority>");
      exit(1);
   }

   ary_size = atoi(argv[1]);
   affinity = atoi(argv[2]);
   priority = atoi(argv[3]);

   /* Init system */   
   set_affinity(affinity);
   set_scheduler(priority);

   /* Init timer */
   timer_init(9);

   /* Init matrix */
   float *input, *output;
   input = (float *)malloc(sizeof(float) * ary_size * ary_size);
   create_matrix(input, ary_size);
   output = (float *)malloc(sizeof(float) * ary_size * ary_size);

   /* OpenCL structures */
   cl_device_id device;
   cl_context context;
   cl_program program;
   cl_kernel kernel;
   cl_command_queue queue;
   cl_int err;
   cl_mem input_buffer, output_buffer;
   cl_event timing_event;
   cl_ulong event_start, event_end;
   
   /* Init input size */
   size_t local_size[2] = { 16, 16 };
   size_t global_size[2] = { (size_t)ceil((double)ary_size/local_size[0])*local_size[0], (size_t)ceil((double)ary_size/local_size[1])*local_size[1]};

   /* Create a device and context */
   device = create_device();
   context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
   cl_error_check("Couldn't create a context", err);

   /* Build program */
   program = build_program(context, device, PROGRAM_FILE);

   /* Create command queue */
   queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
   cl_error_check("Couldn't create a command queue", err);

   /* Create data buffer */
   input_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * ary_size * ary_size, NULL, &err);
   output_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * ary_size * ary_size, NULL, &err);
   cl_error_check("Couldn't create a buffer", err);

   timer_start(TOTAL_RESPONSE_TIME);

   /* Write input buffer */
   err = clEnqueueWriteBuffer(queue, input_buffer, CL_TRUE, 0, sizeof(float) * ary_size * ary_size, input, 0, NULL, NULL);
   cl_error_check("Couldn't create a buffer", err);

   /* Create kernel */
   kernel = clCreateKernel(program, "convolution", &err);
   cl_error_check("Couldn't create a kernel", err);

   /* Create kernel args */
   err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input_buffer);
   err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &output_buffer);
   err |= clSetKernelArg(kernel, 2, sizeof(int), &ary_size);
   cl_error_check("Couldn't create a kernel argument", err);

   /* Launch kernel */
   err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_size, local_size, 0, NULL, &timing_event);
   cl_error_check("Couldn't enqueue the kernel", err);
   clWaitForEvents(1, &timing_event);
   clGetEventProfilingInfo(timing_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &event_start, NULL);
   clGetEventProfilingInfo(timing_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &event_end, NULL);
   float kernel_execution_time = (float)(event_end-event_start)*1e-09;
   printf("[INFO] Kernel excution time: %f sec\n", kernel_execution_time);

   /* Read output data */
   err = clEnqueueReadBuffer(queue, output_buffer, CL_TRUE, 0, sizeof(float) * ary_size * ary_size, output, 0, NULL, NULL);
   cl_error_check("Couldn't read the buffer", err);


   /* Release */
   clReleaseKernel(kernel);
   clReleaseMemObject(output_buffer);
   clReleaseMemObject(input_buffer); 

   free(input);
   free(output);

   timer_stop(TOTAL_RESPONSE_TIME);
   double total_response_time = timer_read(TOTAL_RESPONSE_TIME);

   return 0;
}

void set_affinity(int affinity){
   cpu_set_t mask;
   CPU_ZERO(&mask);
   CPU_SET(affinity, &mask);
   sched_setaffinity(0, sizeof(mask), &mask);

   return;
}

void set_scheduler(int priority){
   struct sched_param sp;
   memset(&sp, 0, sizeof(sp));
   sp.sched_priority = priority;
   sched_setscheduler(0, SCHED_FIFO, &sp);   

   return;
}

void create_matrix(float* matrix, int ary_size){
   for(int i = 0; i < ary_size * ary_size; i++){
      matrix[i] = i+1;
   }
}

void print_matrix(float* matrix, int ary_size){
   for(int i = 0; i < ary_size*ary_size; i++){
      printf("%f\t", matrix[i]);
      if(i%ary_size == ary_size-1) printf("\n");
   }
}

/* Find a GPU or CPU associated with the first available platform */ 
cl_device_id create_device() {

   cl_platform_id platform;
   cl_device_id dev;
   int err;

   /* Identify a platform */
   err = clGetPlatformIDs(1, &platform, NULL);
   if(err < 0) {
      perror("Couldn't identify a platform");
      exit(1);
   } 


   // Access a device
   // GPU
   err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 2, &dev, NULL);
   if(err == CL_DEVICE_NOT_FOUND) {
      // CPU
      err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 2, &dev, NULL);
   }

   if(err < 0) {
      perror("Couldn't access any devices");
      exit(1);   
   }

   return dev;
}

cl_program build_program(cl_context ctx, cl_device_id dev, const char* filename) {

   cl_program program;
   FILE *program_handle;
   char *program_buffer, *program_log;
   size_t program_size, log_size;
   int err;

   /* Read program file and place content into buffer */
   program_handle = fopen(filename, "r");
   if(program_handle == NULL) {
      perror("Couldn't find the program file");
      exit(1);
   }
   fseek(program_handle, 0, SEEK_END);
   program_size = ftell(program_handle);
   rewind(program_handle);
   program_buffer = (char*)malloc(program_size + 1);
   program_buffer[program_size] = '\0';
   fread(program_buffer, sizeof(char), program_size, program_handle);
   fclose(program_handle);

   /* Create program from file 
   */
   program = clCreateProgramWithSource(ctx, 1, 
      (const char**)&program_buffer, &program_size, &err);
   if(err < 0) {
      perror("Couldn't create the program");
      exit(1);
   }
   free(program_buffer);

   /* Build program 

   The fourth parameter accepts options that configure the compilation. 
   These are similar to the flags used by gcc. For example, you can 
   define a macro with the option -DMACRO=VALUE and turn off optimization 
   with -cl-opt-disable.
   */
   err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
   if(err < 0) {

      /* Find size of log and print to std output */
      clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG, 
            0, NULL, &log_size);
      program_log = (char*) malloc(log_size + 1);
      program_log[log_size] = '\0';
      clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG, 
            log_size + 1, program_log, NULL);
      printf("%s\n", program_log);
      free(program_log);
      exit(1);
   }

   return program;
}

void cl_error_check(const char* text, cl_int err){
   if(err < 0) {
      perror(text);
      printf("err # : %d\n", err);
      exit(1);
   }
}