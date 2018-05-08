#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "backprop.h"
#include "omp.h"
#include <time.h>

extern char *strcpy();
extern void exit();

int layer_size = 0;

void backprop_face()
{
  BPNN *net;
  int i;
  float out_err, hid_err;
  net = bpnn_create(layer_size, 16, 1); // (16, 1 can not be changed)
  
  printf("Input layer size : %d\n", layer_size);
  load(net);
  //entering the training kernel, only one iteration
  printf("Starting training kernel\n");
clock_t start = clock();
  bpnn_train_kernel(net, &out_err, &hid_err);
  bpnn_free(net);
clock_t end = clock();
double gpu_time_used = (double(end -start))/CLOCKS_PER_SEC;

  printf("\nFinish the training for one iteration\n");
printf("\n time consumed by GPU is %f",gpu_time_used);
}

int setup(int argc, char **argv)
{
	
  int seed;

  if (argc!=2){
  fprintf(stderr, "usage: backprop <num of input elements>\n");
  exit(0);
  }
  layer_size = atoi(argv[1]);
  if (layer_size%16!=0){
  fprintf(stderr, "The number of input points must be divided by 16\n");
  exit(0);
  }
  

  seed = 7;   
  bpnn_initialize(seed);
  backprop_face();

  exit(0);
}
