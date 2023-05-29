#ifndef __ERRORS__
#define __ERRORS__

#include <stdio.h>

#define EXIT_ARGS             1
#define EXIT_CUDA_DEV         2
#define EXIT_POINT_IOB        3
#define EXIT_INVALID_INFILE   4

void printErrDesc (int errn) {
  switch (errn) {
  case EXIT_ARGS:
    fprintf(stderr, "Invalid or missing argument: ");
    break;
  case EXIT_CUDA_DEV:
    fprintf(stderr, "There are no available device(s) that support CUDA\n");
    break;
  case EXIT_POINT_IOB:
    fprintf(stderr, "Point index out of bounds\n");
    break;
  case EXIT_INVALID_INFILE:
    fprintf(stderr, "Invalid input file\n");
    break;
  default:
    fprintf(stderr, "No error description\n");
    break;
  }
}

#endif