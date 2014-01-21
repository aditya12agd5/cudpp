// -------------------------------------------------------------
// cuDPP -- CUDA Data Parallel Primitives library
// -------------------------------------------------------------
// $Revision$
// $Date$
// ------------------------------------------------------------- 
// This source code is distributed under the terms of license.txt in
// the root directory of this source distribution.
// ------------------------------------------------------------- 

/*
* 
* This is a basic example of how to use the string sort implementations
* in the CUDPP library.
* 
* Usage: ./stringSortCUDPP filename write_output
*
* filename: file to sort with each string on separate line
* write_output: if 0, no output file is written
* 		if 1, output file with suffix _cudpp_output is written
*/

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime_api.h>
#include <time.h>

// includes, project
#include "cudpp.h"
#include <string>

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void runTest( int argc, char** argv);

int printSortedOutput(unsigned int *valuesSorted, unsigned char* stringVals, 
		int numElements, int stringSize, char inputFile[500]) {
	int retval = 0;
	char outFile[500];

	sprintf(outFile,"%s_cudpp_output",inputFile);	
	printf("[DEBUG] writing to output file %s\n", outFile);

	FILE *fp = fopen(outFile,"w");
	for(unsigned int i = 0; i < numElements; ++i) {
		unsigned int index = valuesSorted[i];
		if(index > stringSize) { 
			printf("[ERROR] index %d exceeds global string array size %d\n", index, stringSize);
			return 1;
		}
		while(true) { 
			char ch;
			ch = (char)(stringVals[index]);
			if(ch == '\0') break;
			fprintf(fp,"%c",ch);
			index++;
		}
		fprintf(fp,"\n");
	}	
	return retval;
}

double calculateDiff (struct timespec t1, struct timespec t2) { 
	return (((t1.tv_sec - t2.tv_sec)*1000.0) + (((t1.tv_nsec - t2.tv_nsec)*1.0)/1000000.0));
}



////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main( int argc, char** argv) 
{
    runTest( argc, argv);
}

////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
void
runTest( int argc, char** argv) 
{
	
    struct timespec tsetup1, tsetup2;
    clock_gettime(CLOCK_MONOTONIC, &tsetup1);
    
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        fprintf(stderr, "error: no devices supporting CUDA.\n");
        exit(EXIT_FAILURE);
    }
    int dev = 0;

    if(argc!=3) { 
    	printf("[DEBUG] Correct usage : ./stringSortCUDPP [filename] [write_output]\n");
    	exit(EXIT_FAILURE);
    }
    char inputFile[500];
    sprintf(inputFile,"%s",argv[1]);
    printf("[DEBUG] InputFile : %s\n", inputFile);

    int writeOutput = atoi(argv[2]);
    printf("[DEBUG] writeOutput : %d\n", writeOutput);

    if (dev < 0) dev = 0;
    if (dev > deviceCount-1) dev = deviceCount - 1;
    cudaSetDevice(dev);

    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, dev) == cudaSuccess)
    {
        printf("Using device %d:\n", dev);
        printf("%s; global mem: %dB; compute v%d.%d; clock: %d kHz\n",
               prop.name, (int)prop.totalGlobalMem, (int)prop.major, 
               (int)prop.minor, (int)prop.clockRate);
    }

     clock_gettime(CLOCK_MONOTONIC, &tsetup2);
     printf("[DEBUG] gpu setup time (ms) : %lf\n", calculateDiff(tsetup2, tsetup1));

   // Initialize the CUDPP Library
    CUDPPHandle theCudpp;
    cudppCreate(&theCudpp);

    CUDPPConfiguration config;
    config.algorithm = CUDPP_SORT_STRING;
    config.datatype = CUDPP_UINT;
    config.options = CUDPP_OPTION_FORWARD;
 
    unsigned int *h_valSend, *d_values;
    unsigned char *d_stringVals;
    unsigned char *h_stringVals;
    unsigned int maxStringLength = 40;
    unsigned int stringSize = 0;
    unsigned int numElements = 10000000;
    unsigned int MAXBYTES = numElements*maxStringLength;
    //increase MAXBYTES for larger input size

    h_valSend      = (unsigned int*)malloc(numElements*sizeof(unsigned int));
    h_stringVals = (unsigned char*) malloc(sizeof(unsigned char)*maxStringLength*numElements);
    
    struct timespec tread1, tread2; 
    clock_gettime(CLOCK_MONOTONIC, &tread1);
    
    FILE *fp = fopen(inputFile,"rb");
    if(fp == NULL) { 
    	printf("[DEBUG] cannot load %s\n", inputFile);
	exit(EXIT_FAILURE);
    }
    
    /* read the input file in one fread */
    char *INBUF = (char*)malloc((sizeof(char)*MAXBYTES)+2);
    MAXBYTES = fread(INBUF, 1, MAXBYTES, fp);
    printf("[DEBUG] read bytes %u\n", MAXBYTES);
    
    unsigned int index = 0;
    numElements = 0;
    unsigned int i = 0;
    char c; 

    /* pack the strings of the read file into array of unsigned chars with \0 as delimiter */
    //TODO: can pass newline character as termC also.
    while(i < MAXBYTES) { 
	    h_valSend[numElements] = index;
	    while(true) {
		    c = INBUF[i];
		    if(c == '\n') {
		   	h_stringVals[index] = 0;
			i++;
			index++;
			break;
		    }
		    h_stringVals[index] = c;
		    i++;
		    index++;
	    }
	    numElements++;
    }
 
    stringSize = index;

    clock_gettime(CLOCK_MONOTONIC, &tread2);
    printf("[DEBUG] file read time (ms) : %lf\n", calculateDiff(tread2, tread1));
    printf("[DEBUG] number of elements read %d and stringSize %d\n", numElements, stringSize); 

    /* allocate device arrays */
    cudaMalloc((void **)&d_values, numElements*sizeof(unsigned int));
    cudaMalloc((void **)&d_stringVals, stringSize*sizeof(unsigned char));
    
    cudaMemcpy(d_stringVals, h_stringVals, stringSize*sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_values, h_valSend, numElements * sizeof(unsigned int), cudaMemcpyHostToDevice);

  
    CUDPPHandle plan;   
    CUDPPResult result = cudppPlan(theCudpp, &plan, config, numElements, 1, 0);     
    if(result != CUDPP_SUCCESS) {
	    printf("Error in plan creation\n");
	    cudppDestroyPlan(plan);
	    return;
    }

    struct timespec t1, t2;
    clock_gettime(CLOCK_MONOTONIC, &t1);

    /* string sort Davidson et al. InPar'12 
    cudppStringSort(plan, d_stringVals, d_values, 0, numElements, stringSize);
    */

    /* string sort Deshpande et al. HiPC'13 */
    cudppStringSortRadix( d_stringVals, d_values, 0, numElements, stringSize);
    
    clock_gettime(CLOCK_MONOTONIC, &t2);
    printf("[DEBUG] sort time (ms) : %lf\n", calculateDiff(t2, t1));

    cudaMemcpy((void*)h_valSend, (void*)d_values, numElements * sizeof(unsigned int), 
			    cudaMemcpyDeviceToHost) ;

    result = cudppDestroyPlan(plan);
	
    if (result != CUDPP_SUCCESS) {   
        printf("Error destroying CUDPPPlan for StringSort\n");
        return;
    }

    /* generate an output file with extension _cudpp_output if writeOutput is set to 1 */
    if(writeOutput == 1) { 
	    int retVal = printSortedOutput(h_valSend, h_stringVals, numElements, stringSize, inputFile);
	    printf("string sort %s\n", (retVal == 0) ? "EXECUTED" : "FAILED");
    }
    
    cudaFree(d_values);
    cudaFree(d_stringVals);
	
    free(h_valSend);
    free(h_stringVals);
}
