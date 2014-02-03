// -------------------------------------------------------------
// CUDPP -- CUDA Data Parallel Primitives library
// -------------------------------------------------------------
// $Revision$
// $Date$
// ------------------------------------------------------------- 
// This source code is distributed under the terms of license.txt 
// in the root directory of this source distribution.
// ------------------------------------------------------------- 

/**
* @file
* stringsort_app.cu
*   
* @brief CUDPP application-level merge sorting routines
*/

/** @addtogroup cudpp_app 
* @{
*/

/** @name StringSort Functions
* @{
*/

#include "cuda_util.h"
#include "cudpp.h"
#include "cudpp_util.h"
#include "cudpp_stringsort.h"
#include "kernel/stringsort_kernel.cuh"
#include "limits.h"

#define BLOCKSORT_SIZE 1024
#define DEPTH 8

void dotAdd(unsigned int* d_address,	
	   unsigned int* numSpaces,
	   unsigned int* packedAddress,
	   size_t numElements,
	   size_t stringArrayLength)
{
	int numThreads = 128;
	int numBlocks = (numElements+numThreads-1)/numThreads;
	dotAddInclusive<<<numBlocks, numThreads>>>(numSpaces, d_address, packedAddress, numElements, stringArrayLength);
}

void calculateAlignedOffsets(unsigned int* d_address,	
							 unsigned int* numSpaces,
							 unsigned char* d_stringVals, 
							 unsigned char termC,
							 size_t numElements,
							 size_t stringArrayLength) {
	int numThreads = 128;
	int numBlocks = (numElements+numThreads-1)/numThreads;

	alignedOffsets<<<numBlocks, numThreads>>>(numSpaces, d_address, d_stringVals, termC, numElements, stringArrayLength);

}

struct get_segment_bytes {
        __host__ __device__
        unsigned int operator()(const unsigned long long int& x) const {
                return (unsigned int)(x >> 56);
        }
};

/** @brief This is a wrapper function which internally calls
* the radix sort based string sort main logic.  
*
* It initially allocates a few temporary arrays for string sort,
* then calls string sort logic and copies sorted string positions 
* back to the addresses array.
*
* @param[in] d_arrayStringVals Strings which are delimited by termC.
* @param[in,out] d_arrayAddress Addresses of successive string characters for tie-breaks.
* @param[in] termC Termination character for our strings.
* @param[in] numElements Number of elements in the sort.
* @param[in] stringArrayLength The size of our string array containing characters delimited by termc.
* @param[in] plan Configuration information for string sort.
**/
void cudppStringSortRadixWrapper(
	unsigned char* d_arrayStringVals, 
	unsigned int* d_arrayAddress, 
	unsigned char termC, 
	size_t numElements,
	size_t stringArrayLength,
	const       CUDPPStringSortPlan *plan) 
{

	//allocate some temporary thrust arrays for string sort logic
        thrust::device_vector<unsigned int> d_static_index(numElements);
	thrust::sequence(d_static_index.begin(), d_static_index.end());
	
        thrust::device_vector<unsigned int> d_output_valIndex(numElements);

	//call the main string sort logic
	cudppStringSortRadixMain(d_arrayStringVals, d_arrayAddress, plan->m_packedStringVals, d_static_index, 
		d_output_valIndex, numElements, stringArrayLength);

	//copy sorted output addresses back to d_arrayAddress
	thrust::copy(d_output_valIndex.begin(), d_output_valIndex.end(), thrust::device_pointer_cast(d_arrayAddress));
}

/** @brief This function contains the main radix sort based
* string sort logic.
*   
*
* It iteratively sorts strings from left to right. It loads string 
* characters into MCUs along with segment information (history of 
* previous sort steps), singleton strings are eliminated per iteration
* and minimum bytes are used to record segment information. 
* Refer to: Aditya Deshpande and P J Narayanan, "Can GPUs Sort Strings Efficiently" (HiPC'13)
* for more details.
*
* @param[in] d_arrayStringVals Strings which are delimited by termC.
* @param[in] d_array_valIndex Addresses of successive string characters for tie-breaks.
* @param[in] d_array_segment_keys String Keys packed into 8 byte unsigned long long int.
* @param[in] d_array_static_index Temporary Array to help singleton elimination in string sort logic.
* @param[in,out] d_output_valIndex Addresses of the strings in sorted order are returned through this array.
* @param[in] numElements Number of elements in the sort.
* @param[in] stringArrayLength The size of our string array containing characters delimited by termC.
**/
void cudppStringSortRadixMain(
	unsigned char* d_array_stringVals,
	unsigned int* d_array_valIndex, 
	unsigned long long int* d_array_segment_keys,
	thrust::device_vector<unsigned int> d_static_index,
	thrust::device_vector<unsigned int> &d_output_valIndex,
	size_t numElements, 
	size_t stringArrayLength) {

	int MAX_THREADS_PER_BLOCK = 512;

        //8 characters are already loaded for first sort step, 
	// load successive ones after that 
	unsigned int charPosition = 8;

	//no segment information for first sort step, 
	//future sort steps will have non-zero segment information bytes
        unsigned int segmentBytes = 0;
        
	unsigned int lastSegmentID = 0;
	unsigned int numSorts = 0;

	//convert allocated arrays to device_ptr to enable use of thrust functions on them
	thrust::device_ptr<unsigned int> d_valIndex = thrust::device_pointer_cast(d_array_valIndex);
	thrust::device_ptr<unsigned long long int> d_segment_keys = thrust::device_pointer_cast(d_array_segment_keys);

	while(true) 
	{ 

		//sort 8 byte consisting of segment information and then string characters
                thrust::sort_by_key(
			d_segment_keys,
                        d_segment_keys + numElements,
                        d_valIndex
                );		
		
		numSorts++;

                thrust::device_vector<unsigned long long int> d_segment_keys_out(numElements, 0);
	
                unsigned int *d_array_static_index = thrust::raw_pointer_cast(&d_static_index[0]);
                unsigned int *d_array_output_valIndex = thrust::raw_pointer_cast(&d_output_valIndex[0]);

                unsigned long long int *d_array_segment_keys_out = thrust::raw_pointer_cast(&d_segment_keys_out[0]);

                int numBlocks = 1;
                int numThreadsPerBlock = numElements/numBlocks;

                if(numThreadsPerBlock > MAX_THREADS_PER_BLOCK) 
		{
                        numBlocks = (int)ceil(numThreadsPerBlock/(float)MAX_THREADS_PER_BLOCK);
                        numThreadsPerBlock = MAX_THREADS_PER_BLOCK;
                }
                dim3 grid(numBlocks, 1, 1);
                dim3 threads(numThreadsPerBlock, 1, 1);

		//load successive 8 characters in a temporary array (d_array_segment_keys_out)
                cudaThreadSynchronize();
                
		
		hipcFindSuccessorKernel<<<grid, threads, 0>>>(d_array_stringVals, d_array_segment_keys, d_array_valIndex, d_array_segment_keys_out, numElements, stringArrayLength, charPosition, segmentBytes);
               	
		cudaThreadSynchronize();

		//change char position to reflect more characters loaded
                charPosition+=7;

                thrust::device_vector<unsigned int> d_temp_vector(numElements);
                thrust::device_vector<unsigned int> d_segment(numElements);
                thrust::device_vector<unsigned int> d_stencil(numElements);
                thrust::device_vector<unsigned int> d_map(numElements);

                unsigned int *d_array_temp_vector = thrust::raw_pointer_cast(&d_temp_vector[0]);
                unsigned int *d_array_segment = thrust::raw_pointer_cast(&d_segment[0]);
                unsigned int *d_array_stencil = thrust::raw_pointer_cast(&d_stencil[0]);


                thrust::transform(d_segment_keys_out.begin(), d_segment_keys_out.begin() + numElements, d_temp_vector.begin(), get_segment_bytes());

		thrust::inclusive_scan(d_temp_vector.begin(), d_temp_vector.begin() + numElements, d_segment.begin());

		//eliminate the singleton (bucket size == 1) strings from sort problem
		//write their position to final output (d_array_output_valIndex)
                cudaThreadSynchronize();
                hipcEliminateSingletonKernel<<<grid, threads, 0>>>(d_array_output_valIndex, d_array_valIndex, d_array_static_index, d_array_temp_vector, d_array_stencil, numElements);
                cudaThreadSynchronize();


		thrust::exclusive_scan(d_stencil.begin(), d_stencil.begin() + numElements, d_map.begin());

                thrust::scatter_if(d_segment.begin(), d_segment.begin() + numElements, d_map.begin(),
                                d_stencil.begin(), d_temp_vector.begin());
                thrust::copy(d_temp_vector.begin(), d_temp_vector.begin() + numElements, d_segment.begin());

                thrust::scatter_if(d_valIndex, d_valIndex + numElements, d_map.begin(),
                                d_stencil.begin(), d_temp_vector.begin());

                thrust::copy(d_temp_vector.begin(), d_temp_vector.begin() + numElements, d_valIndex);

                thrust::scatter_if(d_static_index.begin(), d_static_index.begin() + numElements, d_map.begin(),
                                d_stencil.begin(), d_temp_vector.begin());
                thrust::copy(d_temp_vector.begin(), d_temp_vector.begin() + numElements, d_static_index.begin());

                thrust::scatter_if(d_segment_keys_out.begin(), d_segment_keys_out.begin() + numElements, d_map.begin(),
                                d_stencil.begin(), d_segment_keys);
                thrust::copy(d_segment_keys, d_segment_keys + numElements, d_segment_keys_out.begin());


                numElements = *(d_map.begin() + numElements - 1) + *(d_stencil.begin() + numElements - 1);
		if(numElements != 0) 
		{
			//compute the minimum bytes required for segment information in next sort step
			lastSegmentID = *(d_segment.begin() + numElements - 1);
                }

                d_temp_vector.clear();
                d_temp_vector.shrink_to_fit();

                d_stencil.clear();
                d_stencil.shrink_to_fit();

                d_map.clear();
                d_map.shrink_to_fit();

		//continue sort if non-zero number of elements remain after singleton elimination, else terminate
		if(numElements == 0) 
		{
			break;
		}

		//reset the char position since some bytes would now be used by segment information
                segmentBytes = (int) ceil(((float)(log2((float)lastSegmentID+2))*1.0)/8.0);
                charPosition-=(segmentBytes-1);

		int numBlocks1 = 1;
                int numThreadsPerBlock1 = numElements/numBlocks1;

                if(numThreadsPerBlock1 > MAX_THREADS_PER_BLOCK) 
		{
                        numBlocks1 = (int)ceil(numThreadsPerBlock1/(float)MAX_THREADS_PER_BLOCK);
                        numThreadsPerBlock1 = MAX_THREADS_PER_BLOCK;
                }
                dim3 grid1(numBlocks1, 1, 1);
                dim3 threads1(numThreadsPerBlock1, 1, 1);

		//pack segment information and string characters (few of the 8 loaded in find successor step)
		//into d_array_segment_keys
                cudaThreadSynchronize();
                hipcRearrangeSegMCUKernel<<<grid1, threads1, 0>>>(d_array_segment_keys, d_array_segment_keys_out, d_array_segment, segmentBytes, numElements);
                cudaThreadSynchronize();

		//perform future sort on segment information and successive characters
	}
	
	return;
}

/** @brief This function calls a kernel to pack first
* 8 string characters to unsigned long long int.
* 
*   
* @param[in] d_stringVals Strings which are delimited by termC.
* @param[in] d_address Addresses of successive string characters for tie-breaks.
* @param[in,out] d_packedStringVals String Keys packed into 8 byte unsigned long long int are returned.
* @param[in] termC Termination character for our strings.
* @param[in] numElements Number of elements in the sort.
* @param[in] stringArrayLength The size of our string array containing characters delimited by termC.
**/
void cudppStringSortRadixSetup( unsigned char* d_stringVals,
		unsigned int* d_address,
		unsigned long long int* d_packedStringVals,
		unsigned char termC,
		size_t numElements,
		size_t stringArrayLength) 
{ 

		int numBlocks = 1;
                int numThreadsPerBlock = numElements/numBlocks;

                if(numThreadsPerBlock > 512) 
		{
                        numBlocks = (int)ceil(numThreadsPerBlock/(float)512);
                        numThreadsPerBlock = 512;
                }
                dim3 grid(numBlocks, 1, 1);
                dim3 threads(numThreadsPerBlock, 1, 1);

		//Embarrassingly parallel kernel to pack first 8 string characters (d_stringVals)
		//into an unsigned long long int array (d_packedStringVals).
		hipcPackStringsKernel<<<grid, threads, 0>>>(d_stringVals, d_address, d_packedStringVals, termC, numElements, stringArrayLength);
		return;
		
}

void packStrings(unsigned int* packedStrings, 
						 unsigned char* d_stringVals, 
						 unsigned int* d_keys, 						 
						 unsigned int* packedAddress, 
						 unsigned int* address, 
						 size_t numElements,
						 size_t stringArrayLength,
						 unsigned char termC)
{
	unsigned int numThreads = 128;
	unsigned int numBlocks = (numElements + numThreads - 1)/numThreads;

	//Each thread handles one string (irregular parrallelism) other option is to do per character (set of chars)
	//but that requires a binary search per character. Efficiency depends on the dataset
	alignString<<<numBlocks, numThreads>>>(packedStrings, d_stringVals, packedAddress, address, numElements, stringArrayLength, termC);
	createKeys<<<numBlocks, numThreads>>>(d_keys, packedStrings, packedAddress, numElements);

}


void unpackStrings(unsigned int* packedAddress,
				   unsigned int* packedAddressRef,
				   unsigned int* address,
				   unsigned int* addressRef,				   
				   size_t numElements)
{
	unsigned int numThreads = 128;
	unsigned int numBlocks = (numElements + numThreads - 1)/numThreads;

	unpackAddresses<<<numBlocks, numThreads>>>(packedAddress, packedAddressRef, address, addressRef, numElements);
}

/** @brief Performs merge sor utilzing three stages. 
* (1) Blocksort, (2) simple merge and (3) multi merge on a 
* set of strings 
* 
* @param[in,out] pkeys Keys (first four characters of string) to be sorted.
* @param[in,out] pvals Addresses of string locations for tie-breaks
* @param[out] stringVals global string value array (four characters stuffed into a uint)
* @param[in] numElements Number of elements in the sort.
* @param[in] stringArrayLength The size of our string array in uints (4 chars per uint)
* @param[in] plan Configuration information for mergesort.
* @param[in] termC Termination character for our strings
**/
void runStringSort(unsigned int *pkeys, 
				   unsigned int *pvals,
				   unsigned int *stringVals,
				   size_t numElements,
				   size_t stringArrayLength,
				   unsigned char termC,
				   const CUDPPStringSortPlan *plan)
{
	int numPartitions = (numElements+BLOCKSORT_SIZE-1)/BLOCKSORT_SIZE;
	int numBlocks = numPartitions/2;
	int partitionSize = BLOCKSORT_SIZE;
	

	

	unsigned int swapPoint = plan->m_swapPoint;
	unsigned int subPartitions = plan->m_subPartitions;	


	
	int numThreads = 128;	

	blockWiseStringSort<unsigned int, DEPTH> <<<numPartitions, BLOCKSORT_SIZE/DEPTH, 2*(BLOCKSORT_SIZE)*sizeof(unsigned int)>>>
		                     (pkeys, pvals, stringVals, BLOCKSORT_SIZE, numElements, stringArrayLength, termC);
	

	int mult = 1; int count = 0;

	CUDA_SAFE_CALL(cudaThreadSynchronize());
	//we run p stages of simpleMerge until numBlocks <= some Critical level
	while(numPartitions > swapPoint || (partitionSize*mult < 16384 && numPartitions > 1)/* && numPartitions > 1*/)
	{	
		//printf("Running simple merge for %d partitions of size %d\n", numPartitions, partitionSize*mult);
		numBlocks = (numPartitions&0xFFFE);	    
		if(count%2 == 0)
		{ 				
			simpleStringMerge<unsigned int, 2>
				<<<numBlocks, CTASIZE_simple, sizeof(unsigned int)*(2*INTERSECT_B_BLOCK_SIZE_simple+4)>>>(pkeys, plan->m_tempKeys, 				
				pvals, plan->m_tempAddress, stringVals, partitionSize*mult, numElements, count, stringArrayLength, termC);		

			if(numPartitions%2 == 1)
			{			

				int offset = (partitionSize*mult*(numPartitions-1));
				int numElementsToCopy = numElements-offset;												
				simpleCopy<unsigned int>
					<<<(numElementsToCopy+numThreads-1)/numThreads, numThreads>>>(pkeys, pvals, plan->m_tempKeys, plan->m_tempAddress, offset, numElementsToCopy);
			}
		}
		else
		{			
			simpleStringMerge<unsigned int, 2>
				<<<numBlocks, CTASIZE_simple, sizeof(unsigned int)*(2*INTERSECT_B_BLOCK_SIZE_simple+4)>>>(plan->m_tempKeys, pkeys, 				
				plan->m_tempAddress, pvals, stringVals, partitionSize*mult, numElements, count, stringArrayLength, termC);		
			
			if(numPartitions%2 == 1)
			{			
				int offset = (partitionSize*mult*(numPartitions-1));
				int numElementsToCopy = numElements-offset;						
				simpleCopy<unsigned int>
					<<<(numElementsToCopy+numThreads-1)/numThreads, numThreads>>>(plan->m_tempKeys, plan->m_tempAddress, pkeys, pvals, offset, numElementsToCopy);
			}
		}

		mult*=2;
		count++;
		numPartitions = (numPartitions+1)/2;		
	}				


	
	
	//End of simpleMerge, now blocks cooperate to merge partitions
	while (numPartitions > 1)
	{				
		numBlocks = (numPartitions&0xFFFE);	 
		int secondBlocks = ((numBlocks)*subPartitions+numThreads-1)/numThreads;			
		if(count%2 == 1)
		{								
			findMultiPartitions<unsigned int>
				<<<secondBlocks, numThreads>>>(plan->m_tempKeys, plan->m_tempAddress, stringVals, subPartitions, numBlocks, partitionSize*mult, plan->m_partitionStartA, plan->m_partitionSizeA, 
				plan->m_partitionStartB, plan->m_partitionSizeB, numElements, stringArrayLength, termC);			
			

			//int lastSubPart = getLastSubPart(numBlocks, subPartitions, partitionSize, mult, numElements);
			CUDA_SAFE_CALL(cudaThreadSynchronize());
			stringMergeMulti<unsigned int, DEPTH_multi>
				<<<numBlocks*subPartitions, CTASIZE_multi, (2*INTERSECT_B_BLOCK_SIZE_multi+4)*sizeof(unsigned int)>>>(plan->m_tempKeys, pkeys, plan->m_tempAddress, 
				pvals, stringVals, subPartitions, numBlocks, plan->m_partitionStartA, plan->m_partitionSizeA, plan->m_partitionStartB, plan->m_partitionSizeB, mult*partitionSize, 
				count, numElements, stringArrayLength, termC);
			CUDA_SAFE_CALL(cudaThreadSynchronize());
			if(numPartitions%2 == 1)
			{			
				int offset = (partitionSize*mult*(numPartitions-1));
				int numElementsToCopy = numElements-offset;				
				simpleCopy<unsigned int>
					<<<(numElementsToCopy+numThreads-1)/numThreads, numThreads>>>(plan->m_tempKeys, plan->m_tempAddress, pkeys, pvals, offset, numElementsToCopy);
			}

		}
		else
		{

			findMultiPartitions<unsigned int>
				<<<secondBlocks, numThreads>>>(pkeys, pvals, stringVals, subPartitions, numBlocks, partitionSize*mult, plan->m_partitionStartA, plan->m_partitionSizeA, 
				plan->m_partitionStartB, plan->m_partitionSizeB, numElements, stringArrayLength, termC);											
			CUDA_SAFE_CALL(cudaThreadSynchronize());
			//int lastSubPart = getLastSubPart(numBlocks, subPartitions, partitionSize, mult, numElements);
			stringMergeMulti<unsigned int, DEPTH_multi>
				<<<numBlocks*subPartitions, CTASIZE_multi, (2*INTERSECT_B_BLOCK_SIZE_multi+4)*sizeof(unsigned int)>>>(pkeys, plan->m_tempKeys, pvals, 
				plan->m_tempAddress, stringVals, subPartitions, numBlocks, plan->m_partitionStartA, plan->m_partitionSizeA, plan->m_partitionStartB, plan->m_partitionSizeB, mult*partitionSize, 
				count, numElements, stringArrayLength, termC);

			CUDA_SAFE_CALL(cudaThreadSynchronize());
			if(numPartitions%2 == 1)
			{			
				int offset = (partitionSize*mult*(numPartitions-1));
				int numElementsToCopy = numElements-offset;				
				simpleCopy<unsigned int>
					<<<(numElementsToCopy+numThreads-1)/numThreads, numThreads>>>(pkeys, pvals, plan->m_tempKeys, plan->m_tempAddress, offset, numElementsToCopy);
			}

		}


		count++;
		mult*=2;		
		subPartitions*=2;
		numPartitions = (numPartitions+1)/2;				
	}	

	if(count%2==1)
	{
		CUDA_SAFE_CALL(cudaMemcpy(pkeys, plan->m_tempKeys, numElements*sizeof(unsigned int), cudaMemcpyDeviceToDevice));
		CUDA_SAFE_CALL(cudaMemcpy(pvals, plan->m_tempAddress, numElements*sizeof(unsigned int), cudaMemcpyDeviceToDevice));
	}
	
}

#ifdef __cplusplus
extern "C" 
{
#endif


	/**
	* @brief From the programmer-specified sort configuration, 
	*        creates internal memory for performing the sort.
	* 
	* @param[in] plan Pointer to CUDPPStringSortPlan object
	**/
	void allocStringSortStorage(CUDPPStringSortPlan *plan)
	{
	
		if(!plan->m_stringSortRadix) 
		{ 

			CUDA_SAFE_CALL(cudaMalloc((void**)&plan->m_keys, sizeof(unsigned int)*plan->m_numElements));				
			CUDA_SAFE_CALL(cudaMalloc((void**)&plan->m_tempKeys,    sizeof(unsigned int)*plan->m_numElements));		
			CUDA_SAFE_CALL(cudaMalloc((void**)&plan->m_tempAddress,    sizeof(unsigned int)*plan->m_numElements));		
			CUDA_SAFE_CALL(cudaMalloc((void**)&plan->m_packedAddress, sizeof(unsigned int)*(plan->m_numElements+1)));
			CUDA_SAFE_CALL(cudaMalloc((void**)&plan->m_packedAddressRef, sizeof(unsigned int)*(plan->m_numElements)));		
			CUDA_SAFE_CALL(cudaMalloc((void**)&plan->m_addressRef, sizeof(unsigned int)*(plan->m_numElements)));		

			CUDA_SAFE_CALL(cudaMalloc((void**)&plan->m_spaceScan, sizeof(unsigned int)*(plan->m_numElements+1)));		
			CUDA_SAFE_CALL(cudaMalloc((void**)&plan->m_numSpaces, sizeof(unsigned int)*(plan->m_numElements+1)));		

			CUDA_SAFE_CALL(cudaMalloc((void**)&plan->m_partitionSizeA, sizeof(unsigned int)*(plan->m_swapPoint*plan->m_subPartitions*4)));		
			CUDA_SAFE_CALL(cudaMalloc((void**)&plan->m_partitionSizeB, sizeof(unsigned int)*(plan->m_swapPoint*plan->m_subPartitions*4)));		
			CUDA_SAFE_CALL(cudaMalloc((void**)&plan->m_partitionStartA, sizeof(unsigned int)*(plan->m_swapPoint*plan->m_subPartitions*4)));		
			CUDA_SAFE_CALL(cudaMalloc((void**)&plan->m_partitionStartB, sizeof(unsigned int)*(plan->m_swapPoint*plan->m_subPartitions*4)));	
		} 
		else 
		{
			CUDA_SAFE_CALL(cudaMalloc((void**)&plan->m_packedStringVals, sizeof(unsigned long long int)*(plan->m_numElements)));
		}
	}

	/** @brief Deallocates intermediate memory from allocStringSortStorage.
	*
	*
	* @param[in] plan Pointer to CUDPStringSortPlan object
	**/

	void freeStringSortStorage(CUDPPStringSortPlan* plan)
	{
		if(!plan->m_stringSortRadix) 
		{
			cudaFree(plan->m_keys);
			cudaFree(plan->m_packedAddress);
			cudaFree(plan->m_packedAddressRef);
			cudaFree(plan->m_tempKeys);
			cudaFree(plan->m_tempAddress);
			cudaFree(plan->m_addressRef);

			cudaFree(plan->m_numSpaces);
			cudaFree(plan->m_spaceScan);

			cudaFree(plan->m_partitionSizeA);
			cudaFree(plan->m_partitionSizeB);
			cudaFree(plan->m_partitionStartA);
			cudaFree(plan->m_partitionStartB);
		}
		else 
		{
			cudaFree(plan->m_packedStringVals);
		}
	}

	/** @brief Dispatch function to perform a sort on an array with 
	* a specified configuration.
	*
	* This is the dispatch routine which calls stringSort...() with 
	* appropriate template parameters and arguments as specified by 
	* the plan.
	* @param[in,out] keys Keys (first four chars of string) to be sorted.
	* @param[in,out] values Address of string values in array of null terminated strings
	* @param[in] stringVals Global string array
	* @param[in] numElements Number of elements in the sort.
	* @param[in] stringArrayLength The size of our string array in uints (4 chars per uint)
	* @param[in] termC Termination character for our strings
	* @param[in] plan Configuration information for mergeSort.	
	**/

	void cudppStringSortDispatch(unsigned int  *keys,
		                         unsigned int  *values,
		                         unsigned int  *stringVals,
		                         size_t numElements,
								 size_t stringArrayLength,
								 unsigned char termC,
		                         const CUDPPStringSortPlan *plan)
	{

		if(!plan->m_stringSortRadix) 
			runStringSort(keys, values, stringVals, numElements, stringArrayLength, termC, plan);
		else
			cudppStringSortRadixWrapper((unsigned char *) stringVals, values, termC, numElements, stringArrayLength, plan);

	}                            

#ifdef __cplusplus
}
#endif






/** @} */ // end stringsort functions
/** @} */ // end cudpp_app
