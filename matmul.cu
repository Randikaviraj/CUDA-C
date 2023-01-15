/* Program to do matrix multiplication in cuda
This program generates two matrices and multiply them*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "error.cuh"

//Dimensions for matrix1
#define ROWS1 10
#define COLS1 20

//DImensions for matrix2
#define ROWS2 20
#define COLS2 15

/** CUDA kernel to do matrix multiplication**/
__global__ void matMul(int *matC_cuda, int *matA_cuda, int *matB_cuda){
	
	//derive the row and column based on thread configuration
	int row = blockIdx.y*blockDim.y + threadIdx.y;
	int col = blockIdx.x*blockDim.x + threadIdx.x;
	
	//Limit calculations for valid indices
	if(row < ROWS1 && col < COLS2){
	
		int prod=0;
		int k;
		for(k=0;k<COLS1;k++){
			prod=prod+matA_cuda[row*COLS1+k]*matB_cuda[k*COLS2+col];
		}
		matC_cuda[row*COLS2+col]=prod;	
		
	}
	
}

int main(){
	
	//check whether dimensions are valid for matrix multiplication
	if(COLS1!=ROWS2){
		printf("Matrix dimensions are invalid for matrix multiplication\n");
		exit(1);
	}
	
	//Initialize arrays in RAM
	int matA[ROWS1*COLS1];
	int matB[ROWS2*COLS2];
	int matC[ROWS1*COLS2];	
	
	//generate some values for matrixA
	int i,j;
	for(i=0;i<ROWS1;i++){
		for(j=0;j<COLS1;j++){
			matA[i*COLS1+j]=i+j;
		}
	}

	//print the matA
	printf("Matrix A : \n");
	for(i=0;i<ROWS1;i++){
		for(j=0;j<COLS1;j++){
			printf("%5d ",matA[i*COLS1+j]);
		}
		printf("\n");
	}		
	printf("\n");

	
	//generate values for matrixB
	for(i=0;i<ROWS2;i++){
		for(j=0;j<COLS2;j++){
			matB[i*COLS2+j]=i-j;
		}
	}

	//print the matB
	printf("Matrix B : \n");
	for(i=0;i<ROWS2;i++){
		for(j=0;j<COLS2;j++){
			printf("%5d ",matB[i*COLS2+j]);
		}
		printf("\n");
	}	
	printf("\n");

	/********************************** CUDA stuff starts here *******************************/

	//start measuring time
	cudaEvent_t start,stop;
	float elapsedtime;
	cudaEventCreate(&start);
	cudaEventRecord(start,0);	
	
	//pointers for memory allocation in cudaa
	int *matA_cuda;
	int *matB_cuda;
	int *matC_cuda;
	
	//allocate memory in cuda
	cudaMalloc((void **)&matA_cuda,sizeof(int)*ROWS1*COLS1); checkCudaError();
	cudaMalloc((void **)&matB_cuda,sizeof(int)*ROWS2*COLS2); checkCudaError();
	cudaMalloc((void **)&matC_cuda,sizeof(int)*ROWS1*COLS2); checkCudaError();
	
	//copy memory from ram to cuda
	cudaMemcpy(matA_cuda,matA,sizeof(int)*ROWS1*COLS1,cudaMemcpyHostToDevice); checkCudaError();
	cudaMemcpy(matB_cuda,matB,sizeof(int)*ROWS2*COLS2,cudaMemcpyHostToDevice); checkCudaError();
	
	//multiply the matrices in cuda
	dim3 threadsPerBlock(16,16);
	dim3 numBlocks(ceil(COLS2/(float)16),ceil(ROWS1/(float)16));
	matMul<<<numBlocks,threadsPerBlock>>>(matC_cuda,matA_cuda,matB_cuda);
	checkCudaError();
	
	//copy the answer back from cuda to ram
	cudaMemcpy(matC,matC_cuda,sizeof(int)*ROWS1*COLS2,cudaMemcpyDeviceToHost); checkCudaError();

	//free the cuda memory
	cudaFree(matA_cuda); checkCudaError();
	cudaFree(matB_cuda); checkCudaError();
	cudaFree(matC_cuda); checkCudaError();
	
	//end measuring time
	cudaEventCreate(&stop);
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedtime,start,stop);
	
	/********************** CUDA stuff ends here ********************************/
	
	//print the answer
	printf("Answer : \n");	
	for(i=0;i<ROWS1;i++){
		for(j=0;j<COLS2;j++){
			printf("%5d ",matC[i*COLS2+j]);
		}
		printf("\n");
	}	

	fprintf(stderr,"Time spent for operation on CUDA(Including memory allocation and copying) is %1.5f seconds\n",elapsedtime/(float)1000); 	
	
	return 0;

}
