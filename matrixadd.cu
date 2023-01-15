#include <stdio.h>
#include <stdlib.h>

#define ROWS 5
#define COLS 10
#define SIZE ROWS*COLS

/* macro that calls gpuAssert with File and Line number */
#define checkCudaError() { gpuAssert(__FILE__, __LINE__); }

/* check whether the last CUDA function or CUDA kernel launch is erroneous and if yes an error message will be printed
and then the program will be aborted*/
void gpuAssert(const char *file, int line){

	cudaError_t code = cudaGetLastError();
	if (code != cudaSuccess) {
        fprintf(stderr, "Cuda error: %s \n in file : %s line number : %d", cudaGetErrorString(code), file, line );
        exit(1);
   }
}

//kernel that does the matrix addition. Just add each element to the respective one
__global__ void addMatrix(int *ans_cuda,int *matA_cuda,int *matB_cuda){
	
	/*blockDim.y gives the height of a block along y axis
	  blockDim.x gives the width of a block along x axis
	  blockIdx.y gives the index of the current block along the y axis
	  blockIdx.x gives the index of the current block along the x axis
	  threadIdx.y gives the index of the current thread in the current block along y axis
	  threadIdx.x gives the index of the current thread in the current block along x axis
	  */
	
	//calculate the row number based on block IDs and thread IDs
	int row = blockDim.y * blockIdx.y + threadIdx.y;
	//calculate the column number based on block IDs and thread IDs
	int col = blockDim.x * blockIdx.x + threadIdx.x;

	//to remove any indices beyond the size of the array
	if (row<ROWS && col <COLS){
		
		//conversion of 2 dimensional indices to single dimension
		int position = row*COLS + col;
	
		//do the calculation
		ans_cuda[position]=matA_cuda[position]+matB_cuda[position];
	
	}
}


int main(){

	int matA[ROWS][COLS];
	int matB[ROWS][COLS];
	int ans[ROWS][COLS];
	
	int i=0,j=0,k=0;
	for(i=0;i<ROWS;i++){
		for(j=0;j<COLS;j++){
			matA[i][j]=k;
			matB[i][j]=ROWS*COLS-k;
			k++;
		}
	}
	

/*************************CUDA STUFF STARTS HERE************************/	
	
	//variables for time measurements
	cudaEvent_t start,stop;
	
	//pointers for cuda memory locations
	int *matA_cuda;
	int *matB_cuda;
	int *ans_cuda;	

	//the moment at which we start measuring the time
	cudaEventCreate(&start);
	cudaEventRecord(start,0);
	
	//allocate memory in cuda
	cudaMalloc((void **)&matA_cuda,sizeof(int)*SIZE); checkCudaError();
	cudaMalloc((void **)&matB_cuda,sizeof(int)*SIZE); checkCudaError();	
	cudaMalloc((void **)&ans_cuda,sizeof(int)*SIZE); checkCudaError();
		
	//copy contents from ram to cuda
	cudaMemcpy(matA_cuda, matA, sizeof(int)*SIZE, cudaMemcpyHostToDevice); checkCudaError();
	cudaMemcpy(matB_cuda, matB, sizeof(int)*SIZE, cudaMemcpyHostToDevice); checkCudaError();	 

	//thread configuration 
	dim3 numBlocks(ceil(COLS/(float)16),ceil(ROWS/(float)16));
	dim3 threadsPerBlock(16,16);
	
	//do the matrix addition on CUDA
	addMatrix<<<numBlocks,threadsPerBlock>>>(ans_cuda,matA_cuda,matB_cuda);
	cudaDeviceSynchronize(); checkCudaError();

	//copy the answer back
	cudaMemcpy(ans, ans_cuda, sizeof(int)*SIZE, cudaMemcpyDeviceToHost);
	checkCudaError();

	//the moment at which we stop measuring time 
	cudaEventCreate(&stop);
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);

	//free the memory we allocated on CUDA
	cudaFree(matA_cuda);
	cudaFree(matB_cuda);
	cudaFree(ans_cuda);
	
/*************************CUDA STUFF ENDS HERE************************/

	for(i=0;i<ROWS;i++){
		for(j=0;j<COLS;j++){
			printf("%d ",ans[i][j]);
		}
		puts("");
	}
	
	return 0;

}