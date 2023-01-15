#include <stdio.h>
#include <stdlib.h>

#define SIZE 1025
#define THREADBLOCKSIZE 256

__global__ void addVector(int vectorA[SIZE],int vectorB[SIZE],int vectorC[SIZE]);

int main(int argc, char const *argv[])
{
    int vectorA[SIZE];
    int vectorB[SIZE];
    int vectorC[SIZE];

    for (int i = 0; i < SIZE; i++)
    {
        vectorA[i] = i;
        vectorB[i] = SIZE-i;
    }

    int *vectorA_cuda;
    int *vectorB_cuda;
    int *vectorC_cuda;
    cudaError_t code;
    cudaMalloc((void **)&vectorA_cuda,sizeof(int)*SIZE);
    code = cudaGetLastError();

    if (code != cudaSuccess)
    {
        printf("Error occured in %s at function %s line no %d \n",__FILE__,__FUNCTION__,__LINE__);
        exit(1);
    }
    cudaMalloc((void **)&vectorB_cuda,sizeof(int)*SIZE);
    code = cudaGetLastError();
    if (code != cudaSuccess)
    {
        printf("Error occured in %s at function %s line no %d \n",__FILE__,__FUNCTION__,__LINE__);
        exit(1);
    }
    cudaMalloc((void **)&vectorC_cuda,sizeof(int)*SIZE);
    code = cudaGetLastError();
    if (code != cudaSuccess)
    {
        printf("Error occured in %s at function %s line no %d \n",__FILE__,__FUNCTION__,__LINE__);
        exit(1);
    }
    
    cudaMemcpy(vectorA_cuda,vectorA,sizeof(int)*SIZE,cudaMemcpyHostToDevice);
    code = cudaGetLastError();
    if (code != cudaSuccess)
    {
        printf("Error occured in %s at function %s line no %d \n",__FILE__,__FUNCTION__,__LINE__);
        exit(1);
    }
    cudaMemcpy(vectorB_cuda,vectorB,sizeof(int)*SIZE,cudaMemcpyHostToDevice);
    code = cudaGetLastError();
    if (code != cudaSuccess)
    {
        printf("Error occured in %s at function %s line no %d \n",__FILE__,__FUNCTION__,__LINE__);
        exit(1);
    }

    int noOfThreadBlocks = (int)(SIZE/THREADBLOCKSIZE) + 1;
    addVector<<<noOfThreadBlocks,THREADBLOCKSIZE>>>(vectorA_cuda,vectorB_cuda,vectorC_cuda);

    cudaDeviceSynchronize();
    code = cudaGetLastError();
    if (code != cudaSuccess)
    {
        printf("Error occured in %s at function %s line no %d \n",__FILE__,__FUNCTION__,__LINE__);
        exit(1);
    }
    cudaMemcpy(vectorC,vectorC_cuda,sizeof(int)*SIZE,cudaMemcpyDeviceToHost);
    code = cudaGetLastError();
    if (code != cudaSuccess)
    {
        printf("Error occured in %s at function %s line no %d \n",__FILE__,__FUNCTION__,__LINE__);
        exit(1);
    }
    
    printf("Answer is : ");
    for (int i = 0; i < SIZE; i++)
    {
       printf("%d ",vectorC[i]);
    }
    
    return 0;
}

__global__ void addVector(int vectorA[SIZE],int vectorB[SIZE],int vectorC[SIZE]){
    int thread_Id =blockDim.x*blockIdx.x + threadIdx.x;
    if (thread_Id < SIZE)
        vectorC[thread_Id] = vectorA[thread_Id]+ vectorB[thread_Id];
}
