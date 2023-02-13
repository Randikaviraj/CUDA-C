/* C Program to to merge sort of a list in ascending order 
Note that this only supports lists that are powers of 2 
*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>


//size of the list
#define SIZE 1048576*16

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


void mergesort(float *list);

//check whether a certain number is a power of 2
int isPowerOfTwo(int num){
	int i=0;
	int val=1;
	for(i=0;val<=num;i++){
		if((val=pow(2,i))==num){
			return 1;
		}
		
	}				
	return 0;	

}

int main(){
	
	//check the condition that check that checks whether the size is a power of 2
	if(!isPowerOfTwo(SIZE)){
		fprintf(stderr,"This implementation needs the list size to be a power of two\n");
		exit(1);
	}
	
	//allocate a list
	float *list = (float *)malloc(sizeof(float)*SIZE);
	if(list==NULL){
		perror("Mem full");
		exit(1);
	}
	
	srand(time(NULL));
	int i;
	//generate some random values
	for(i=0;i<SIZE;i++){
		list[i]=rand()/(float)100000;
	}
	
	//print the input list
	printf("The input list is : \n");
	for(i=0;i<SIZE;i++){
		printf("%.2f ",list[i]);
	}
	printf("\n\n");
	
	//start measuring time
	double start = clock();
	
	//do sorting
	mergesort(list);
	
	//stop measuring time
	double stop = clock();
	
	//print the answer
	printf("The sorted list is : \n");
	for(i=0;i<SIZE;i++){
		printf("%.2f\n ",list[i]);
	}
	printf("\n\n");	
	
	//print the elapsed time
	double elapsedtime = (stop-start)/(float)CLOCKS_PER_SEC;
	fprintf(stderr, "The elapsed time for soring is %f seconds\n",elapsedtime);
	free(list);
	return 0;
}


/* merge two lists while sorting them in ascending order
* For example say there are two arrays 
* while one being 1 3 5 and the other being 2 4 7
* when merge they become 1 2 3 4 5 7
* When storing the two lists they are stored in same array and the
* two arrays are specified using the index of leftmost element, middle element and the last element
* For example say the two arrays are there in memory as a single array 1 3 5 2 4 7
* Here l=0 m=3 and r=5 specifies the two arrays separately
* */

__global__ void cuda_merge(float *list,float *temp,int step){
	
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
	int left = tid * step * 2;
    int middle = left + step;
    int right =(tid + 1) * step * 2 - 1;
    if(left <SIZE && right <= SIZE){
		
        //calculate the total number of elements
        int n=right-left+1;
        
       	
        
        //i is used for indexing elements in left array and j is used for indexing elements in the right array
        int i=left;
        int j=middle;
        
        //k is the index for the temporary array
        int k=i;
        
        /*now merge the two lists in ascending order
        check the first element remaining in each list and select the lowest one from them. Then put it to temp
        put increase the relevant index i or j*/
        
        while(i<middle && j<=right){
            if(list[i]<list[j]){
                temp[k]=list[i];
                i++;
            }
            else{
                temp[k]=list[j];
                j++;
            }
            k++;
        }

        //if there are remaining ones in an array append those to the end
        while (i<middle){
            temp[k]=list[i];
            i++;
            k++;
        }
        while (j<=right){
            temp[k]=list[j];
            j++;
            k++;
        }
                
        //now copy back the sorted array in temp to the original
        for(i=left,k=left;i<=right;i++,k++){
            list[i]=temp[k];	
        }
        
	}


}

void cpu_merge(float *list, int left,int middle,int right){
	
	//calculate the total number of elements
	int n=right-left+1;
	
	//create a new temporary array to do merge
	float *temp=(float *)malloc(sizeof(float)*n);
	
	//i is used for indexing elements in left array and j is used for indexing elements in the right array
	int i=left;
	int j=middle;
	
	//k is the index for the temporary array
	int k=0;
	
	/*now merge the two lists in ascending order
	check the first element remaining in each list and select the lowest one from them. Then put it to temp
	put increase the relevant index i or j*/
	
	while(i<middle && j<=right){
		if(list[i]<list[j]){
			temp[k]=list[i];
			i++;
		}
		else{
			temp[k]=list[j];
			j++;
		}
		k++;
	}

	//if there are remaining ones in an array append those to the end
	while (i<middle){
		temp[k]=list[i];
		i++;
		k++;
	}
	while (j<=right){
		temp[k]=list[j];
		j++;
		k++;
	}
			
	//now copy back the sorted array in temp to the original
	for(i=left,k=0;i<=right;i++,k++){
		list[i]=temp[k];	
	}
	
	//free the temporary array
	free(temp);
}

/* carry out merge sort ascending*/
void mergesort(float *list){
	float *list_cuda;
	float *temp_cuda;

    //allocate memory in cuda device
	cudaMalloc((void **)&list_cuda,sizeof(float)*SIZE); checkCudaError();
	cudaMalloc((void **)&temp_cuda,sizeof(float)*SIZE); checkCudaError();

    //copy contents from main memory to cuda device memory
	cudaMemcpy(list_cuda,list,sizeof(float)*SIZE,cudaMemcpyHostToDevice); checkCudaError();

	int left,middle,right;
	//step means the distance to the next list
	//loop till the merging happens for a list of the size of the original list
	int step=1;
	while(step<SIZE-1 && step < 32768){
		int numBlocks = (int)((SIZE/(2*step))/256) + 1;
		int threadsPerBlock = 256;
		printf("numBlock %d step %d \n",numBlocks,step);
		//do for all lists in the main list
		//call the cuda kernel
		cuda_merge<<<numBlocks,threadsPerBlock>>>(list_cuda,temp_cuda,step); 
		cudaDeviceSynchronize();
		checkCudaError();	

		//next list size
		step=step*2;		
	}
	cudaMemcpy(list,list_cuda,sizeof(float)*SIZE,cudaMemcpyDeviceToHost); checkCudaError();

	while(step<SIZE-1){
		
		//do for all lists in the main list
		int i=0;
		while(i+2*step<=SIZE){
			
			//calculate the index of the first element of the first list		
			left=i;
			
			//calculate the index of the first element of  the second list
			middle=i+step;		
			
			//calculate the last element of the second list			
			right=i+2*step-1;

			//merge them	
			cpu_merge(list,left,middle,right);
			
			//next list pair
			i=right+1;
		}
		
		//next list size
		step=step*2;		
	}
	

	cudaFree(list_cuda); checkCudaError();
	cudaFree(temp_cuda); checkCudaError();
	
}
