/* C program to generate a mandelbrot set image */

#include <math.h>
#include <stdio.h>
#include <time.h>

//The width of the generated image in pixels
#define WIDTH 1027
//The height of the generated image in pixels
#define HEIGHT 768
//The value on the real axis which maps to the left most pixel of the image
#define XMIN -2.0
//The value of the real axis which maps to the right most pixel of the image
#define XMAX 1
//The value in the imaginary axis which maps to the top pixels of the image
#define YMIN -1.25
//The value in the imaginary axis which maps to the bottom pixels of the image
#define YMAX 1.25
//The value that we consider as infinity.
#define INF 4
//The maximum number of times  that is tried to check whether infinity is reached.
#define MAXN 3000
//find maximum of two numbers
#define max(a,b) (((a)>(b))?(a):(b))
//File name for output image
#define FILENAME "image.ppm"

//The matrix to save the m andelbrot set
int mandel_set[HEIGHT][WIDTH]; 
//The RGB values for the mandelbrot image
unsigned char image[WIDTH *HEIGHT * 3];


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



/**************************************Mandelbrot calculation ***********************************************/
// CUDA kernel to docheck a given complex number is in Mandelbrot set. return 0 if is in mandelbrot set else return a value based on divergence to later assign a color
__global__ void isin_mandelbrot(int *blank_cuda, unsigned char *image_cuda){
	//derive the row and column based on thread configuration
	int row = blockIdx.y*blockDim.y + threadIdx.y;
	int col = blockIdx.x*blockDim.x + threadIdx.x;

    /************************************ Pixel  transformations ***********************************************/
    //Here x is the pixel number and this is linearly transformed to a value in the real axis to a value between XMIN and XMAX
    float realc = XMIN+col*(XMAX-XMIN)/(float)WIDTH;
    //Here y is the pixel number and this is linearly transformed to a value in the imaginary axis to a value between YMIN and YMAX
    float imagc = YMAX-row*(YMAX-YMIN)/(float)HEIGHT;

	//initialize values
	int i=0;
	float realz_next=0,imagz_next=0;
	float abs=0;
	float realz=0;
	float imagz=0;
	
	//do the calculations till the inifinity(a large number) is reached or the maximum number of iterations is reached
	while(i<MAXN && abs<INF){
		
		//mandelbrot equations
		realz_next=realz*realz-imagz*imagz+realc;
		imagz_next=2*realz*imagz+imagc;
		
		//absolute value
		abs=realz*realz+imagz*imagz;
		
		//next values
		realz=realz_next;
		imagz=imagz_next;
		i++;
	}
	
    //Limit calculations for valid indices
	if(row < HEIGHT && col < WIDTH){
		int index = row*WIDTH+col;
        //if the number of iterations had reached maximum that means hasnt reached infinity and we say the number of in not in mandelbrot set
	    if (i==MAXN)
		    blank_cuda[index]=0;
        //if the max number of iterations hasnt reached that means it has hit the infinity before that
        // then we say it is not in mandelbrot set and return the number of iterations, to later compute a color value
        else	
            blank_cuda[index]=i;
		
		/* Create the mandelbrot RGB image matrix mbased on the mandelbrot matrix. This is an array with RGB values*/
		//Generate the RGB matrix based on divergence value
		int color = blank_cuda[index];
		int n = index * 3;
		image_cuda[n]= i==0? 0 : ((color+10)%256); /*Calculate R value in RGB based on divergence.*/
		image_cuda[n+1]= i==0? 0 : ((color+100) % 9 * (255/9)); /*Calculate G value in RGB based on divergence*/
		image_cuda[n+2]= i==0? 0 : ((color + 234) % 7 * (255/7)) ;  /*Calculate B value in RGB based on divergence*/
	}
	

}


/* Create the mandelbrot matrix. If a pixel is in mandelbrot set value is 0. else the divergence */
void plot_and_createimage(int blank[HEIGHT][WIDTH],unsigned char image[WIDTH *HEIGHT * 3]){
    //pointers for arrays to be put on cuda memory
	int *blank_cuda;
	unsigned char *image_cuda;

    //allocate memory in cuda device
	cudaMalloc((void **)&blank_cuda,sizeof(int)*HEIGHT*WIDTH); checkCudaError();
	cudaMalloc((void **)&image_cuda,sizeof(unsigned char)*HEIGHT*WIDTH*3); checkCudaError();

    dim3 threadsPerBlock(16,16);
	dim3 numBlocks(ceil(WIDTH/(float)16),ceil(HEIGHT/(float)16));
    //calculate whether is in mandelbrot or not, for each pixel. If not, the divergence is entered as the value. If yes 0 is entered
	isin_mandelbrot<<<numBlocks,threadsPerBlock>>>(blank_cuda,image_cuda);
    cudaDeviceSynchronize();
	checkCudaError();

    //copy the answers back from cuda to ram
	cudaMemcpy(blank,blank_cuda,sizeof(int)*HEIGHT*WIDTH,cudaMemcpyDeviceToHost); checkCudaError();
	cudaMemcpy(image,image_cuda,sizeof(unsigned char)*HEIGHT*WIDTH*3,cudaMemcpyDeviceToHost); checkCudaError();
	
}



/***********************************main function*********************************************************/

int main(int argc, char** argv) {

	//start measuring time
	clock_t begin,end;
	begin=clock();

	//create the mandelbrot matrix and generate the mandelbrot RGB image matrix
	plot_and_createimage(mandel_set,image);
  
	//stop measuring time and print
	end=clock();
	double cputime=(double)((end-begin)/(float)CLOCKS_PER_SEC);
	printf("Time using GPU for calculation is %.10f\n",cputime);

	/* Write the image to file*/
  
	//meta data for the file
    const int MaxColorComponentValue=255; 
    char *comment="# ";//comment should start with # 
        
    //create new file,give it a name and open it in binary mode     
    FILE * fp=fopen(FILENAME,"wb"); // b -  binary mode 
    
    //write ASCII metadata header to the file
    fprintf(fp,"P6\n %s\n %d\n %d\n %d\n",comment,WIDTH,HEIGHT,MaxColorComponentValue);
    
    //compute and write image data bytes to the file
    fwrite(image,1,WIDTH *HEIGHT * 3,fp);
			
    //close the file 
    fclose(fp);
		
	return 0;
}