
#include "wb.h"
#include <stdio.h>

#define MASK_WIDTH 3
#define BLOCK_WIDTH 32 // BLOCK_WIDTH = TILE_WIDTH + (MASK_WIDTH - 1)
#define TILE_WIDTH 30 // TILE_WIDTH = BLOCK_WIDTH - (MASK_WIDTH + 1)


void wbImage_save(const wbImage_t& image, const char* fName){
	std::ostringstream oss;
        oss << "P6\n" << "# Created for blurring output" << "\n" << image.width << " " << image.height << "\n" << image.colors << "\n";
	//oss << "P6\n" << "# Created by GIMP version 2.10.8 PNM plug-in" << "\n" << image.width << " " << image.height << "\n" << image.colors << "\n";

        std::string headerStr(oss.str());

	std::ofstream outFile(fName, std::ios::binary);
        outFile.write(headerStr.c_str(), headerStr.size());

        const int numElements = image.width * image.height * image.channels;

        unsigned char* rawData = new unsigned char[numElements];

        for (int i = 0; i < numElements; ++i)
        {
            rawData[i] = static_cast<unsigned char>(image.data[i] * wbInternal::kImageColorLimit + 0.5f);
        }

        outFile.write(reinterpret_cast<char*>(rawData), numElements);
        outFile.close();

        delete [] rawData;
}

__global__ void without_shared_memory(float *input, float *output,  int height, int width, int channels, float* mask)
{
		
int col = blockIdx.x * blockDim.x + threadIdx.x;  
int row = blockIdx.y * blockDim.y + threadIdx.y;  

for (int color = 0; color < channels; color++) 
{
	float ouput_sum = 0.0f;


	if ( (col < width) && (row < height))
	{

		int n_start_col = col - (MASK_WIDTH/2);
		int n_start_row = row - (MASK_WIDTH/2);
		 // printf("n_start_col = %d n_start_row = %d \n" ,n_start_col,n_start_row );


		for (int j = 0; j < MASK_WIDTH; j++) 
		{
			for (int k = 0; k < MASK_WIDTH; k++) 
			{

				int cur_row = n_start_row + j ;
				int cur_col = n_start_col + k;

				if (cur_row > -1  && cur_row < height  && cur_col > -1 && cur_col < width)
				{
					ouput_sum +=  input[(cur_row * width + cur_col)* channels + color] * mask[j*MASK_WIDTH + k] ;
				}
				else
				{
				ouput_sum += 0.0f;
				}
			}
		}
}

	output[(row * width + col)*channels + color] = ouput_sum;
	
	}
}

//__constant__  float mask[9] = {1.0/9,1.0/9,1.0/9,1.0/9,1.0/9,1.0/9,1.0/9,1.0/9,1.0/9}; 

//__global__ void with_shared_memory(float *input, float *output,  int height, int width, int channels /*,  float* mask*/)  // with constant memory for mask

__global__ void with_shared_memory(float *input, float *output,  int height, int width, int channels, float* mask)
{
		
	int tx = threadIdx.x;    
	int ty = threadIdx.y;    
	int col_out = blockIdx.x * TILE_WIDTH + tx;  
	int row_out = blockIdx.y * TILE_WIDTH + ty; 
	int col_in = col_out - MASK_WIDTH / 2; 
	int row_in = row_out - MASK_WIDTH / 2;  
	
	__shared__ float N_ds[BLOCK_WIDTH][BLOCK_WIDTH];  //BLOCK_WIDTH = TILE_WIDTH + MASK_WIDTH - 1

	for (int color = 0; color < channels; color++) 
	{
		float sum = 0.0f;

	
		if ((row_in >= 0) && (row_in < height) && (col_in >= 0) && (col_in < width))
		{
			N_ds[ty][tx] = input[(row_in * width + col_in)* channels + color];
		}
		else
		{
			N_ds[ty][tx] = 0.0f; 
		}
		__syncthreads();

	
		if (ty < TILE_WIDTH && tx < TILE_WIDTH) 
		{
			for (int i = 0; i < MASK_WIDTH; i++)
			{
				for (int j = 0; j < MASK_WIDTH; j++)
				{
					sum += mask[i*MASK_WIDTH + j] * N_ds[i + ty][j + tx];
				}
			
			}
			if (row_out < height && col_out < width)
			{
				output[(row_out * width + col_out)*channels + color] = sum;
		    }
			
		}
		__syncthreads();
	}
}


int main(int argc, char ** argv) {

	int nDevices;
	cudaGetDeviceCount(&nDevices);
	printf("cudaGetDeviceCount: %d\n", nDevices);
	printf("There are %d CUDA devices.\n", nDevices);

	for (int i = 0; i < nDevices; i++) 
	{
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, i);
		printf("Device Number: %d \n", i);
		printf("Device name: %s \n ", prop.name);
		printf("Block dimensions: %d x %d  x %d \n", prop.maxThreadsDim[0],prop.maxThreadsDim[1],  prop.maxThreadsDim[2]);
		printf("Maximum number of threads per block: %d\n", prop.maxThreadsPerBlock);		
		printf ("Grid dimensions:  %d x %d x %d \n", prop.maxGridSize[0],  prop.maxGridSize[1],  prop.maxGridSize[2]);
		
	}

	char * inputImageFile;
	char * outputImageFile;
	wbImage_t inputImage;
	wbImage_t outputImage;

	float * hostInputImageData;
	float * hostOutputImageData;

	inputImageFile = argv[1];
	outputImageFile = argv[2];
	printf("Loading %s\n", inputImageFile);
	inputImage = wbImport(inputImageFile);
	hostInputImageData = wbImage_getData(inputImage);

	int imageWidth = wbImage_getWidth(inputImage);
	int imageHeight = wbImage_getHeight(inputImage);
	int imageChannels = wbImage_getChannels(inputImage);
	
	hostInputImageData = wbImage_getData(inputImage);
	
  	printf("%d %d %d\n", imageWidth, imageHeight, imageChannels);
  	printf("%f %f %f\n", hostInputImageData[0], hostInputImageData[1], hostInputImageData[2]);

	hostOutputImageData = hostInputImageData;
  	outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);
	outputImage.data = hostOutputImageData;
   	wbImage_save(outputImage, outputImageFile);


	
	float *d_input_image_data, *d_output_image_data, *d_mask_data;

	size_t image_size = imageWidth * imageHeight * imageChannels * sizeof(float);
	size_t mask_size =  MASK_WIDTH * MASK_WIDTH * sizeof(float);
	
	cudaMalloc((void **)&d_input_image_data,image_size );
	cudaMalloc((void **)&d_output_image_data, image_size);
	cudaMalloc((void **)&d_mask_data,mask_size);
	
	 float *hostMaskData = (float*)malloc(mask_size);
	
	for (int i = 0; i < (MASK_WIDTH * MASK_WIDTH) ; i++)
	{
		hostMaskData[i] = 1.0 / (MASK_WIDTH * MASK_WIDTH);
	}	
	
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	
	
	cudaMemcpy(d_input_image_data, hostInputImageData, image_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_mask_data, hostMaskData, mask_size, cudaMemcpyHostToDevice);
		
	
	dim3 dimBlock(BLOCK_WIDTH, BLOCK_WIDTH, 1);
	//dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
	
	//dim3 dimGrid((imageWidth -1) / (BLOCK_WIDTH + 1), (imageHeight- 1 )/ (BLOCK_WIDTH + 1), 1);  // without_shared_memory
	 dim3 dimGrid((imageWidth -1)  / (TILE_WIDTH+1) , (imageHeight- 1) / (TILE_WIDTH+1), 1);  //with_shared_memory
	
	// Start time

	cudaEventRecord(start);
                                                      
	//without_shared_memory <<< dimGrid, dimBlock >>> ( d_input_image_data, d_output_image_data, imageHeight, imageWidth, imageChannels,  d_mask_data);
	 with_shared_memory <<< dimGrid, dimBlock >>> (d_input_image_data, d_output_image_data, imageHeight, imageWidth, imageChannels ,  d_mask_data); // without constant memory  for mask
	// with_shared_memory <<< dimGrid, dimBlock >>> (d_input_image_data, d_output_image_data,  imageHeight, imageWidth, imageChannels /* ,d_mask_data*/); //with constant memory for mask

	cudaEventRecord(stop);

	
	cudaMemcpy(hostOutputImageData, d_output_image_data, image_size, cudaMemcpyDeviceToHost);
	
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);

	printf("  run time is  %f  milliseconds \n " , milliseconds);

	// Save image 
	wbImage_save(outputImage, outputImageFile);
	

	free(hostMaskData);
	wbImage_delete(outputImage);
	wbImage_delete(inputImage);

	cudaFree(d_input_image_data);
	cudaFree(d_output_image_data);
	cudaFree(d_mask_data);

	
	return 0;
}
