# image_blurring_with_CUDA
   
Firstly, I write image blurring code with non-shared memory for all pictures.

 >>nvcc test.cu
 >> output.exe 1.ppm 1_out.ppm


For rgb I added for block  “ for (int color = 0; color < channels; color++)”.
Then I run image blurring code  with shared memory for all pictures.. 
When we use shared memory, we see a significant reduction in code execution time. This is because we use the same data over and over, and if we recall it in shared memory, it will be faster to retrieve data than global memory.

 Also I used __syncthreads()  twice in this code.The firstly __syncthreads()   is enures that threads have finished loading input data into N_ds[] before any of them can move foward. Second   __syncthreads()   is ensure that all threads finished N_ds []elements in shared memory before any of them, move to the next iteration and load the elements  in the next tiles. Otherwise  threads would load elemnts too early and corrupt the input data for other threads.

Lastly I used shared memoy and constant memory for all pictures.
 
The fastest running code was the code I used in the constant memory and the shared memory. Because constant memory is cached to constant cache and it can be accessed by each thread, so all threads needs to acces mask elements in the same order. 

Then , I added  __constant__ variable for mask. 
 


picture 	non-shared memory execution time(ms)	shared memory execution time(ms)	shared memory and constant memory execution
 time(ms)
1.ppm               	1.663328	                            0.907202	                                  0.78048
2.ppm	                6.501664	                            3.744288	                                  3.244064
3.ppm	                14.506112                           	8.383712	                                  7.408128
4.ppm	                25.38336                            	15.124896	                                  13.163232
5.ppm               	40.081791	                            23.758017	                                  20.58544
6.ppm               	56.329407                           	33.941921	                                  29.522272
7.ppm               	75.391396	                            45.061089	                                  40.249729
8.ppm               	98.765121                            	59.157726	                                  52.655006
9.ppm	                125.083809	                          74.433823	                                  67.103455
10.ppm              	153.996735	                          94.170845	                                  82.63591

 
 
The larger the image size, the more noticeable the time difference. Because more and more threads are being used. So the thread that uses global memory or shared memory or constant memory is growing.


