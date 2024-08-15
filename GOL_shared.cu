%%writefile gol.cu
#include<cuda.h>
#include<iostream>
#include<stdio.h>
#include<vector>
#include<fstream>
#include<sstream>
#include <filesystem>


#define BLOCKSIZE 16
using namespace std;
namespace fs = std::filesystem;

__global__ void populate(int *gpu_cur,int *gpu_prev,int height,int width){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    __shared__ int shared_prev[BLOCKSIZE + 2][BLOCKSIZE + 2];
    int idx=row*width+col;
 
    int shared_row = threadIdx.y + 1;
    int shared_col = threadIdx.x + 1;

    if (row < height && col < width) {
        shared_prev[shared_row][shared_col] = gpu_prev[idx];
        if (threadIdx.y == 0 && row > 0)
            shared_prev[0][shared_col] = gpu_prev[(row - 1) * width + col];
        if (threadIdx.y == blockDim.y - 1 && row < height - 1)
            shared_prev[BLOCKSIZE + 1][shared_col] = gpu_prev[(row + 1) * width + col];
        if (threadIdx.x == 0 && col > 0)
            shared_prev[shared_row][0] = gpu_prev[row * width + col - 1];
        if (threadIdx.x == blockDim.x - 1 && col < width - 1)
            shared_prev[shared_row][BLOCKSIZE + 1] = gpu_prev[row * width + col + 1];
    }

    if (threadIdx.x == 0 && threadIdx.y == 0 && row > 0 && col > 0)
        shared_prev[0][0] = gpu_prev[(row - 1) * width + (col - 1)];
    if (threadIdx.x == blockDim.x - 1 && threadIdx.y == 0 && row > 0 && col < width - 1)
        shared_prev[0][BLOCKSIZE + 1] = gpu_prev[(row - 1) * width + (col + 1)];
    if (threadIdx.x == 0 && threadIdx.y == blockDim.y - 1 && row < height - 1 && col > 0)
        shared_prev[BLOCKSIZE + 1][0] = gpu_prev[(row + 1) * width + (col - 1)];
    if (threadIdx.x == blockDim.x - 1 && threadIdx.y == blockDim.y - 1 && row < height - 1 && col < width - 1)
        shared_prev[BLOCKSIZE + 1][BLOCKSIZE + 1] = gpu_prev[(row + 1) * width + (col + 1)];

    __syncthreads();  
    if(row<1 || row>height-2 || col<1 || col>width-2)
      return;
  
    int dx[8]={-1,-1,0,1,1,1,0,-1};
    int dy[8]={0,1,1,1,0,-1,-1,-1};
    int liveneighbours=0;
    for(int i=0;i<8;i++){
      if(shared_prev[shared_row+dx[i]][shared_col+dy[i]])
        liveneighbours++;
    }
    if(liveneighbours>3||liveneighbours<2)
      gpu_cur[idx]=0;
    else if(liveneighbours==3)
      gpu_cur[idx]=1;
}

int main(){
    int height=100,width=100;
    int timesteps;
    cout<<"Enter the timesteps:";
    cin>>timesteps;
    size_t s=sizeof(int)*height*width;
    int *scene=new int[height*width];

    for(int i=0;i<height;i++){
        for(int j=0;j<width;j++){
            scene[i*width+j]=0;
        }
      }
    
    scene[48*width+48]=1;
    scene[49*width+48]=1;
    scene[50*width+48]=1;
    scene[48*width+49]=1;
    scene[49*width+49]=1;
    scene[50*width+49]=1;
    scene[48*width+50]=1;
    scene[49*width+50]=1;
    scene[50*width+50]=1;

    
    
 /*   scene[48*width+48]=1;
    scene[49*width+48]=1;
    scene[49*width+47]=1;
    scene[50*width+48]=1;
    scene[50*width+49]=1;
    
*/
/*    scene[13*width+13]=1;
    scene[13*width+14]=1;
    scene[13*width+15]=1;
    scene[14*width+12]=1;
    scene[14*width+13]=1;
    scene[14*width+14]=1;
  */  

    /*scene[13*width+13]=1;
    scene[14*width+13]=1;
    scene[15*width+13]=1;
*/

    cout<<"printing the initial config\n";

    fs::create_directory("output");
    stringstream filename;
    filename<<"output/state_1.txt";
    string fn=filename.str();
    ofstream file(fn);
    if (!file.is_open()) {
        cerr << "Error opening file: " << fn << std::endl;
        return;
    }


    for(int i=0;i<height;i++){
        for(int j=0;j<width;j++){
            cout<<scene[i*width+j]<<" ";
            file << scene[i * width + j] << " ";
        }
        file<<"\n";
        cout<<endl;
      }


    int count=0;
    int *gpu_cur,*gpu_prev;
    cudaMalloc(&gpu_cur,s);
    cudaMalloc(&gpu_prev,s);
    cudaMemcpy(gpu_cur,scene,s,cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_prev,scene,s,cudaMemcpyHostToDevice);
    dim3 blocksize(BLOCKSIZE,BLOCKSIZE);
    dim3 gridsize((width+BLOCKSIZE-1)/BLOCKSIZE,(height+BLOCKSIZE-1)/BLOCKSIZE);
   // vector<int *> result;


    //calculating the time 
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    while(count<timesteps-1){
      //cout<<"inside loop "<<count<<endl;
      populate<<<gridsize,blocksize>>>(gpu_cur,gpu_prev,height,width);
      cudaDeviceSynchronize();

     // int *tempscene=new int[height*width];
     // cudaMemcpy(tempscene,gpu_cur,s,cudaMemcpyDeviceToHost);
     cudaMemcpy(scene,gpu_cur,s,cudaMemcpyDeviceToHost);
      cudaMemcpy(gpu_prev,gpu_cur,s,cudaMemcpyDeviceToDevice);
     // result.push_back(tempscene);

      stringstream filename;
      filename<<"output/state_"<<count+2<<".txt";
      string fn=filename.str();
      ofstream file(fn);
      if (!file.is_open()) {
          cerr << "Error opening file: " << fn << std::endl;
          return;
      }
      for(int i=0;i<height;i++){
        for(int j=0;j<width;j++){
            //cout<<scene[i*width+j]<<" ";
            file << scene[i * width + j] << " ";
        }
        //cout<<endl;
        file<<"\n";
      }
    count++;
    }

    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cout << "Time for shared memory:"<< milliseconds << " ms" << endl;

    cudaFree(gpu_cur);
    /*
    for(int i=0;i<result.size();i++){
        cout<<"timestep:"<<i<<"\n";
        for(int j=0;j<height;j++){
        for(int k=0;k<width;k++){
            cout<<result[i][j*width+k]<<" ";
            
        }
        cout<<endl;
      }
    }*/

    return 0;
}
