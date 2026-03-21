#include <cuda_runtime.h>

#define BLOCK_SIZE  256
#define WARP_SIZE   32

__global__ void reduce_kernel(const float* input, float* output, int N) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    float sum = 0.0f;
    if (i < N) sum = input[i];


    const int WARP_NUM = BLOCK_SIZE / WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    __shared__ float sum_s[WARP_NUM];


    // 对于一个Warp而言
    #pragma unroll
    for (int s = WARP_SIZE >> 1; s > 0; s >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, s);
    }


    if (lane_id == 0) {
        sum_s[warp_id] = sum;
    }

    __syncthreads(); // 不同Warp之间数据同步

    if (threadIdx.x < WARP_SIZE) {
        if (threadIdx.x < WARP_NUM) {
            // 有WARP_NUM个归约后的结果sum，放到一个Warp即可，对于有效线程:
            sum = sum_s[threadIdx.x];
        }else {
            // 对于无效线程：
            sum = 0.0f;
        }

        #pragma unroll
        for (int s = WARP_SIZE >> 1; s > 0; s >>= 1) {
            sum += __shfl_down_sync(0xffffffff, sum, s);
        }

        if (threadIdx.x == 0) {
            atomicAdd(output, sum);
        }
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    cudaMemset(output, 0, sizeof(float));
    int threadsPerBlock = BLOCK_SIZE;
    int blocksPerGrid = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    reduce_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
}
