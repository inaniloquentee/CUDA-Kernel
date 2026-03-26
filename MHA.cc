#include <cuda_runtime.h>
#include <math.h>
#include <float.h>

__inline__ __device__ double warp_reduce_sum_double(double val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__inline__ __device__ double block_reduce_double(double val, double* smem) {
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;
    int num_warps = (blockDim.x + 31) / 32;

    val = warp_reduce_sum_double(val);
    if (lane == 0) smem[wid] = val;
    __syncthreads();

    double sum = (threadIdx.x < num_warps) ? smem[lane] : 0.0;
    if (wid == 0) sum = warp_reduce_sum_double(sum);

    if (threadIdx.x == 0) smem[0] = sum;
    __syncthreads();

    double ret = smem[0];
    __syncthreads();
    return ret;
}

extern __shared__ double shared_mem_d[];

__global__ void mha_kernel(const float* Q, const float* K, const float* V, float* output, int N, int d_model, int h, int d_k, double scale) {

    int tid = threadIdx.x;
    int token_id = blockIdx.x;
    int head_idx = blockIdx.y;

    int head_offset = head_idx * d_k;
    int q_offset = token_id * d_model + head_offset;

    double q_val = (tid < d_k) ? (double)Q[q_offset + tid] : 0.0;

    double* s_scores = shared_mem_d;
    double* smem_reduce = &shared_mem_d[N]; 

    for (int t = 0; t < N; t++) {
        int kv_offset = t * d_model + head_offset;
        double k_val = (tid < d_k) ? (double)K[kv_offset + tid] : 0.0;
        
        double dot = q_val * k_val;
        double score = block_reduce_double(dot, smem_reduce) * scale;

        if (tid == 0) {
            s_scores[t] = score;
        }
    }
    __syncthreads();

    if (tid == 0) {
        double m = -DBL_MAX;
        for (int t = 0; t < N; t++) {
            m = fmax(m, s_scores[t]);
        }

        double e_sum = 0.0;
        for (int t = 0; t < N; t++) {
            s_scores[t] = exp(s_scores[t] - m);
            e_sum += s_scores[t];
        }
        smem_reduce[32] = e_sum;
    }
    __syncthreads();

    if (tid < d_k) {
        double res = 0.0;
        double total_e_sum = smem_reduce[32]; 

        for (int t = 0; t < N; t++) {
            double prob = s_scores[t] / total_e_sum;
            double v_val = (double)V[t * d_model + head_offset + tid];
            res += prob * v_val;
        }

        output[q_offset + tid] = (float)res;
    }
}

extern "C" void solve(const float* Q, const float* K, const float* V, float* output, int N, int d_model, int h) {

    int d_k = d_model / h;
    int threads_per_block = ((d_k + 31) / 32) * 32;
    

    size_t shared_mem_size = (N + 33) * sizeof(double);


    double scale = 1.0 / sqrt((double)d_k);

    dim3 grid(N, h); 
    dim3 block(threads_per_block);

    mha_kernel<<<grid, block, shared_mem_size>>>(Q, K, V, output, N, d_model, h, d_k, scale);
    cudaDeviceSynchronize();
}
