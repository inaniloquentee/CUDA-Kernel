#include <cuda_runtime.h>

// Q, K, V, output are device pointers

extern __shared__ float shared_mem[];


__inline__ __device__ block_reduce(float dot, float* smem_reduce) {

}
__global__ void mha_kernel(const float* Q, const float* K, const float* V, float* output, int N, int d_model, int h, int d_k, float* scale) {

    int tid = threadIdx.x
    int head_idx = blockIdx.x;
    int token_id = blockIdx.y;
    int head_offset = head_idx * d_k + tid;

    int q_offset = token_id * d_model + head_offset;
    float q_val = Q[q_offset];
    float* s_scores = shared_mem;
    float* smem_reduce = &shared_mem[N]; 

    // 遍历k中的token
    for (int t = 0; t < N; t ++) {
        int kv_offset = t * d_model + head_offset;
        float k_val = tid < d_k? K[kv_offset]: 0.0f;
        float dot = q_val * k_val;
        float score = block_reduce(dot, smem_reduce) * scale;

        if (tid == 0) {
            shared_mem[tid] = score;
        }
    }
    __syncthreads();

    if (tid == 0) {
        float m = -FLT_MAX;
        float e_sum = 0.0f;
        for (int t = 0; t < N; t ++) {
            m = fmax(m, score[t]);
        }

        float e_sum = 0.0f;
        for (int t = 0; t < N; t ++) {
            score[t] = expf(score[t] - m);
            e_sum += score[t];
        }
        smem_reduce[32] = e_sum;
    }

    __syncthreads();

    
}
extern "C" void solve(const float* Q, const float* K, const float* V, float* output, int N, int d_model, int h) {

    int d_k = d_model / h;
    int threads_per_block = ((d_k + 32 - 1) / 32) * 32;
    int shared_mem_size = (N + 33) * sizeof(float);

    float* scale = 1.0f / sqrtf((float)d_k);

    dim3 grid(N, h)
    dim3 block(threads_per_block);

    mha_kernel<<<grid, block, shared_mem_size>>>(Q, K, V, output, N, d_model, h, d_k, scale);
}
