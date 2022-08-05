#define LOCAL_SIZE 16

__constant float filter[9] = { 1, 2, 3, 4, 5, 6, 7, 8, 9 };

__kernel void convolution(
    __global float* input,
    __global float* output,
    int ary_size
){
    unsigned int row = get_local_id(0);
    unsigned int col = get_local_id(1);
    unsigned int g_row = LOCAL_SIZE * get_group_id(0) + row;
    unsigned int g_col = LOCAL_SIZE * get_group_id(1) + col;

    if(g_row >= ary_size || g_col >= ary_size) return;

    float sum = 0;
    for(int i = 0; i < 3; i++){
        for(int j = 0; j < 3; j++){
            int _row = g_row + i - 1;
            int _col = g_col + j - 1;

            if(_row < 0 | _col < 0 | _row >= ary_size | _col >= ary_size) continue;

            sum += input[_row * ary_size + _col] * filter[i * 3 + j];
        }
    }
    
    output[g_row * ary_size + g_col] = sum;

    return;
}