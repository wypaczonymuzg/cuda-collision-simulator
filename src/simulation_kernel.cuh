#define RADIUS 5
#define ALIGN 4

extern __shared__ float sh_memory[];
__global__ void draw(TColor *dst, float* array, int imageW, int imageH,
		int size) {

	const int ix = blockDim.x * blockIdx.x + threadIdx.x;
	const int iy = blockDim.y * blockIdx.y + threadIdx.y;

	int temp;
	float E, r1;
	int x = 0;

	const int threads = blockDim.x * blockDim.y;

	const int serialId = threadIdx.x + threadIdx.y * blockDim.x;

	for (int i = serialId; i < size * 6; i += threads)
		sh_memory[i] = array[i];

	__syncthreads();

	for (int i = 0; i < size * 6; i += 6) {
		int idx = i;
		r1 = sqrt(
				((float) sh_memory[idx] - ix) * ((float) sh_memory[idx] - ix)
						+ ((float) sh_memory[idx + 1] - iy)
								* ((float) sh_memory[idx + 1] - iy));
		if (r1 < sh_memory[idx + 4]) {
			x++;
			break;
		}
	}

	if (x != 0) {
		E = 1;
	} else {
		E = 0;
	}
	if (ix < imageW && iy < imageH) {
		dst[imageW * iy + ix] = make_color(E, E, E, 0);
	}
}
/*
 __device__ float* bodyBodyInteraction(float* bi, float* bj, float delta) {
 float newPosition[6];
 // r_ij  [3 FLOPS]
 r.x = bj.x - bi.x;
 r.y = bj.y - bi.y;
 r.z = bj.z - bi.z;
 // distSqr = dot(r_ij, r_ij) + EPS^2  [6 FLOPS]
 float distSqr = r.x * r.x + r.y * r.y + r.z * r.z + EPS2;
 // invDistCube =1/distSqr^(3/2)  [4 FLOPS (2 mul, 1 sqrt, 1 inv)]
 float distSixth = distSqr * distSqr * distSqr;
 float invDistCube = 1.0f / sqrtf(distSixth);
 // s = m_j * invDistCube [1 FLOP]
 float s = bj.w * invDistCube;
 // a_i =  a_i + s * r_ij [6 FLOPS]
 ai.x += r.x * s;
 ai.y += r.y * s;
 ai.z += r.z * s;
 return newPosition;
 }


 __device__ float* tile_calculation(float* myPosition, float delta) {
 int i;
 float newPosition[2];
 for (i = 0; i < blockDim.x; i++) {
 newPosition = calculateNewPosition(myPosition, sh_memory[i], accel);
 }
 return newPosition;
 }
 */
__global__ void calculate(float* array, int imageW, int imageH, int size,
		float delta) {
	const int ix = blockDim.x * blockIdx.x + threadIdx.x;
	const int iy = blockDim.y * blockIdx.y + threadIdx.y;
	const int threads = blockDim.x * blockDim.y;

	int i, tile;
	float myPosition[6];
	float newPosition[4];
	float collisionVector[2];
	float r1;
	int gtid = 6 * (blockIdx.x * blockDim.x + threadIdx.x);
	myPosition[0] = array[gtid];
	myPosition[1] = array[gtid + 1];
	myPosition[2] = array[gtid + 2];
	myPosition[3] = array[gtid + 3];
	myPosition[4] = array[gtid + 4];
	myPosition[5] = array[gtid + 5];

	const int serialId = threadIdx.x + threadIdx.y * blockDim.x;

	for (int i = serialId; i < size * 6; i += threads)
		sh_memory[i] = array[i];

	__syncthreads();

	for (int i = 0; i < size * 6; i += 6) {
		int idx = i;
		r1 = sqrt(
				((float) sh_memory[idx] - myPosition[0])
						* ((float) sh_memory[idx] - myPosition[0])
						+ ((float) sh_memory[idx + 1] - myPosition[1])
								* ((float) sh_memory[idx + 1] - myPosition[1]));
		if (r1 < sh_memory[idx + 5] + myPosition[5] && r1 != 0) {

			//collisionVector[0] = sh_memory[idx] - myPosition[0];
			//collisionVector[1] = sh_memory[idx + 1] - myPosition[1];

			myPosition[2] = (myPosition[2]
					* (myPosition[5] - sh_memory[idx + 5])
					+ (2 * sh_memory[idx + 5] * sh_memory[idx + 2]))
					/ (myPosition[5] + sh_memory[idx + 5]);

			myPosition[3] = (myPosition[3]
					* (myPosition[5] - sh_memory[idx + 5])
					+ (2 * sh_memory[idx + 5] * sh_memory[idx + 3]))
					/ (myPosition[5] + sh_memory[idx + 5]);


		}
	}

	//calculate balls connections



	//

	newPosition[0] = myPosition[0] + myPosition[2] * delta;
	newPosition[1] = myPosition[1] + myPosition[3] * delta;
	if (newPosition[0] > imageW) {
		myPosition[2] = myPosition[2] * (-1.f);
		myPosition[0] = imageW - 1;
	} else if (newPosition[0] < 0) {
		myPosition[2] = myPosition[2] * (-1.f);
		myPosition[0] = 1;
	}

	if (newPosition[1] > imageH) {
		myPosition[3] = myPosition[3] * (-1.f);
		myPosition[1] = imageH-1;

	} else if (newPosition[1] < 0) {
		myPosition[3] = myPosition[3] * (-1.f);
		myPosition[1] =1;

	}
	// Save the result in global memory for the integration step.
	array[gtid] = newPosition[0];
	array[gtid + 1] = newPosition[1];
	array[gtid + 2] = myPosition[2];
	array[gtid + 3] = myPosition[3];
	array[gtid + 4] = myPosition[4];
	array[gtid + 5] = myPosition[5];

}

extern "C" void cuda_draw(TColor *d_dst, float* array, int imageW, int imageH,
		int size) {
	dim3 threads(BLOCKDIM_X, BLOCKDIM_Y);

	dim3 grid(iDivUp(imageW, BLOCKDIM_X), iDivUp(imageH, BLOCKDIM_Y));

	unsigned int aligned = size;
	aligned += ALIGN - aligned % ALIGN;

	int sh_size = aligned * sizeof(float) * 6;
	//printf("sh_size : %d\n",sh_size);
	draw<<<grid, threads, sh_size>>>(d_dst, array, imageW, imageH, size);
}

extern "C" void cuda_calculate(float* array, int imageW, int imageH, int size,
		float deltaTime) {

	int blocks = size / BLOCKDIM_X * BLOCKDIM_Y;

	dim3 threads(BLOCKDIM_X, BLOCKDIM_Y);

	dim3 grid(blocks);

	unsigned int aligned = size;
	aligned += ALIGN - aligned % ALIGN;

	int sh_size = aligned * sizeof(float) * 6;

	calculate<<<grid, threads, sh_size>>>(array, imageW, imageH, size,
			deltaTime);

}

