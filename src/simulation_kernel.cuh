#define RADIUS 5
#define ALIGN 4

extern __shared__ float sh_memory[];

__global__ void calculate(float* array, int imageW, int imageH, int size,
		float delta) {
	const int threads = blockDim.x * blockDim.y;
	float myPosition[6];
	float newPosition[4];
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
		if (r1 < sh_memory[idx + 4] + myPosition[4] && r1 != 0) {

			//collisionVector[0] = sh_memory[idx] - myPosition[0];
			//collisionVector[1] = sh_memory[idx + 1] - myPosition[1];

			myPosition[2] = (myPosition[2]
					* (myPosition[5] - sh_memory[idx + 5])
					+ (2 * sh_memory[idx + 5] * sh_memory[idx + 2]))
					/ (myPosition[5] + sh_memory[idx + 5]);
			myPosition[0] = myPosition[0] + myPosition[2];

			myPosition[3] = (myPosition[3]
					* (myPosition[5] - sh_memory[idx + 5])
					+ (2 * sh_memory[idx + 5] * sh_memory[idx + 3]))
					/ (myPosition[5] + sh_memory[idx + 5]);
			myPosition[1] = myPosition[1] + myPosition[3];
		}
		__syncthreads();
	}

	//calculate ballz connections

	//
	//Gravity
	//float xGrav = 0;
	//float yGrav = -50;
	//acceleration
	//a =F/m
	float ax = 0;	//xGrav/myPosition[5];
	float ay = 0;	//yGrav/myPosition[5];
	//v = a*t
	float vx = ax * delta;
	float vy = ay * delta;

	myPosition[2] = myPosition[2] + vx;
	myPosition[3] = myPosition[3] + vy;

	//
	newPosition[0] = myPosition[0] + myPosition[2] * delta;
	newPosition[1] = myPosition[1] + myPosition[3] * delta;

	if (newPosition[0] > imageW) {
		myPosition[2] = -1.0f * myPosition[2];
		myPosition[0] = imageW - 1;
	} else if (newPosition[0] < 0) {
		myPosition[2] = -1.0f * myPosition[2];
		myPosition[0] = 1;
	}

	if (newPosition[1] > imageH) {
		myPosition[3] = -1.0f * myPosition[3];
		myPosition[1] = imageH - 1;

	} else if (newPosition[1] < 0) {
		myPosition[3] = -1.0f * myPosition[3];
		myPosition[1] = 1;

	}
	// Save the result in global memory for the integration step.
	array[gtid] = newPosition[0];
	array[gtid + 1] = newPosition[1];
	array[gtid + 2] = myPosition[2];
	array[gtid + 3] = myPosition[3];
	array[gtid + 4] = myPosition[4];
	array[gtid + 5] = myPosition[5];

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

