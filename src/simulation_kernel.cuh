#define RADIUS 5
#define ALIGN 16

extern __shared__ Circle sh_memory[];

__global__ void calculate(Circle** array, int imageW, int imageH, int size,
		float delta) {

	const int threads = blockDim.x * blockDim.y;
	//Circle myPosition;
	float newPosition[4];
	float r1;

	int gtid = 6 * (blockIdx.x * blockDim.x + threadIdx.x);
	printf("tralala %f\n",array[gtid]->x);

	Circle myCircle(*array[gtid]);

	/*
	 myPosition[0] = array[gtid];
	 myPosition[1] = array[gtid + 1];
	 myPosition[2] = array[gtid + 2];
	 myPosition[3] = array[gtid + 3];
	 myPosition[4] = array[gtid + 4];
	 myPosition[5] = array[gtid + 5];
	 */

	const int serialId = threadIdx.x + threadIdx.y * blockDim.x;

	printf("threads  \n");

	for (int i = serialId; i < size * 6; i += threads)
		sh_memory[i] = *array[i];

	__syncthreads();

	for (int i = 0; i < size ; i++) {
		int idx = i;
		r1 = ((float) sh_memory[idx].x - myCircle.x)
				* ((float) sh_memory[idx].x - myCircle.x)
				+ ((float) sh_memory[idx].y - myCircle.y)
						* ((float) sh_memory[idx].y - myCircle.y);
		if (r1
				< sh_memory[idx].radius * sh_memory[idx].radius
						+ myCircle.radius * myCircle.radius && r1 != 0) {

			Vector n(myCircle.x - sh_memory[idx].x, myCircle.y - sh_memory[idx].y);

			float length = sqrt(n.x * n.x + n.y + n.y);

			float nxnormalized = n.x/length;
			float nynormalized  = n.y / length;

			// DotProduct = (x1*x2 + y1*y2 + z1*z2)
			float dot1 = myCircle.velocity.x * n.x + myCircle.velocity.y * n.y;
			float dot2 = sh_memory[idx].velocity.x * n.x
					+ sh_memory[idx].velocity.y * n.y;

			float optimizedP = (2.0 * (dot1 - dot2))
					/ (myCircle.mass + sh_memory[idx].mass);

			float v1x = myCircle.velocity.x
					- optimizedP * sh_memory[idx].mass * n.x;
			float v1y = myCircle.velocity.y
					- optimizedP * sh_memory[idx].mass * n.y;

			//x
			myCircle.velocity.x = (v1x * (myCircle.mass - sh_memory[idx].mass)
					+ (2 * sh_memory[idx].mass * sh_memory[idx].velocity.x))
					/ (myCircle.mass + sh_memory[idx].mass);
			//y
			myCircle.velocity.y = (v1y * (myCircle.mass - sh_memory[idx].mass)
					+ (2 * sh_memory[idx].mass * sh_memory[idx].velocity.y))
					/ (myCircle.mass + sh_memory[idx ].mass);

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
	/*
	 float ax = 0;	//xGrav/myPosition[5];
	 float ay = 0;	//yGrav/myPosition[5];
	 //v = a*t
	 float vx = ax * delta;
	 float vy = ay * delta;

	 myPosition[2] = myPosition[2] + vx;
	 myPosition[3] = myPosition[3] + vy;
	 */
	//
	newPosition[0] = myCircle.x + myCircle.velocity.x * delta;
	newPosition[1] = myCircle.y + myCircle.velocity.y * delta;

	if (newPosition[0] > imageW) {
		myCircle.velocity.x = -1.0f * myCircle.velocity.x;
		myCircle.x = imageW - 10;
	} else if (newPosition[0] < 0) {
		myCircle.velocity.x = -1.0f * myCircle.velocity.x;
		myCircle.x = 10;
	}

	if (newPosition[1] > imageH) {
		myCircle.velocity.y = -1.0f * myCircle.velocity.y;
		myCircle.y = imageH - 10;

	} else if (newPosition[1] < 0) {
		myCircle.velocity.y = -1.0f * myCircle.velocity.y;
		myCircle.velocity.y = 10;

	}
	// Save the result in global memory for the integration step.
	array[gtid]->x = newPosition[0];
	array[gtid]->y = newPosition[1];

}

extern "C" void cuda_calculate(Circle** array, int imageW, int imageH, int size,
		float deltaTime) {

	int blocks = size / BLOCKDIM_X * BLOCKDIM_Y;

	dim3 threads(BLOCKDIM_X, BLOCKDIM_Y);

	dim3 grid(blocks);

	unsigned int aligned = size;
	aligned += ALIGN - aligned % ALIGN;

	int sh_size = aligned * sizeof(Circle)	;

	printf("sh_size = %d\n",sh_size);


	calculate<<<grid, threads, sh_size>>>(array, imageW, imageH, size,
			deltaTime);

}

