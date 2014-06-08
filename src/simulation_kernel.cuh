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
		r1 = ((float) sh_memory[idx] - myPosition[0])
				* ((float) sh_memory[idx] - myPosition[0])
				+ ((float) sh_memory[idx + 1] - myPosition[1])
						* ((float) sh_memory[idx + 1] - myPosition[1]);
		if (r1
				< sh_memory[idx + 4] * sh_memory[idx + 4]
						+ myPosition[4] * myPosition[4] && r1 != 0) {

			//collisionVector[0] = sh_memory[idx] - myPosition[0];
			//collisionVector[1] = sh_memory[idx + 1] - myPosition[1];
			/*
			 // First, find the normalized vector n from the center of
			 // circle1 to the center of circle2
			 Vector n = circle1.center - circle2.center;
			 n.normalize();

			 // Find the length of the component of each of the movement
			 // vectors along n.
			 // a1 = v1 . n
			 // a2 = v2 . n
			 float a1 = v1.dot(n);
			 float a2 = v2.dot(n);

			 // Using the optimized version,
			 // optimizedP =  2(a1 - a2)
			 //              -----------
			 //                m1 + m2
			 float optimizedP = (2.0 * (a1 - a2)) / (circle1.mass + circle2.mass);

			 // Calculate v1', the new movement vector of circle1
			 // v1' = v1 - optimizedP * m2 * n
			 Vector v1' = v1 - optimizedP * circle2.mass * n;

			 // Calculate v1', the new movement vector of circle1
			 // v2' = v2 + optimizedP * m1 * n
			 Vector v2' = v2 + optimizedP * circle1.mass * n;

			 circle1.setMovementVector(v1');
			 circle2.setMovementVector(v2');
			 */
			float vectorX,vectorY;
			vectorX = myPosition[0] - sh_memory[idx];
			vectorY = myPosition[1] - sh_memory[idx+1];
			float vector = sqrt(vectorX*vectorX + vectorY+vectorY);

			vectorX = vectorX/vector;
			vectorY = vectorY/vector;

			// DotProduct = (x1*x2 + y1*y2 + z1*z2)
			float dot1 = myPosition[2]*vectorX+myPosition[3]*vectorY;
			float dot2 = sh_memory[idx+2]*vectorX +sh_memory[idx+3]*vectorY;

			float optimizedP = (2.0 * (dot1 - dot2)) / ( myPosition[5]+ sh_memory[idx+5]);

			float v1x = myPosition[2] - optimizedP * sh_memory[idx+5] * vectorX;
			float v1y = myPosition[3] - optimizedP * sh_memory[idx+5] * vectorX;


			//x
			myPosition[2] = (v1x
					* (myPosition[5] - sh_memory[idx + 5])
					+ (2 * sh_memory[idx + 5] * sh_memory[idx + 2]))
					/ (myPosition[5] + sh_memory[idx + 5]);
			//y
			myPosition[3] = (v1y
					* (myPosition[5] - sh_memory[idx + 5])
					+ (2 * sh_memory[idx + 5] * sh_memory[idx + 3]))
					/ (myPosition[5] + sh_memory[idx + 5]);

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

	newPosition[0] = myPosition[0] + myPosition[2] * delta;
	newPosition[1] = myPosition[1] + myPosition[3] * delta;

	if (newPosition[0] > imageW) {
		myPosition[2] = -1.0f * myPosition[2];
		myPosition[0] = imageW - 10;
	} else if (newPosition[0] < 0) {
		myPosition[2] = -1.0f * myPosition[2];
		myPosition[0] = 10;
	}

	if (newPosition[1] > imageH) {
		myPosition[3] = -1.0f * myPosition[3];
		myPosition[1] = imageH - 10;

	} else if (newPosition[1] < 0) {
		myPosition[3] = -1.0f * myPosition[3];
		myPosition[1] = 10;

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

