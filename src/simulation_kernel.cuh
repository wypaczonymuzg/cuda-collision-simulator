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
	/*
	 // Early Escape test: if the length of the movevec is less
	 // than distance between the centers of these circles minus
	 // their radii, there's no way they can hit.
	 double dist = B.center.distance(A.center);
	 double sumRadii = (B.radius + A.radius);
	 dist -= sumRadii;
	 if(movevec.Magnitude() < dist){
	 return false;
	 }

	 // Normalize the movevec
	 Vector N = movevec.copy();
	 N.normalize();

	 // Find C, the vector from the center of the moving
	 // circle A to the center of B
	 Vector C = B.center.minus(A.center);

	 // D = N . C = ||C|| * cos(angle between N and C)
	 double D = N.dot(C);

	 // Another early escape: Make sure that A is moving
	 // towards B! If the dot product between the movevec and
	 // B.center - A.center is less that or equal to 0,
	 // A isn't isn't moving towards B
	 if(D <= 0){
	 return false;
	 }

	 // Find the length of the vector C
	 double lengthC = C.Magnitude();

	 double F = (lengthC * lengthC) - (D * D);

	 // Escape test: if the closest that A will get to B
	 // is more than the sum of their radii, there's no
	 // way they are going collide
	 double sumRadiiSquared = sumRadii * sumRadii;
	 if(F >= sumRadiiSquared){
	 return false;
	 }

	 // We now have F and sumRadii, two sides of a right triangle.
	 // Use these to find the third side, sqrt(T)
	 double T = sumRadiiSquared - F;

	 // If there is no such right triangle with sides length of
	 // sumRadii and sqrt(f), T will probably be less than 0.
	 // Better to check now than perform a square root of a
	 // negative number.
	 if(T < 0){
	 return false;
	 }

	 // Therefore the distance the circle has to travel along
	 // movevec is D - sqrt(T)
	 double distance = D - sqrt(T);

	 // Get the magnitude of the movement vector
	 double mag = movevec.Magnitude();

	 // Finally, make sure that the distance A has to move
	 // to touch B is not greater than the magnitude of the
	 // movement vector.
	 if(mag < distance){
	 return false;
	 }

	 // Set the length of the movevec so that the circles will just
	 // touch
	 movevec.normalize();
	 movevec.times(distance);

	 return true;
	 */
	float movex =myPosition[2] * delta;
	float movey = myPosition[3] * delta;

	for (int i = 0; i < size * 6; i += 6) {
		int idx = i;
		float dist = sqrt(
				(myPosition[0] - sh_memory[idx])
						* (myPosition[0] - sh_memory[idx])
						+ ((myPosition[1] - sh_memory[idx + 1])
								* (myPosition[0] - sh_memory[idx])));
		float sumRad = myPosition[5] + sh_memory[idx + 5];

		dist -= sumRad;
		float mvx = myPosition[2] * delta;
		float mvy = myPosition[3] * delta;
		float mag = sqrt(mvx * mvx + mvy * mvy);

		if (mag < dist)
			continue;

		float vx = myPosition[2];
		float vy = myPosition[3];
		float vecLen = sqrt(vx * vx + vy * vy);
		float vxn = vx / vecLen;
		float vyn = vy / vecLen;

		float cx = myPosition[0] - sh_memory[idx];
		float cy = myPosition[1] - sh_memory[idx + 1];

		float d = sqrt(vxn * vxn + vyn * vyn) * sqrt(cx * cx + cy * cy)
				* ((cx - vxn) / sqrt(vxn * vxn + vyn * vyn)
						+ sqrt(cx * cx + cy * cy));
		if (d <= 0)
			continue;

		float lenC = sqrt(cx * cx + cy * cy);
		float F = lenC * lenC - d * d;

		float sumRdSq = sumRad * sumRad;
		if (F >= sumRdSq)
			continue;

		float T = sumRdSq - F;

		if (T < 0)
			continue;

		dist = d - sqrt(T);
		if (mag < dist)
			continue;

		float mvlen = sqrt(mvx * mvx + mvy * mvy);
		float nmvx = mvx / mvlen;
		float nmvy = mvy / mvlen;

		movex = 0;//movex = nmvx * dist;
		movey = 0;;//movey = nmvy * dist;

		myPosition[2] = (myPosition[2] * (myPosition[5] - sh_memory[idx + 5])
				+ (2 * sh_memory[idx + 5] * sh_memory[idx + 2]))
				/ (myPosition[5] + sh_memory[idx + 5]);

		myPosition[3] = (myPosition[3] * (myPosition[5] - sh_memory[idx + 5])
				+ (2 * sh_memory[idx + 5] * sh_memory[idx + 3]))
				/ (myPosition[5] + sh_memory[idx + 5]);

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
	newPosition[0] = myPosition[0] + movex;
	newPosition[1] = myPosition[1] + movey;

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

