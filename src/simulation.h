#ifndef SIMULATION_H
#define SIMULATION_H

typedef unsigned int TColor;

#define BLOCKDIM_X 8
#define BLOCKDIM_Y 8

#define DELTA 0.016

#ifndef MAX
#define MAX(a,b) ((a < b) ? b : a)
#endif
#ifndef MIN
#define MIN(a,b) ((a < b) ? a : b)
#endif

class Vector {
public:
	float x, y;
	__host__ __device__ Vector(float nx, float ny) {
		x = nx;
		y = ny;
	}
	__host__ __device__ Vector() {
		x = 0;
		y = 0;
	}

};

class Circle {
public:
	float x, y;
	Vector velocity;
	float radius;
	float mass;
	__host__ __device__ Circle(float nx, float ny, Vector nvector,
			float nradius, float nmass) {
		x = nx;
		y = ny;
		velocity = nvector;
		radius = nradius;
		mass = nmass;
	}
};

// CUDA wrapper functions for allocation/freeing texture arrays
extern "C" cudaError_t CUDA_Bind2TextureArray();
extern "C" cudaError_t CUDA_UnbindTexture();
extern "C" cudaError_t CUDA_MallocArray(uchar4 **h_Src, int imageW, int imageH);
extern "C" cudaError_t CUDA_FreeArray();

// CUDA kernel functions

extern "C" void cuda_calculate(float* array, int imageW, int imageH,
		int sh_size, float delta);
#endif
