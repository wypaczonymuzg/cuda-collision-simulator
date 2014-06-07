// OpenGL Graphics includes
#include <GL/glew.h>
#include <GL/freeglut.h>

// CUDA utilities and system includes
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

// Includes
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "simulation.h"

// includes, project
#include <helper_functions.h> // includes for helper utility functions
#include <helper_cuda.h>      // includes for cuda error checking and initialization
#define CUDA_CHECK_RETURN(value) {											\
	cudaError_t _m_cudaStat = value;										\
	if (_m_cudaStat != cudaSuccess) {										\
		fprintf(stderr, "Error %s at line %d in file %s\n",					\
				cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);		\
		exit(1);															\
	} }
////////////////////////////////////////////////////////////////////////////////
// Global data handlers and parameters
////////////////////////////////////////////////////////////////////////////////
//OpenGL PBO and texture "names"
GLuint gl_PBO, gl_Tex;
struct cudaGraphicsResource *cuda_pbo_resource; // handles OpenGL-CUDA exchange
//Source image on the host side
uchar4 *h_Src;
int imageW = 1024, imageH = 768;
GLuint shader;

int sh_size;
int num_of_bodies;
float* h_array;
float* d_array;
float delta;
//int zal = 0;

////////////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////////////
int g_Kernel = 0;
bool g_FPS = false;
bool g_Diag = false;
StopWatchInterface *timer = NULL;

const int frameN = 24;
int frameCounter = 0;

#define BUFFER_DATA(i) ((char *)0 + i)

// Auto-Verification Code
const int frameCheckNumber = 4;
int fpsCount = 0;        // FPS count for averaging
int fpsLimit = 1;        // FPS limit for sampling
unsigned int frameCount = 0;
unsigned int g_TotalErrors = 0;

int *pArgc = NULL;
char **pArgv = NULL;

#define REFRESH_DELAY     10 //ms
void displayCUDAInfo() {
	const int kb = 1024;
	const int mb = kb * kb;
	printf("CUDA INFO:\n=========\n\nCUDA version:   v%d\n", CUDART_VERSION);

	int devCount;
	cudaGetDeviceCount(&devCount);
	printf("CUDA Devices: \n\n");

	for (int i = 0; i < devCount; ++i) {
		cudaDeviceProp props;
		cudaGetDeviceProperties(&props, i);
		printf("%d : %s:%d.%d\n", i, props.name, props.major, props.minor);
		printf("  Global memory:   %dmb\n", props.totalGlobalMem / mb);
		printf("  Shared memory:   %dkb\n", props.sharedMemPerBlock / kb);
		printf("  Constant memory: %dkb\n", props.totalConstMem / kb);
		printf("  Block registers: %d\n", props.regsPerBlock);

		printf("  Warp size:         %d\n", props.warpSize);
		printf("  Threads per block: %d\n", props.maxThreadsPerBlock);
		printf("  Max block dimensions: [ %d, %d, %d ]\n",
				props.maxThreadsDim[0], props.maxThreadsDim[1],
				props.maxThreadsDim[2]);
		printf("  Max grid dimensions:  [ %d, %d, %d ]\n\n=========\n\n",
				props.maxGridSize[0], props.maxGridSize[1],
				props.maxGridSize[2]);
	}
}

void computeFPS() {
	frameCount++;
	fpsCount++;

	if (fpsCount == fpsLimit) {
		char fps[256];
		float ifps = 1.f / (sdkGetAverageTimerValue(&timer) / 1000.f);
		delta = sdkGetAverageTimerValue(&timer)/1000;
		sprintf(fps, "Sim: %3.1f fps \t delta :%f ", ifps,delta);

		glutSetWindowTitle(fps);
		fpsCount = 0;

		//fpsLimit = (int)MAX(ifps, 1.f);
		sdkResetTimer(&timer);
	}
}

void runSimulation(TColor *d_dst) {
	switch (g_Kernel) {
	case 0:
		break;
	case 1:

		cuda_draw(d_dst, d_array, imageW, imageH, num_of_bodies);

		//printf("cuda_calculate delta : %f\n",delta);
		cuda_calculate(d_array, imageW, imageH, num_of_bodies,delta);

		//printf("cuda_draw\n");


		break;
	}

	getLastCudaError("Filtering kernel execution failed.\n");
}

void displayFunc(void) {

	sdkStartTimer(&timer);
	TColor *d_dst = NULL;
	size_t num_bytes;

	if (frameCounter++ == 0) {
		sdkResetTimer(&timer);
	}

	// DEPRECATED:
	//checkCudaErrors(cudaGLMapBufferObject((void**)&d_dst, gl_PBO));
	checkCudaErrors(cudaGraphicsMapResources(1, &cuda_pbo_resource, 0));
	getLastCudaError("cudaGraphicsMapResources failed");
	checkCudaErrors(
			cudaGraphicsResourceGetMappedPointer((void ** ) &d_dst, &num_bytes,
					cuda_pbo_resource));
	getLastCudaError("cudaGraphicsResourceGetMappedPointer failed");

	checkCudaErrors(CUDA_Bind2TextureArray());

	runSimulation(d_dst);

	checkCudaErrors(CUDA_UnbindTexture());
// DEPRECATED: checkCudaErrors(cudaGLUnmapBufferObject(gl_PBO));
	checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0));

// Common display code path
	{
		glClear(GL_COLOR_BUFFER_BIT);

		glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, imageW, imageH, GL_RGBA,
				GL_UNSIGNED_BYTE, BUFFER_DATA(0));
		glBegin(GL_TRIANGLES);
		glTexCoord2f(0, 0);
		glVertex2f(-1, -1);
		glTexCoord2f(2, 0);
		glVertex2f(+3, -1);
		glTexCoord2f(0, 2);
		glVertex2f(-1, +3);
		glEnd();
		glFinish();
	}

	if (frameCounter == frameN) {
		frameCounter = 0;

		if (g_FPS) {
			printf("FPS: %3.1f\n", frameN / (sdkGetTimerValue(&timer) * 0.001));
			g_FPS = false;
		}
	}

	glutSwapBuffers();

	sdkStopTimer(&timer);
	computeFPS();
}

void timerEvent(int value) {
	glutPostRedisplay();
	glutTimerFunc(REFRESH_DELAY, timerEvent, 0);
}

void shutDown(unsigned char k, int /*x*/, int /*y*/) {
	switch (k) {
	case '\033':
	case 'q':
	case 'Q':
		printf("Shutting down...\n");

		sdkStopTimer(&timer);
		sdkDeleteTimer(&timer);

		checkCudaErrors(CUDA_FreeArray());
		free(h_Src);

		exit(EXIT_SUCCESS);
		break;
	}
}

int initGL(int *argc, char **argv) {
	printf("Initializing GLUT...\n");
	glutInit(argc, argv);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
	glutInitWindowSize(imageW, imageH);
	glutInitWindowPosition(512 - imageW / 2, 384 - imageH / 2);
	glutCreateWindow(argv[0]);
	printf("OpenGL window created.\n");

	glewInit();
	printf("Loading extensions: %s\n", glewGetErrorString(glewInit()));

	if (!glewIsSupported(
			"GL_VERSION_1_5 GL_ARB_vertex_buffer_object GL_ARB_pixel_buffer_object")) {
		fprintf(stderr, "Error: failed to get minimal extensions for demo\n");
		fprintf(stderr, "This sample requires:\n");
		fprintf(stderr, "  OpenGL version 1.5\n");
		fprintf(stderr, "  GL_ARB_vertex_buffer_object\n");
		fprintf(stderr, "  GL_ARB_pixel_buffer_object\n");
		fflush(stderr);
		return false;
	}

	return 0;
}

// shader for displaying floating-point texture
static const char *shader_code = "!!ARBfp1.0\n"
		"TEX result.color, fragment.texcoord, texture[0], 2D; \n"
		"END";

GLuint compileASMShader(GLenum program_type, const char *code) {
	GLuint program_id;
	glGenProgramsARB(1, &program_id);
	glBindProgramARB(program_type, program_id);
	glProgramStringARB(program_type, GL_PROGRAM_FORMAT_ASCII_ARB,
			(GLsizei) strlen(code), (GLubyte *) code);

	GLint error_pos;
	glGetIntegerv(GL_PROGRAM_ERROR_POSITION_ARB, &error_pos);

	if (error_pos != -1) {
		const GLubyte *error_string;
		error_string = glGetString(GL_PROGRAM_ERROR_STRING_ARB);
		fprintf(stderr, "Program error at position: %d\n%s\n", (int) error_pos,
				error_string);
		return 0;
	}

	return program_id;
}

void initOpenGLBuffers() {
	printf("Creating GL texture...\n");
	glEnable(GL_TEXTURE_2D);
	glGenTextures(1, &gl_Tex);
	glBindTexture(GL_TEXTURE_2D, gl_Tex);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, imageW, imageH, 0, GL_RGBA,
			GL_UNSIGNED_BYTE, h_Src);
	printf("Texture created.\n");

	printf("Creating PBO...\n");

	glGenBuffers(1, &gl_PBO);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, gl_PBO);
	glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, imageW * imageH * 4, h_Src,
			GL_STREAM_COPY);
//While a PBO is registered to CUDA, it can't be used
//as the destination for OpenGL drawing calls.
//But in our particular case OpenGL is only used
//to display the content of the PBO, specified by CUDA kernels,
//so we need to register/unregister it only once.
// DEPRECATED: checkCudaErrors(cudaGLRegisterBufferObject(gl_PBO) );
	checkCudaErrors(
			cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, gl_PBO,
					cudaGraphicsMapFlagsWriteDiscard));
	GLenum gl_error = glGetError();

	if (gl_error != GL_NO_ERROR) {
		fprintf(stderr, "GL Error in file '%s' in line %d :\n", __FILE__,
				__LINE__);
		fprintf(stderr, "%s\n", gluErrorString(gl_error));
		exit(EXIT_FAILURE);
	}

	printf("PBO created.\n");

// load shader program
	shader = compileASMShader(GL_FRAGMENT_PROGRAM_ARB, shader_code);
}

void cleanup() {
	sdkDeleteTimer(&timer);

	glDeleteProgramsARB(1, &shader);
}
float RandomFloat(float a, float b) {
	float random = ((float) rand()) / (float) RAND_MAX;
	float diff = b - a;
	float r = random * diff;
	return a + r;
}

int main(int argc, char **argv) {

	char *dump_file = NULL;
	int clearColorbit = 255 << 24 | 255 << 16 | 255 << 8 | 255;
	int n;

//float* array;
	pArgc = &argc;
	pArgv = argv;

	displayCUDAInfo();
	g_Kernel = 0;
	printf("starting...\n\n");

	printf("choose simulation : \n1 : draw 1 \n2 : none\n3 : none\n");
	n = 1;
	//scanf("%d", &n);
	switch (n) {
	case 1:
		num_of_bodies = 64;

		float temp;

		sh_size = sizeof(float) * num_of_bodies * 6;

		printf("number of bodies = %d \t sh_size = %d \n", num_of_bodies,
				sh_size * sizeof(float));
		h_array = (float*) malloc(sh_size);

		for (int i = 0; i < num_of_bodies * 6; i += 6) {
			h_array[i] = RandomFloat(0, imageW); 		//x
			h_array[i + 1] = RandomFloat(0, imageH); 	//y
			h_array[i + 2] = RandomFloat(-101, 101);		//vx
			h_array[i + 3] = RandomFloat(-101, 101);		//vy
			h_array[i + 4] = RandomFloat(4, 9); 		//radius
			h_array[i + 5] = h_array[i + 4];	// mass
			printf(
					"i : %d\t x : %f\t y : %f\t vx : %f\t vy : %f\t : rad : %f\t mass : %f\n",
					i/6, h_array[i], h_array[i + 1], h_array[i + 2],
					h_array[i + 3], h_array[i + 4], h_array[i + 5]);
		}

		g_Kernel = 1;
		break;
	default:
		g_Kernel = 0;
		break;
	}

	initGL(&argc, argv);
	cudaGLSetGLDevice(gpuGetMaxGflopsDeviceId());

	h_Src = (uchar4*) malloc(imageH * imageW * 4);
	memset(h_Src, clearColorbit, imageH * imageW * 4);

	checkCudaErrors(CUDA_MallocArray(&h_Src, imageW, imageH));

	//copying to device memory

	initOpenGLBuffers();

	if (g_Kernel != 0) {
		checkCudaErrors(cudaMalloc((void** ) &d_array, sh_size));
		checkCudaErrors(
				cudaMemcpy(d_array, h_array, sh_size, cudaMemcpyHostToDevice));

		printf("Starting GLUT main loop...\n");

		glutDisplayFunc(displayFunc);
		glutKeyboardFunc(shutDown);

		sdkCreateTimer(&timer);
		sdkStartTimer(&timer);

		glutTimerFunc(REFRESH_DELAY, timerEvent, 0);
		glutMainLoop();

// cudaDeviceReset causes the driver to clean up all state. While
// not mandatory in normal operation, it is good practice.  It is also
// needed to ensure correct operation when the application is being
// profiled. Calling cudaDeviceReset causes all profile data to be
// flushed before the application exits
		cudaDeviceReset();
		return 0;
		exit(EXIT_SUCCESS);
	}
}
