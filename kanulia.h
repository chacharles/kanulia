#ifndef _KANULIA_KERNAL_h_
#define _KANULIA_KERNAL_h_

#include <vector_types.h>

#define ABS(x) (x>0)?(x):(-(x))

// square distance where nothing is visible
#define OUTMANDELBOX 4.0

// distance for the back ground whatever it is
#define OUTBOX 10.0

// step distance for each computation
#define RAYSTEP 0.001

// bool yes if the form is cutted
#define CUTJULIA 0

// Flag for Julia type
#define JULIA2D 0
#define CLOUDJULIA 1
#define JULIA4D 2
#define DIRECTIMAGE 4
#define CROSSEYE 8
#define JULIA4DSLICE 16

// kind of field of view
#define KANULFOV 30. //0.7

// position of the observer
#define ZOBSERVER -2.0 //-3.0

// half space between eyes in crosseye mode
#define SPACEEYE -0.1

// how do cross eye cross
#define CROSSANGLE 0.00001



// 4D rotations angles for Julia 4D
/*__device__ __constant__ float aanglexw;
__device__ __constant__ float aangleyw;
__device__ __constant__ float aanglexy;
__device__ __constant__ float aanglexz;*/

void reshapeFunc(int w, int h);

extern "C" void RunJulia4Drepart(uchar4 *dst, const int imageW, const int imageH,
					const float4 Off,
					const float4 JS,
					const float4 angle,
					const double scale, const double scalei,
					const double xJOff, const double yJOff, const double scaleJ,
					const float xblur, const float yblur, // blur coeff for julia 4D
					const unsigned int maxgropix,
					const unsigned int gropix, const unsigned int nbloc, const unsigned int crn,
					const uchar4 colors, const int frame, const int animationFrame, const int numSMs, const int julia, const int julia4D,
					const bool cutjulia);

__device__ inline void HSL2RGB(float h, const float sl, const float ll, int *rc, int *gc, int *bc);

__device__ inline int CalcJulia4Dhue(const float xPos, const float yPos, const float zPos, const float wPos, const float4 JS, float *hue);
__device__ inline int CalcMandelBox4Dhue(const float xPos, const float yPos, const float zPos, const float wPos, float *hue);

__device__ inline int CalcMandel4Dcore(const float xPos, const float yPos, const float zPos, const float wPos, const float4 JS, float *hue);

extern "C" int inEmulationMode();

__device__ inline void rotate4(float *px, float *py, float *pz, float *pw, const float4 angle);
__device__ inline void rotate4inv(float *px, float *py, float *pz, float *pw, const float4 angle);
__device__ inline void rotate3(float *px, float *py, float *pz, const float4 angle);

#endif
