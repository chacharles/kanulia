/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/*
    Mandelbrot sample
    submitted by Mark Granger, NewTek

    CUDA 2.0 SDK - updated with double precision support
    CUDA 2.1 SDK - updated to demonstrate software block scheduling using atomics
    CUDA 2.2 SDK - updated with drawing of Julia sets by Konstantin Kolchin, NVIDIA
*/

// OpenGL Graphics includes
#include <GL/glew.h>
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#include <GL/wglew.h>
#endif
#if defined(__APPLE__) || defined(__MACOSX)
  #pragma clang diagnostic ignored "-Wdeprecated-declarations"
  #include <GLUT/glut.h>
  #ifndef glutCloseFunc
  #define glutCloseFunc glutWMCloseFunc
  #endif
#else
#include <GL/freeglut.h>
#endif

// CUDA runtime
// CUDA utilities and system includes
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <helper_functions.h>
#include <helper_cuda.h>
#include <helper_cuda_gl.h>
#include <rendercheck_gl.h>

// Includes
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cstdio>
#include <math.h>
#include <time.h>

#include "resource.h"
#include "kanulia.h"
#include "kanulia_kernel.h"
//#include "kanulia_gold.h"

#include <windows.h>
#include <ShellAPI.h>
#include <WinUser.h>


// Define the files that are to be save and the reference images for validation
/*const char *sOriginal[] =
{
"mandelbrot.ppm",
NULL
};*/

/*const char *sReference[] =
{
"reference_fp32.ppm",
"reference_fp64.ppm",
NULL
};*/

// Set to 1 to time frame generation
#define RUN_TIMING 0

// Random number macros
#define RANDOMSEED(seed) ((seed) = ((seed) * 1103515245 + 12345))
#define RANDOMBITS(seed, bits) ((unsigned int)RANDOMSEED(seed) >> (32 - (bits)))

#define REFRESH_DELAY     10 //ms

/* handle of the parameters windows */
HWND hParamsWnd = 0;

//OpenGL PBO and texture "names"
GLuint gl_PBO, gl_Tex, gl_Shader;
struct cudaGraphicsResource *cuda_pbo_resource; // handles OpenGL-CUDA exchange

//Source image on the host side
uchar4 *h_Src = 0;

//Original image width and height
int2 imageDim;

// Starting iteration limit
unsigned int crunch = 10;

// Start mandel selection 1/julia heigh of windows
int julia = 4;

// Start with
int julia4D = JULIA4D;

struct ViewParams {
	// Angle for julia4D view
	float4 angle;
	float4 vangle;

	// Starting position and scale
	float4 Off;
	double scale = 3.2;
	double scalei = 3.2;

	// Starting stationary position and scale motion
	double xdOff = 0.0;
	double ydOff = 0.0;
	double dscale = 1.0;

	// Starting position and scale for Julia
	double xJOff = -0.5;
	double yJOff = 0.0;
	double scaleJ = 3.2;

	// Starting Julia seed point
	float4 JSOff;

	// Origine, Destination and step for julia seed move
	float4 OriJSOff;
	float4 DesJSOff;
	double StepOffre = 1.;
	double StepOffim = 1.;

	// Starting stationary position and scale motion for Julia
	double xJdOff = 0.0;
	double yJdOff = 0.0;
	double dscaleJ = 1.0;

	// cut julia or not
	bool cutjulia = false;
};
struct ViewParams viewParams;

// Affichage bloc par bloc
unsigned int bloc = 0;
unsigned int nbloc = 1;
unsigned int sqrnb = 1;

// Affichage pixélisé de la julia
unsigned int maxgropix = 16; // taille du bloc maximal pixelisé de calcul rapide
unsigned int gropix = maxgropix; // taille du gros bloc pixelisé courant

// Starting animation frame and anti-aliasing pass
int animationFrame = 0;
int animationStep = 0;
unsigned int pass = 0;
unsigned int maxpass = 128;

// SHIFT ALT and CTRL status
int modifiers = 0;

// Starting color multipliers and random seed
int colorSeed = 0;
uchar4 colors;

// Timer ID
StopWatchInterface *hTimer = NULL; // pour l'affichage des fps
StopWatchInterface *hETATimer = NULL; // pour l'affichage du temps restant a calculer
float totalETAtime; // accumule le temps de chaque pass.

// User interface variables
int lastx = 0;
int lasty = 0;
bool leftClicked = false;
bool middleClicked = false;
bool rightClicked = false;

//bool haveDoubles = false;
int numSMs = 0;          // number of multiprocessors

// Auto-Verification Code
const int frameCheckNumber = 60;
int fpsCount = 0;        // FPS count for averaging
int fpsLimit = 1;        // FPS limit for sampling
unsigned int frameCount = 0;
unsigned int g_TotalErrors = 0;
bool g_Verify = false, g_AutoQuit = false;

//KanParam param;

// CheckFBO/BackBuffer class objects
//CheckRender       *g_CheckRender = NULL;

#define MAX_EPSILON 50

#define MAX(a,b) ((a > b) ? a : b)

#define BUFFER_DATA(i) ((char *)0 + i)

/*void AutoQATest()
{
if (g_CheckRender && g_CheckRender->IsQAReadback()) {
char temp[256];
sprintf(temp, "AutoTest: Mandelbrot");
glutSetWindowTitle(temp);

if (g_AutoQuit) {
printf("Summary: %d comparison error!\n", g_TotalErrors);
printf("Test %s!\n", (g_TotalErrors==0) ? "PASSED" : "FAILED");
exit(0);
}
}
}*/



/* open the parameters Dialog box */
void openParams() {
	MessageBox(0, "not yet !", "sorry", MB_OK);
/*	if (hParamsWnd == 0) {
		hParamsWnd = (0, MAKEINTRESOURCE(IDD_PARAMVIEW), 0, DialogParams, 0);
	}
	hParamsWnd*/
}

void openHelp()
{
	ShellExecute(NULL, "open", "http://code.google.com/p/kanulia/wiki/Control",
		NULL, NULL, SW_SHOWNORMAL);
	// System::Windows::Forms::MessageBox::Show("Hello, Windows Forms");

	// Application::Run(gcnew KanParam());
	// ShellExecute(0, "open", "http://code.google.com/p/kanulia/wiki/Control", 0, 0, 1);
	/*SHELLEXECUTEINFO	shellInfo = { 0 };

	shellInfo.cbSize	= sizeof( shellInfo );
	shellInfo.fMask		= SEE_MASK_NOCLOSEPROCESS | SEE_MASK_FLAG_NO_UI |
	SEE_MASK_CLASSNAME;
	shellInfo.lpFile	= "http://code.google.com/p/kanulia/wiki/Control";
	shellInfo.nShow		= SW_SHOWNORMAL;
	shellInfo.lpClass	= "htmlfile";		// "opennew" is not supported for HTTP class, only HTMLFile
	shellInfo.lpVerb	= "opennew";		// open in a new window

	ShellExecuteEx( &shellInfo );*/
}

void newpic()
{
	pass = 0;
	// pour les images directes, on met le gropix à 1
	if (julia4D & DIRECTIMAGE) maxgropix = 1;
	gropix = maxgropix;
	bloc = 0;
	nbloc = 1;
	sqrnb = 1;
	sdkResetTimer(&hETATimer);
	totalETAtime = 0.f;
}

void computeFPS()
{
	frameCount++;
	fpsCount++;
	if (fpsCount == fpsLimit - 1) {
		g_Verify = true;
	}
	if (fpsCount == fpsLimit) {
		char fps[256];
		float ifps = 1.f / (sdkGetAverageTimerValue(&hTimer) / 1000.f);
		float ETAtimer = sdkGetTimerValue(&hETATimer) / 1000.f;

		if (pass < maxpass)
		{
			if (pass == 0)
			{

				if (bloc == 0)
					sprintf(fps, "%Kanulia START %3.1f fps", ifps);
				else
				{
					float timeleft = (ETAtimer)*(float)(maxgropix*maxgropix) / (float)(bloc);
					float tottimeleft = timeleft*maxpass - ETAtimer;
					timeleft -= ETAtimer;

					int hh = (int)(tottimeleft) / 3600;
					int mm = ((int)(tottimeleft) / 60) % 60;
					int ss = (int)(tottimeleft) % 60;

					int mm2 = (int)(timeleft) / 60;
					int ss2 = (int)(timeleft) % 60;
					sprintf(fps, "%Kanulia %02d:%02d -- %02d:%02d:%02d -- %3.1f fps", mm2, ss2, hh, mm, ss, ifps);
				}
			}
			else
			{
				if (bloc == 0)
					totalETAtime = (ETAtimer)*(float)(maxpass) / (float)(pass);
				float timeleft = totalETAtime - ETAtimer;
				int hh = (int)(timeleft) / 3600;
				int mm = ((int)(timeleft) / 60) % 60;
				int ss = (int)(timeleft) % 60;
				float perc = (ETAtimer / totalETAtime)*100.f;
				sprintf(fps, "%Kanulia %04.2f%% -- %02d:%02d:%02d -- %3.1f fps", perc, hh, mm, ss, ifps);
			}
		}
		else
			sprintf(fps, "%Kanulia DONE %3.1f fps", ifps);
		//         ((g_CheckRender && g_CheckRender->IsQAReadback()) ? "AutoTest: " : ""), ifps);

		glutSetWindowTitle(fps);
		fpsCount = 0;
		//        if (g_CheckRender && !g_CheckRender->IsQAReadback()) fpsLimit = (int)MAX(ifps, 1.f);

		//cutilCheckError(
		sdkResetTimer(&hTimer);

		//        AutoQATest();
	}
}

// Get a sub-pixel sample location
void GetSample(int sampleIndex, float &x, float &y)
{
	static const unsigned char pairData[128][2] = {
		{ 64, 64 }, { 0, 0 }, { 1, 63 }, { 63, 1 }, { 96, 32 }, { 97, 95 }, { 36, 96 }, { 30, 31 },
		{ 95, 127 }, { 4, 97 }, { 33, 62 }, { 62, 33 }, { 31, 126 }, { 67, 99 }, { 99, 65 }, { 2, 34 },
		{ 81, 49 }, { 19, 80 }, { 113, 17 }, { 112, 112 }, { 80, 16 }, { 115, 81 }, { 46, 15 }, { 82, 79 },
		{ 48, 78 }, { 16, 14 }, { 49, 113 }, { 114, 48 }, { 45, 45 }, { 18, 47 }, { 20, 109 }, { 79, 115 },
		{ 65, 82 }, { 52, 94 }, { 15, 124 }, { 94, 111 }, { 61, 18 }, { 47, 30 }, { 83, 100 }, { 98, 50 },
		{ 110, 2 }, { 117, 98 }, { 50, 59 }, { 77, 35 }, { 3, 114 }, { 5, 77 }, { 17, 66 }, { 32, 13 },
		{ 127, 20 }, { 34, 76 }, { 35, 110 }, { 100, 12 }, { 116, 67 }, { 66, 46 }, { 14, 28 }, { 23, 93 },
		{ 102, 83 }, { 86, 61 }, { 44, 125 }, { 76, 3 }, { 109, 36 }, { 6, 51 }, { 75, 89 }, { 91, 21 },
		{ 60, 117 }, { 29, 43 }, { 119, 29 }, { 74, 70 }, { 126, 87 }, { 93, 75 }, { 71, 24 }, { 106, 102 },
		{ 108, 58 }, { 89, 9 }, { 103, 23 }, { 72, 56 }, { 120, 8 }, { 88, 40 }, { 11, 88 }, { 104, 120 },
		{ 57, 105 }, { 118, 122 }, { 53, 6 }, { 125, 44 }, { 43, 68 }, { 58, 73 }, { 24, 22 }, { 22, 5 },
		{ 40, 86 }, { 122, 108 }, { 87, 90 }, { 56, 42 }, { 70, 121 }, { 8, 7 }, { 37, 52 }, { 25, 55 },
		{ 69, 11 }, { 10, 106 }, { 12, 38 }, { 26, 69 }, { 27, 116 }, { 38, 25 }, { 59, 54 }, { 107, 72 },
		{ 121, 57 }, { 39, 37 }, { 73, 107 }, { 85, 123 }, { 28, 103 }, { 123, 74 }, { 55, 85 }, { 101, 41 },
		{ 42, 104 }, { 84, 27 }, { 111, 91 }, { 9, 19 }, { 21, 39 }, { 90, 53 }, { 41, 60 }, { 54, 26 },
		{ 92, 119 }, { 51, 71 }, { 124, 101 }, { 68, 92 }, { 78, 10 }, { 13, 118 }, { 7, 84 }, { 105, 4 }
	};

	x = (1.0f / (float)maxpass) * (0.5f + (float)pairData[sampleIndex][0]);
	y = (1.0f / (float)maxpass) * (0.5f + (float)pairData[sampleIndex][1]);
} // GetSample

void rotate4(float4 *p, const float4 angle)
{
	float t;
	if (angle.x != 0.) {
		t = (*p).y * cos(angle.x) + (*p).z * sin(angle.x);
		(*p).z = -(*p).y * sin(angle.x) + (*p).z * cos(angle.x);
		(*p).y = t;
	};
	if (angle.y != 0.) {
		t = (*p).x * cos(angle.y) + (*p).z * sin(angle.y);
		(*p).z = -(*p).x * sin(angle.y) + (*p).z * cos(angle.y);
		(*p).x = t;
	};
	if (angle.z != 0.) {
		t = (*p).z * cos(angle.z) + (*p).w * sin(angle.z);
		(*p).w = -(*p).z * sin(angle.z) + (*p).w * cos(angle.z);
		(*p).z = t;
	};
	if (angle.w != 0.) {
		t = (*p).y * cos(angle.w) + (*p).w * sin(angle.w);
		(*p).w = -(*p).y * sin(angle.w) + (*p).w * cos(angle.w);
		(*p).y = t;
	};
}

// OpenGL display function
void displayFunc(void)
{
	if (viewParams.StepOffre < 1.)
	{
		viewParams.JSOff.x = viewParams.OriJSOff.x + (viewParams.DesJSOff.x - viewParams.OriJSOff.x)*(float)viewParams.StepOffre;
		viewParams.JSOff.y = viewParams.OriJSOff.y + (viewParams.DesJSOff.y - viewParams.OriJSOff.y)*(float)viewParams.StepOffre;

		viewParams.StepOffre += 0.003;
		//		printf("StepOffre\n");
		newpic();
	}
	if (viewParams.StepOffim < 1.)
	{
		viewParams.JSOff.z = viewParams.OriJSOff.z + (viewParams.DesJSOff.z - viewParams.OriJSOff.z)*(float)viewParams.StepOffim;
		viewParams.JSOff.w = viewParams.OriJSOff.w + (viewParams.DesJSOff.w - viewParams.OriJSOff.w)*(float)viewParams.StepOffim;

		viewParams.StepOffim += 0.003;
		//		printf("StepOffim\n");
		newpic();
	}
	if (viewParams.vangle.x != 0.)
	{
		viewParams.angle.x += viewParams.vangle.x;
		newpic();
	}
	if (viewParams.vangle.y != 0.)
	{
		viewParams.angle.y += viewParams.vangle.y;
		newpic();
	}
	if (viewParams.vangle.z != 0.)
	{
		viewParams.angle.z += viewParams.vangle.z;
		newpic();
	}
	if (viewParams.vangle.w != 0.)
	{
		viewParams.angle.w += viewParams.vangle.w;
		newpic();
	}
	if ((viewParams.xdOff != 0.0) || (viewParams.ydOff != 0.0)) {
		viewParams.Off.x += (float)viewParams.xdOff;
		viewParams.Off.y += (float)viewParams.ydOff;
		printf("xdOff\n");
		newpic();
	}
	if (viewParams.dscale != 1.0) {
		viewParams.scale *= viewParams.dscale;
		printf("dscale\n");
		newpic();
	}
	if (animationStep) {
		animationFrame -= animationStep;
		printf("animationStep\n");
		newpic();
	}

#if RUN_TIMING
	pass = 0;
#endif

	if (pass < maxpass) {
		float timeEstimate;
		int startPass = pass;
		uchar4 *d_dst = NULL;
		sdkResetTimer(&hTimer);
		// cutilSafeCall(
		// DEPRECATED: checkCudaErrors(cudaGLMapBufferObject((void**)&d_dst, gl_PBO));
		checkCudaErrors(cudaGraphicsMapResources(1, &cuda_pbo_resource, 0));
		size_t num_bytes;
		checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&d_dst, &num_bytes, cuda_pbo_resource));

		// Render anti-aliasing passes until we run out time (60fps approximately)
		do {
			float xs, ys;

			// Get the anti-alias sub-pixel sample location
			GetSample(pass & 127, xs, ys);

			// Get the pixel scale and offset
			double s = viewParams.scale / (float)imageDim.x;
			double si = viewParams.scalei / (float)imageDim.x;
			float4 Pos = { (xs - (float)imageDim.x * 0.5f) * s + viewParams.Off.x,
				(ys - (float)imageDim.y * 0.5f) * s + viewParams.Off.y,
				(xs - (float)imageDim.x * 0.5f) * si + viewParams.Off.z,
				(ys - (float)imageDim.y * 0.5f) * si + viewParams.Off.w
			};
			// same for Julia
			double sj = viewParams.scaleJ / (float)imageDim.x;
			double xj;
			double yj;
			if (julia4D == JULIA2D) // blury in 2D mode
			{
				xj = (xs - (double)imageDim.x * 0.5f) * sj + viewParams.xJOff;
				yj = (ys - (double)imageDim.y * 0.5f) * sj + viewParams.yJOff;
			}
			else // but not if in 4D mode
			{
				xj = (0.5f - (double)imageDim.x * 0.5f) * sj + viewParams.xJOff;
				yj = (0.5f - (double)imageDim.y * 0.5f) * sj + viewParams.yJOff;
			}

			int rebloc = 0;
			int bbloc = bloc;
			for (unsigned int mx = sqrnb / 2; mx >= 1; mx /= 2)
			{
				switch (bbloc % 4)
				{
					/*			case 0:
					rebloc += 0;
					break;*/
				case 1:
					rebloc += (sqrnb*mx) + mx; // une ligne taille mx et un mx pixel
					break;
				case 2:
					rebloc += mx;
					break;
				case 3:
					rebloc += (sqrnb*mx);
					break;
				}
				bbloc = bbloc / 4;
			}

			RunJulia4Drepart(d_dst, // destination buffer
				imageDim.x, imageDim.y, // windows size
				Pos, viewParams.JSOff, viewParams.angle, s, si, xj, yj, sj, // seed point and scale for mandel brod and julia
				(2.0 * xs) - 1.0, (2.0 * ys) - 1.0, // blur modification
				maxgropix, gropix, rebloc, crunch,
				colors, pass, // color palette, and pass number
				animationFrame,
				numSMs, julia, julia4D, viewParams.cutjulia);

			cudaDeviceSynchronize();

			// Estimate the total time of the frame if one more pass is rendered
			timeEstimate = 0.001f * sdkGetTimerValue(&hTimer) * ((float)(pass + 1 - startPass) / (float)((pass - startPass) ? (pass - startPass) : 1));
			printf("startpass=%d pass=%d M=%d gropix=%d blc= %d rblc=%d nblc=%d Estimate=%5.8f\n", startPass, pass, maxgropix, gropix, bloc, rebloc, nbloc, timeEstimate);

			// ajustage du maxgropix en fonction du temps mis pour calculer
			if (gropix == maxgropix) // on est dans la pass la plus grossiere
			{
				if ((maxgropix>  1) && (timeEstimate<1. / 30.))
				{
					maxgropix /= 2;
					newpic();
				}
				if ((maxgropix<16) && (timeEstimate>1. / 8.))
				{
					maxgropix *= 2;
					newpic();
				}
			}

			bloc++;
			if (bloc == nbloc)
			{
				// si on viens de terminer une passe au niveau le plus fin, on incremente
				if (gropix == 1) pass++;

				if (gropix>1) gropix /= 2;

				bloc = 0;

				// On calcul le nombre de bloc
				sqrnb = maxgropix / gropix;
				nbloc = sqrnb*sqrnb;
				//				nbloc=1;
				//				for (unsigned int mx=maxgropix;mx>gropix;mx/=2) nbloc*=4;
				// si on est dans la 1ere image, on éviteles cases deja affiché
				if (pass == 0) bloc = (nbloc / 4);
			}


		} while ((pass < maxpass) && (timeEstimate < 1.0f / 60.0f) && !RUN_TIMING);
		//cutilSafeCall(cudaGLUnmapBufferObject(gl_PBO);
		checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0));
#if RUN_TIMING
		printf("GPU = %5.8f\n", 0.001f * cutGetTimerValue(hTimer));
#endif
	}

	// display image
	glBindProgramARB(GL_FRAGMENT_PROGRAM_ARB, gl_Shader);
	glBindTexture(GL_TEXTURE_2D, gl_Tex);
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, imageDim.x, imageDim.y, GL_RGBA, GL_UNSIGNED_BYTE, BUFFER_DATA(0));

	glBegin(GL_QUADS);
	glTexCoord2f(0.0f, 0.0f); glVertex2f(0.0f, 0.0f);
	glTexCoord2f(1.0f, 0.0f); glVertex2f(1.0f, 0.0f);
	glTexCoord2f(1.0f, 1.0f); glVertex2f(1.0f, 1.0f);
	glTexCoord2f(0.0f, 1.0f); glVertex2f(0.0f, 1.0f);
	glEnd();

	glBindTexture(GL_TEXTURE_2D, 0);
	glDisable(GL_FRAGMENT_PROGRAM_ARB);
	/*   if (g_CheckRender && g_CheckRender->IsQAReadback() && g_Verify) {
	printf("> (Frame %d) Readback BackBuffer\n", frameCount);
	g_CheckRender->readback( imageDim.x, imageDim.y, (GLuint)NULL );
	g_CheckRender->savePPM ( sOriginal[0], true, NULL);
	if (!g_CheckRender->PPMvsPPM(sOriginal[0], sReference[haveDoubles], MAX_EPSILON)) {
	g_TotalErrors++;
	}
	g_Verify = false;
	g_AutoQuit = true;
	}*/

 	//sdkStopTimer(&hTimer);
	glutSwapBuffers();

	computeFPS();
} // displayFunc

void cleanup()
{

	if (h_Src) {
		free(h_Src);
		h_Src = 0;
	}

	//    if (g_CheckRender) {
	//        delete g_CheckRender; g_CheckRender = NULL;
	//    }
	sdkStopTimer(&hTimer);
	sdkDeleteTimer(&hTimer);

	//DEPRECATED: checkCudaErrors(cudaGLUnregisterBufferObject(gl_PBO));
	cudaGraphicsUnregisterResource(cuda_pbo_resource);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

	glDeleteBuffers(1, &gl_PBO);
	glDeleteTextures(1, &gl_Tex);
	glDeleteProgramsARB(1, &gl_Shader);
}

// OpenGL keyboard function
void keyboardFunc(unsigned char k, int x, int y)
{
//	int seed;
	switch (k){
	case '\033':
	case 'q':
	case 'Q':
		printf("Shutting down...\n");
		/*cutilCheckError(*/sdkStopTimer(&hTimer);
		/*cutilCheckError(*/sdkDeleteTimer(&hTimer);
		/*cutilCheckError(*/sdkStopTimer(&hETATimer);
		/*cutilCheckError(*/sdkDeleteTimer(&hETATimer);
		/*cutilSafeCall(*/cudaGLUnregisterBufferObject(gl_PBO);
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
		printf("Shutdown done.\n");
		exit(0);
		break;

	case '?':
	case 'h':
	case 'H':
		printf("Off.x = %5.8f\n", viewParams.Off.x);
		printf("Off.y = %5.8f\n", viewParams.Off.y);
		printf("scale = %e\n", viewParams.scale);
		printf("detail = %d\n", crunch);
		printf("color = %d\n", colorSeed);
		printf("\n");
		openHelp();
		break;

	case 'r': case 'R':
		// Reset all values to their defaults
		viewParams.Off.x = -0.5;
		viewParams.Off.y = 0.0;
		viewParams.scale = 3.2;
		viewParams.xdOff = 0.0;
		viewParams.ydOff = 0.0;
		viewParams.dscale = 1.0;
		colorSeed = 0;
		colors.x = 3;
		colors.y = 5;
		colors.z = 7;
		crunch = 10;
		animationFrame = 0;
		animationStep = 0;
		newpic();
		viewParams.angle.x = 0.;
		viewParams.angle.y = 0.;
		viewParams.angle.z = 0.;
		viewParams.angle.w = 0.;
		viewParams.vangle.x = 0.;
		viewParams.vangle.y = 0.;
		viewParams.vangle.z = 0.;
		viewParams.vangle.w = 0.;
		viewParams.cutjulia = 0;
		break;

	case 'c':	
	case 'C':
		viewParams.cutjulia = !viewParams.cutjulia;
		newpic();
		break;
	/*	case 'c':
	seed = ++colorSeed;
		if (seed) {
			colors.x = RANDOMBITS(seed, 4);
			colors.y = RANDOMBITS(seed, 4);
			colors.z = RANDOMBITS(seed, 4);
		}
		else {
			colors.x = 3;
			colors.y = 5;
			colors.z = 7;
		}
		newpic();
		break;

	case 'C':
		seed = --colorSeed;
		if (seed) {
			colors.x = RANDOMBITS(seed, 4);
			colors.y = RANDOMBITS(seed, 4);
			colors.z = RANDOMBITS(seed, 4);
		}
		else {
			colors.x = 3;
			colors.y = 5;
			colors.z = 7;
		}
		newpic();
		break;*/

	case 'a':
		if (animationStep < 0)
			animationStep = 0;
		else {
			animationStep++;
			if (animationStep > 8)
				animationStep = 8;
		}
		break;

	case 'A':
		if (animationStep > 0)
			animationStep = 0;
		else {
			animationStep--;
			if (animationStep < -8)
				animationStep = -8;
		}
		break;

	case 'd':
		if (crunch < 0x4000) {
			if (crunch < 128) crunch += 1;
			else crunch *= 2;
			newpic();
		}
		printf("detail = %d\n", crunch);
		break;

	case 'D':
		if (crunch > 2) {
			if (crunch <= 128) crunch -= 1;
			else crunch /= 2;
			newpic();
		}
		printf("detail = %d\n", crunch);
		break;

	case 'j':
		if (julia < 32) {
			julia *= 2;
			newpic();
		}
		printf("julia = %d\n", julia);
		break;

	case 'J':
		if (julia > 1) {
			julia /= 2;
			newpic();
		}
		printf("julia = %d\n", julia);
		break;

	case '4':	// Left arrow key
		viewParams.Off.x -= 0.05f * (float)viewParams.scale;
		newpic();
		break;

	case '8':	// Up arrow key
		viewParams.Off.y += 0.05f * (float)viewParams.scale;
		newpic();
		break;

	case '6':	// Right arrow key
		viewParams.Off.x += 0.05f * (float)viewParams.scale;
		newpic();
		break;

	case '2':	// Down arrow key
		viewParams.Off.y -= 0.05f *(float)viewParams.scale;
		newpic();
		break;

	case '+':
		if ((julia < 32) && (x < imageDim.x / julia) && (y  > imageDim.y - imageDim.y / julia))
		{
			viewParams.scale /= 1.1f;
			viewParams.Off.x += (x - (double)(imageDim.x / julia) / 2.) * 0.1 * (viewParams.scale / (double)(imageDim.x / julia));
			viewParams.Off.y -= (y - (double)(imageDim.y - imageDim.y / (2 * julia))) * 0.1 * (viewParams.scale / (double)(imageDim.x / julia));
		}
		else if ((julia < 32) && ((imageDim.x - x) < imageDim.x / julia) && (y  > imageDim.y - imageDim.y / julia))
		{
			viewParams.scalei /= 1.1f;
			viewParams.Off.z += ((imageDim.x - x) - (double)(imageDim.x / julia) / 2.)          * 0.1 * (viewParams.scalei / (double)(imageDim.x / julia));
			viewParams.Off.w -= (y - (double)(imageDim.y - imageDim.y / (2 * julia))) * 0.1 * (viewParams.scalei / (double)(imageDim.x / julia));
		}
		else {
			viewParams.scaleJ /= 1.1f;
			viewParams.xJOff += (x - (double)(imageDim.x) / 2.) * 0.1 * (viewParams.scaleJ / (double)imageDim.x);
			viewParams.yJOff -= (y - (double)(imageDim.y) / 2.) * 0.1 * (viewParams.scaleJ / (double)imageDim.x);
		};
		printf("Zoom In\n");
		newpic();
		break;

	case '-':
		newpic();
		if ((julia < 32) && (x < imageDim.x / julia) && (y  > imageDim.y - imageDim.y / julia))
		{
			viewParams.Off.x -= (x - (double)(imageDim.x / julia) / 2.) * 0.1 * (viewParams.scale / (double)(imageDim.x / julia));
			viewParams.Off.y += (y - (double)(imageDim.y - imageDim.y / (2 * julia))) * 0.1 * (viewParams.scale / (double)(imageDim.x / julia));
			viewParams.scale *= 1.1f;
		}
		else if ((julia < 32) && ((imageDim.x - x) < imageDim.x / julia) && (y  > imageDim.y - imageDim.y / julia))
		{
			viewParams.Off.z -= ((imageDim.x - x) - (double)(imageDim.x / julia) / 2.)           * 0.1 * (viewParams.scalei / (double)(imageDim.x / julia));
			viewParams.Off.w += (y - (double)(imageDim.y - imageDim.y / (2 * julia))) * 0.1 * (viewParams.scalei / (double)(imageDim.x / julia));
			viewParams.scalei *= 1.1f;
		}
		else {
			viewParams.xJOff -= (x - (double)(imageDim.x) / 2.) * 0.1 * (viewParams.scaleJ / (double)imageDim.x);
			viewParams.yJOff += (y - (double)(imageDim.y) / 2.) * 0.1 * (viewParams.scaleJ / (double)imageDim.x);
			viewParams.scaleJ *= 1.1f;
		};
		("Zoom Out\n");
		newpic();
		break;

	case 'u':
		viewParams.vangle.x += 0.001f;
		break;
	case 'i':
		viewParams.vangle.y += 0.001f;
		break;
	case 'o':
		viewParams.vangle.z += 0.001f;
		break;
	case 'p':
		viewParams.vangle.w += 0.001f;
		break;
	case 'U':
		viewParams.vangle.x -= 0.001f;
		break;
	case 'I':
		viewParams.vangle.y -= 0.001f;
		break;
	case 'O':
		viewParams.vangle.z -= 0.001f;
		break;
	case 'P':
		viewParams.vangle.w -= 0.001f;
		break;
	case 'w':
		imageDim.x = 2560;
		imageDim.y = 1600;

		glutReshapeWindow(imageDim.x, imageDim.y);

		break;

	case 'g': // Direct Image On/Off pas de pixelisation progressive
		if (julia4D & JULIA4D)
		{
			julia4D = julia4D ^ DIRECTIMAGE;
			printf("julia4D = %d\n", julia4D);
		}
		newpic();
		break;

	case 'f': // Cross Eye On/Off
		if (julia4D & JULIA4D)
		{
			julia4D = julia4D ^ CROSSEYE;
			printf("julia4D = %d\n", julia4D);
		}
		newpic();
		break;
	case 'k': // Cross Eye On/Off
		if (julia4D & JULIA4D)
		{
			julia4D = julia4D ^ JULIA4DSLICE;
			printf("julia4D = %d\n", julia4D);
		}
		newpic();
		break;
	case '²':
	case '~':
		openParams();
		break;

	default:
		break;
	}

} // keyboardFunc

// OpenGL mouse click function
void mouseFunc(int button, int state, int x, int y)
{
	modifiers = glutGetModifiers();

	// left mouse button
	if (button == 0)
	{
		leftClicked = !leftClicked;
		// in the mandelbro select
		if ((julia < 32) && (x < imageDim.x / julia) && (y  > imageDim.y - imageDim.y / julia))
		{
			// real seed point
			viewParams.StepOffre = 1.;
			viewParams.JSOff.x = viewParams.Off.x + (x - (double)(imageDim.x / julia) / 2.) * (viewParams.scale / (double)(imageDim.x / julia));
			viewParams.JSOff.y = viewParams.Off.y - (y - (double)(imageDim.y - imageDim.y / (2 * julia))) * (viewParams.scale / (double)(imageDim.x / julia));

			newpic();
		};
		if ((julia < 32) && ((imageDim.x - x) < imageDim.x / julia) && (y  > imageDim.y - imageDim.y / julia))
		{
			// imaginary seed point
			viewParams.StepOffim = 1.;
			viewParams.JSOff.z = viewParams.Off.z + ((imageDim.x - x) - (double)(imageDim.x / julia) / 2.)           * (viewParams.scalei / (double)(imageDim.x / julia));
			viewParams.JSOff.w = viewParams.Off.w - (y - (double)(imageDim.y - imageDim.y / (2 * julia))) * (viewParams.scalei / (double)(imageDim.x / julia));

			newpic();
		};
	}
	// middle mouse button
	if (button == 1) {
		// in the mandelbro select button released
		if ((state == GLUT_UP) && ((julia < 32) && (x < imageDim.x / julia) && (y  > imageDim.y - imageDim.y / julia))) {
			// printf("Middle Clicked %3.2f %3.2f \n" , ( x - (double) ( imageDim.y ) / 2. ) , ( y - (double) ( imageDim.y ) / 2. ) );

			viewParams.OriJSOff.x = viewParams.JSOff.x;
			viewParams.OriJSOff.y = viewParams.JSOff.y;
			viewParams.DesJSOff.x = viewParams.Off.x + (x - (double)(imageDim.x / julia) / 2.)           * (viewParams.scale / (double)(imageDim.x / julia));
			viewParams.DesJSOff.y = viewParams.Off.y - (y - (double)(imageDim.y - imageDim.y / (2 * julia))) * (viewParams.scale / (double)(imageDim.x / julia));
			viewParams.StepOffre = 0.;

			newpic();
		}
		// in the mandelbro select button released
		if ((state == GLUT_UP) && ((julia < 32) && ((imageDim.x - x) < imageDim.x / julia) && (y  > imageDim.y - imageDim.y / julia))) {
			// printf("Middle Clicked %3.2f %3.2f \n" , ( x - (double) ( imageDim.y ) / 2. ) , ( y - (double) ( imageDim.y ) / 2. ) );

			viewParams.OriJSOff.z = viewParams.JSOff.z;
			viewParams.OriJSOff.w = viewParams.JSOff.w;
			viewParams.DesJSOff.z = viewParams.Off.z + ((imageDim.x - x) - (double)(imageDim.x / julia) / 2.)* (viewParams.scalei / (double)(imageDim.x / julia));
			viewParams.DesJSOff.w = viewParams.Off.w - (y - (double)(imageDim.y - imageDim.y / (2 * julia))) * (viewParams.scalei / (double)(imageDim.x / julia));
			viewParams.StepOffim = 0.;

			newpic();
		}
		middleClicked = !middleClicked;
	}
	// right button
	if (button == 2)
		rightClicked = !rightClicked;

	/*    if (leftClicked && (modifiers & GLUT_ACTIVE_SHIFT))
	{
	leftClicked = 0;
	middleClicked = 1;
	}*/

	if (state == GLUT_UP) {
		leftClicked = 0;
		middleClicked = 0;

		// Used for wheels, has to be up
/*		if (button == GLUT_WHEEL_UP)
		{
			if ((julia < 32) && (x < imageDim.x / julia) && (y  > imageDim.y - imageDim.y / julia))
			{
				scale /= 1.1f;
				Off.x += (x - (double)(imageDim.x / julia) / 2.) * 0.1 * (scale / (double)(imageDim.x / julia));
				Off.y -= (y - (double)(imageDim.y - imageDim.y / (2 * julia))) * 0.1 * (scale / (double)(imageDim.x / julia));
			}
			else if ((julia < 32) && ((imageDim.x - x) < imageDim.x / julia) && (y  > imageDim.y - imageDim.y / julia))
			{
				scalei /= 1.1f;
				Off.z += ((imageDim.x - x) - (double)(imageDim.x / julia) / 2.)          * 0.1 * (scalei / (double)(imageDim.x / julia));
				Off.w -= (y - (double)(imageDim.y - imageDim.y / (2 * julia))) * 0.1 * (scalei / (double)(imageDim.x / julia));
			}
			else {
				scaleJ /= 1.1f;
				xJOff += (x - (double)(imageDim.x) / 2.) * 0.1 * (scaleJ / (double)imageDim.x);
				yJOff -= (y - (double)(imageDim.y) / 2.) * 0.1 * (scaleJ / (double)imageDim.x);
			};
			newpic();
			printf("Wheel Up\n");
		}
		else if (button == GLUT_WHEEL_DOWN)
		{
			if ((julia < 32) && (x < imageDim.x / julia) && (y  > imageDim.y - imageDim.y / julia))
			{
				Off.x -= (x - (double)(imageDim.x / julia) / 2.) * 0.1 * (scale / (double)(imageDim.x / julia));
				Off.y += (y - (double)(imageDim.y - imageDim.y / (2 * julia))) * 0.1 * (scale / (double)(imageDim.x / julia));
				scale *= 1.1f;
			}
			else if ((julia < 32) && ((imageDim.x - x) < imageDim.x / julia) && (y  > imageDim.y - imageDim.y / julia))
			{
				Off.z -= ((imageDim.x - x) - (double)(imageDim.x / julia) / 2.)           * 0.1 * (scalei / (double)(imageDim.x / julia));
				Off.w += (y - (double)(imageDim.y - imageDim.y / (2 * julia))) * 0.1 * (scalei / (double)(imageDim.x / julia));
				scalei *= 1.1f;
			}
			else {
				xJOff -= (x - (double)(imageDim.x) / 2.) * 0.1 * (scaleJ / (double)imageDim.x);
				yJOff += (y - (double)(imageDim.y) / 2.) * 0.1 * (scaleJ / (double)imageDim.x);
				scaleJ *= 1.1f;
			};
			newpic();
			printf("Wheel Down\n");
		}*/
	}

	lastx = x;
	lasty = y;
	viewParams.xdOff = 0.0;
	viewParams.ydOff = 0.0;
	viewParams.dscale = 1.0;
} // clickFunc

// OpenGL mouse motion function
void motionFunc(int x, int y)
{
	double fx = (double)((lastx - x) / 10.0) / (double)(imageDim.x);
	double fy = (double)((y - lasty) / 10.0) / (double)(imageDim.x);

	//    int modifiers = glutGetModifiers();

	if (leftClicked) {
		if ((julia < 32) && (x < imageDim.x / julia) && (y  > imageDim.y - imageDim.y / julia))
		{
			viewParams.JSOff.x = viewParams.Off.x + (x - (double)(imageDim.x / julia) / 2.0) * (viewParams.scale / (double)(imageDim.x / julia));
			viewParams.JSOff.y = viewParams.Off.y - (y - (double)(imageDim.y - imageDim.y / (2.0 * julia))) * (viewParams.scale / (double)(imageDim.x / julia));
			newpic();
		}
		else if ((julia < 32) && ((imageDim.x - x) < imageDim.x / julia) && (y  > imageDim.y - imageDim.y / julia))
		{
			viewParams.JSOff.z = viewParams.Off.z + ((imageDim.x - x) - (double)(imageDim.x / julia) / 2.0) * (viewParams.scalei / (double)(imageDim.x / julia));
			viewParams.JSOff.w = viewParams.Off.w - (y - (double)(imageDim.y - imageDim.y / (2.0 * julia))) * (viewParams.scalei / (double)(imageDim.x / julia));
			newpic();
		}
		else
		{
			if (leftClicked && (modifiers & GLUT_ACTIVE_SHIFT))
			{
				viewParams.angle.z -= (float)fx*100.0f;
				viewParams.angle.w -= (float)fy*100.0f;

				//			printf("Motion fx=%f fy=%f\n",fx,fy);
				//			printf("angle.z=%f angle.w=%f\n",angle.z,angle.w);
			}
			else
			{
				viewParams.angle.y -= (float)fx*100.0f;
				viewParams.angle.x -= (float)fy*100.0f;
				//	float4 delangle = {0.0,0.0,-fx*100.0,-fy*100.0};
				//	float4 delangle1st = {0.0,-fy*100.0,0.0,0.0};
				//	rotate4(&angle,delangle);
				/*				angle.x += delangle.x;
				angle.y += delangle.y;
				angle.z += delangle.z;
				angle.w += delangle.w;*/
				printf("Motion fx=%f fy=%f\n", fx, fy);
				printf("angle.y=%f angle.x=%f\n", viewParams.angle.y, viewParams.angle.x);
			}
			newpic();

		};
	}
	else {
		viewParams.xdOff = 0.0f;
		viewParams.ydOff = 0.0f;
	}

	if (middleClicked)
		if ((julia < 32) && (x < imageDim.x / julia) && (y  > imageDim.y - imageDim.y / julia))
		{
			if (fy > 0.0f) {
				viewParams.dscale = 1.0 - fy;
				viewParams.dscale = viewParams.dscale < 1.05 ? viewParams.dscale : 1.05;
			}
			else {
				viewParams.dscale = 1.0 / (1.0 + fy);
				viewParams.dscale = viewParams.dscale >(1.0 / 1.05) ? viewParams.dscale : (1.0 / 1.05);
			}
		}
		else {
			viewParams.angle.z -= (float)fx*100.0f;
			viewParams.angle.w -= (float)fy*100.0f;
			newpic();
			//			printf("Motion fx=%f fy=%f\n",fx,fy);
			//			printf("angle.z=%f angle.w=%f\n",angle.z,angle.w);

		}
	else
		viewParams.dscale = 1.0;

	lastx = x;
	lasty = y;
} // motionFunc

void idleFunc()
{
	glutPostRedisplay();
}

void mainMenu(int i)
{

	switch (i) {
	case 10:
		openHelp();
		break;
	case 20:
		openParams();
		break;
	}
	newpic();
}

void juliaMenu(int i)
{
	switch (i) {
	case 1:
		if (julia < 16) julia *= 2;
		break;
	case 2:
		if (julia > 1) julia /= 2;
		break;
	case 3:
		julia4D = JULIA2D; //"Flat Julia2D"
		break;
	case 4:
		viewParams.angle.x = 0.;
		viewParams.angle.y = 0.;
		viewParams.angle.z = 0.;
		viewParams.angle.w = 0.;
		crunch = 16;
		julia4D = CLOUDJULIA; //"Cloudy Julia4D"
		break;
	case 5:
		viewParams.angle.x = 0.;
		viewParams.angle.y = 0.;
		viewParams.angle.z = 0.;
		viewParams.angle.w = 0.;
		crunch = 16;
		julia4D = JULIA4D; //"Solid Julia4D"
		break;

	}
	newpic();
}

void colorMenu(int i)
{
	int seed;
	switch (i) {
	case 1:
		seed = --colorSeed;
		if (seed) {
			colors.x = RANDOMBITS(seed, 4);
			colors.y = RANDOMBITS(seed, 4);
			colors.z = RANDOMBITS(seed, 4);
		}
		else {
			colors.x = 3;
			colors.y = 5;
			colors.z = 7;
		}
		break;
	case 2:
		seed = ++colorSeed;
		if (seed) {
			colors.x = RANDOMBITS(seed, 4);
			colors.y = RANDOMBITS(seed, 4);
			colors.z = RANDOMBITS(seed, 4);
		}
		else {
			colors.x = 3;
			colors.y = 5;
			colors.z = 7;
		}
		break;
	case 3:
		if (animationStep < 0)
			animationStep = 0;
		else {
			animationStep++;
			if (animationStep > 8)
				animationStep = 8;
		}
		break;
	case 4:
		if (animationStep > 0)
			animationStep = 0;
		else {
			animationStep--;
			if (animationStep < -8)
				animationStep = -8;
		}
		break;
	};
	newpic();
}

void timerEvent(int value)
{
	glutPostRedisplay();
	glutTimerFunc(REFRESH_DELAY, timerEvent, 0);
}

void initMenus()
{
	int colormenu = glutCreateMenu(colorMenu);
	glutAddMenuEntry("Previous Palette", 1);
	glutAddMenuEntry("Next Palette", 2);
	glutAddMenuEntry("Forward Animation", 3);
	glutAddMenuEntry("Backward Animation", 4);

	int juliamenu = glutCreateMenu(juliaMenu);
	glutAddMenuEntry("Reduce selection", 1);
	glutAddMenuEntry("Increase Selection", 2);
	glutAddMenuEntry("Flat Julia2D", 3);
	glutAddMenuEntry("Cloudy Julia4D", 4);
	glutAddMenuEntry("Solid Julia4D", 5);

	glutCreateMenu(mainMenu);
//	glutAddSubMenu("Julia", juliamenu);
//	glutAddSubMenu("Color", colormenu);
	glutAddMenuEntry("Help", 10);
	glutAddMenuEntry("Params", 20);
	glutAttachMenu(GLUT_RIGHT_BUTTON);
}

// gl_Shader for displaying floating-point texture
static const char *shader_code =
"!!ARBfp1.0\n"
"TEX result.color, fragment.texcoord, texture[0], 2D; \n"
"END";

GLuint compileASMShader(GLenum program_type, const char *code)
{
	GLuint program_id;
	glGenProgramsARB(1, &program_id);
	glBindProgramARB(program_type, program_id);
	glProgramStringARB(program_type, GL_PROGRAM_FORMAT_ASCII_ARB, (GLsizei)strlen(code), (GLubyte *)code);

	GLint error_pos;
	glGetIntegerv(GL_PROGRAM_ERROR_POSITION_ARB, &error_pos);

	if (error_pos != -1)
	{
		const GLubyte *error_string;
		error_string = glGetString(GL_PROGRAM_ERROR_STRING_ARB);
		fprintf(stderr, "Program error at position: %d\n%s\n", (int)error_pos, error_string);
		return 0;
	}

	return program_id;
}

void createBuffers(int w, int h)
{
	if (h_Src) {
		free(h_Src);
		h_Src = 0;
	}

	if (gl_Tex) {
		glDeleteTextures(1, &gl_Tex);
		gl_Tex = 0;
	}
	if (gl_PBO) {
		// cudaGLUnregisterBufferObject(gl_PBO);
		cudaGraphicsUnregisterResource(cuda_pbo_resource);
		glDeleteBuffers(1, &gl_PBO);
		gl_PBO = 0;
	}

	// allocate new buffers
	h_Src = (uchar4 *)malloc(w * h * 4);

	printf("Creating GL texture...\n");
	glEnable(GL_TEXTURE_2D);
	glGenTextures(1, &gl_Tex);
	glBindTexture(GL_TEXTURE_2D, gl_Tex);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, h_Src);
	printf("Texture created.\n");

	printf("Creating PBO...\n");
	glGenBuffers(1, &gl_PBO);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, gl_PBO);
	glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, w * h * 4, h_Src, GL_STREAM_COPY);
	//While a PBO is registered to CUDA, it can't be used
	//as the destination for OpenGL drawing calls.
	//But in our particular case OpenGL is only used
	//to display the content of the PBO, specified by CUDA kernels,
	//so we need to register/unregister it only once.
	// cutilSafeCall( cudaGLRegisterBufferObject(gl_PBO);

	checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, gl_PBO,
		cudaGraphicsMapFlagsWriteDiscard));

	printf("PBO created.\n");

	// This is the buffer we use to readback results into
	gl_Shader = compileASMShader(GL_FRAGMENT_PROGRAM_ARB, shader_code);
}


void initOpenGLBuffers(int w, int h)
{
	// delete old buffers
	if (h_Src)
	{
		free(h_Src);
		h_Src = 0;
	}

	if (gl_Tex)
	{
		glDeleteTextures(1, &gl_Tex);
		gl_Tex = 0;
	}

	if (gl_PBO)
	{
		//DEPRECATED: checkCudaErrors(cudaGLUnregisterBufferObject(gl_PBO));
		cudaGraphicsUnregisterResource(cuda_pbo_resource);
		glDeleteBuffers(1, &gl_PBO);
		gl_PBO = 0;
	}

	// check for minimized window
	if ((w == 0) && (h == 0))
	{
		return;
	}

	// allocate new buffers
	h_Src = (uchar4 *)malloc(w * h * 4);

	printf("Creating GL texture...\n");
	glEnable(GL_TEXTURE_2D);
	glGenTextures(1, &gl_Tex);
	glBindTexture(GL_TEXTURE_2D, gl_Tex);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, h_Src);
	printf("Texture created.\n");

	printf("Creating PBO...\n");
	glGenBuffers(1, &gl_PBO);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, gl_PBO);
	glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, w * h * 4, h_Src, GL_STREAM_COPY);
	//While a PBO is registered to CUDA, it can't be used
	//as the destination for OpenGL drawing calls.
	//But in our particular case OpenGL is only used
	//to display the content of the PBO, specified by CUDA kernels,
	//so we need to register/unregister it only once.

	// DEPRECATED: checkCudaErrors( cudaGLRegisterBufferObject(gl_PBO) );
	checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, gl_PBO,
		cudaGraphicsMapFlagsWriteDiscard));
	printf("PBO created.\n");

	// load shader program
//	gl_Shader = compileASMShader(GL_FRAGMENT_PROGRAM_ARB, shader_code);
}


void reshapeFunc(int w, int h)
{
	glViewport(0, 0, w, h);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);

	maxgropix = 16;/*un espece de rapport entre les tailles plutot*/
	newpic();
	createBuffers(w, h);
	imageDim.x = w;
	imageDim.y = h;

}

void initGL(int *argc, char **argv)
{
	printf("Initializing GLUT...\n");
	glutInit(argc, argv);

	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
	glutInitWindowSize(imageDim.x, imageDim.y);
	glutInitWindowPosition(0, 0);
	glutCreateWindow(argv[0]);

	glutDisplayFunc(displayFunc);
	glutKeyboardFunc(keyboardFunc);
	glutMouseFunc(mouseFunc);
	glutMotionFunc(motionFunc);
	glutReshapeFunc(reshapeFunc);
	glutTimerFunc(REFRESH_DELAY, timerEvent, 0);
	initMenus();

	printf("Loading extensions: %s\n", glewGetErrorString(glewInit()));

	if (!glewIsSupported("GL_VERSION_1_5 GL_ARB_vertex_buffer_object GL_ARB_pixel_buffer_object"))
	{
		fprintf(stderr, "Error: failed to get minimal extensions for demo\n");
		fprintf(stderr, "This sample requires:\n");
		fprintf(stderr, "  OpenGL version 1.5\n");
		fprintf(stderr, "  GL_ARB_vertex_buffer_object\n");
		fprintf(stderr, "  GL_ARB_pixel_buffer_object\n");
		exit(EXIT_SUCCESS);
	}

	printf("OpenGL window created.\n");
}

void initData(int argc, char **argv)
{
	srand(time(NULL));
	viewParams.angle = { 0.0, 0.0, 0.0, 0.0 };
	viewParams.vangle = { 0.0, 0.0, 0.0, 0.0 };
	viewParams.Off =   { (rand() % 1000) / 1000.0 - 0.5, (rand() % 1000) / 1000.0 - 0.5, (rand() % 1000) / 1000.0 - 0.5, (rand() % 1000) / 1000.0 - 0.5 };
	viewParams.JSOff = { (rand() % 1000) / 1000.0 - 0.5, (rand() % 1000) / 1000.0 - 0.5, (rand() % 1000) / 1000.0 - 0.5, (rand() % 1000) / 1000.0 - 0.5 };//{ -0.5, 0.1, 0.0, 0.0 };
	viewParams.OriJSOff = { 0.0, 0.0, 0.0, 0.0 };
	viewParams.DesJSOff = { 0.0, 0.0, 0.0, 0.0 };


	// check for hardware double precision support
	int dev = 0;
	dev = findCudaDevice(argc, (const char **)argv);

	cudaDeviceProp deviceProp;
	checkCudaErrors(cudaGetDeviceProperties(&deviceProp, dev));

	numSMs = deviceProp.multiProcessorCount;

	if (checkCmdLineFlag(argc, (const char **)argv, "scale"))
	{
		viewParams.scale = getCmdLineArgumentFloat(argc, (const char **)argv, "xOff");
	}

	colors.w = 0;
	colors.x = 3;
	colors.y = 5;
	colors.z = 7;
	printf("Data initialization done.\n");
}

////////////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
	//::MessageBox(NULL,"pouet","coucou",MB_OK);

	// use command-line specified CUDA device, otherwise use device with highest Gflops/s
	/*if( cutCheckCmdLineFlag(argc, (const char**)argv, "device") )
	cutilDeviceInit(argc, argv);
	else
	cudaSetDevice( cutGetMaxGflopsDeviceId() );

	// check for hardware double precision support
	int dev = 0;
	cutGetCmdLineArgumenti(argc, (const char **) argv, "device", &dev);

	cudaDeviceProp deviceProp;
	cutilSafeCall(cudaGetDeviceProperties(&deviceProp, dev);
	printf("Compute capability %d.%d\n", deviceProp.major, deviceProp.minor);
	int version = deviceProp.major*10 + deviceProp.minor;
	//    haveDoubles = (version >= 13);
	//    if (inEmulationMode()) {
	// workaround since SM13 kernel doesn't produce correct output in emulation mode
	//        haveDoubles = false;
	//    }
	numSMs = deviceProp.multiProcessorCount;

	// parse command line arguments
	bool bQAReadback = false;
	bool bFBODisplay = false;

	imageDim.x = 1152;
	imageDim.y = 720;
	//	imageDim.x = 1920;
	//	imageDim.y = 1200;

	colors.w = 0;
	colors.x = 3;
	colors.y = 5;
	colors.z = 7;
	printf("Data init done.\n");

	printf("Initializing GLUT...\n");
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
	glutInitWindowSize(imageDim.x, imageDim.y);
	glutInitWindowPosition(0, 0);
	glutCreateWindow(argv[0]);

	printf("Loading extensions: %s\n", glewGetErrorString(glewInit()));
	/*if (bFBODisplay) {
	if (!glewIsSupported( "GL_VERSION_2_0 GL_ARB_fragment_program GL_EXT_framebuffer_object" )) {
	fprintf(stderr, "Error: failed to get minimal extensions for demo\n");
	fprintf(stderr, "This sample requires:\n");
	fprintf(stderr, "  OpenGL version 2.0\n");
	fprintf(stderr, "  GL_ARB_fragment_program\n");
	fprintf(stderr, "  GL_EXT_framebuffer_object\n");
	cleanup();
	exit(-1);
	}
	} else {
	if (!glewIsSupported( "GL_VERSION_1_5 GL_ARB_vertex_buffer_object GL_ARB_pixel_buffer_object" )) {
	fprintf(stderr, "Error: failed to get minimal extensions for demo\n");
	fprintf(stderr, "This sample requires:\n");
	fprintf(stderr, "  OpenGL version 1.5\n");
	fprintf(stderr, "  GL_ARB_vertex_buffer_object\n");
	fprintf(stderr, "  GL_ARB_pixel_buffer_object\n");
	cleanup();
	exit(-1);
	}
	}*/
	printf("OpenGL window created.\n");

	// Creating the Auto-Validation Code
	/*    if (bQAReadback) {
	if (bFBODisplay) {
	g_CheckRender = new CheckFBO(imageDim.x, imageDim.y, 4);
	} else {
	g_CheckRender = new CheckBackBuffer(imageDim.x, imageDim.y, 4);
	}
	g_CheckRender->setPixelFormat(GL_RGBA);
	g_CheckRender->setExecPath(argv[0]);
	g_CheckRender->EnableQAReadback(true);
	}*/

	// use command-line specified CUDA device, otherwise use device with highest Gflops/s
	findCudaGLDevice(argc, (const char **) argv);

	// If the GPU does not meet SM1.1 capabilities, we quit
	if (!checkCudaCapabilities(1, 1))
	{

		// cudaDeviceReset causes the driver to clean up all state. While
		// not mandatory in normal operation, it is good practice.  It is also
		// needed to ensure correct operation when the application is being
		// profiled. Calling cudaDeviceReset causes all profile data to be
		// flushed before the application exits
		cudaDeviceReset();
		exit(EXIT_SUCCESS);
	}

	// Otherwise it succeeds, we will continue to run this sample
	initData(argc, argv);

	// Initialize OpenGL context first before the CUDA context is created.  This is needed
	// to achieve optimal performance with OpenGL/CUDA interop.
	initGL(&argc, argv);
	initOpenGLBuffers(imageDim.x, imageDim.y);

	/*	glutDisplayFunc(displayFunc);
	glutIdleFunc(idleFunc);
	glutKeyboardFunc(keyboardFunc);
	glutMouseFunc(mouseFunc);
	glutMotionFunc(motionFunc);
	glutReshapeFunc(reshapeFunc);*/
	initMenus();

	printf("Starting GLUT main loop...\n");
	printf("\n");

	// timer pour fps
	/*cutilCheckError(*/sdkCreateTimer(&hTimer);
	/*cutilCheckError(*/sdkStartTimer(&hTimer);
	// timer pour total time
	/*cutilCheckError(*/sdkCreateTimer(&hETATimer);
	/*cutilCheckError(*/sdkStartTimer(&hETATimer);

	glutCloseFunc(cleanup);

	glutMainLoop();

	//cudaThreadExit();

	cudaDeviceReset();
	//cutilExit(argc, argv);
} // main
