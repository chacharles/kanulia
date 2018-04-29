/*
    Mandelbrot sample
    submitted by Mark Granger, NewTek

    CUDA 2.0 SDK - updated with double precision support
    CUDA 2.1 SDK - updated to demonstrate software block scheduling using atomics
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "windows.h"
#include <GL/glew.h>
#include "KanParam.h"

#if defined(__APPLE__) || defined(MACOSX)
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif

#include <cuda_runtime_api.h>
#include <cutil_inline.h>
#include <cuda_gl_interop.h>
/*#include <rendercheck_gl.h>*/

#include "kanulia.h"

using namespace System;
using namespace System::ComponentModel;
using namespace System::Collections;
using namespace System::Windows::Forms;
using namespace System::Data;
using namespace System::Drawing;

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

//OpenGL PBO and texture "names"
GLuint gl_PBO, gl_Tex;

//Source image on the host side
uchar4 *h_Src = 0;

//Original image width and height
int imageW, imageH;

// Starting iteration limit
unsigned int crunch = 10;

// Start mandel selection 1/julia heigh of windows
int julia = 4;

// Start with
int julia4D = JULIA4D;
// Angle for julia4D view
float4 angle = {0.0,0.0,0.0,0.0};
/*float anglexw = 0.;
float angleyw = 0.;
float anglexy = 0.;
float anglexz = 0.;*/
float4 vangle = {0.0,0.0,0.0,0.0};
/*float vanglexw = 0.;
float vangleyw = 0.;
float vanglexy = 0.;
float vanglexz = 0.;*/

// Starting position and scale
float4 Off = {-0.5,0.0,0.0,0.0};
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
float4 JSOff = {-0.5,0.1,0.0,0.0};

// Origine, Destination and step for julia seed move
float4 OriJSOff = {0.0,0.0,0.0,0.0};
float4 DesJSOff = {0.0,0.0,0.0,0.0};
double StepOffre = 1.;
double StepOffim = 1.;

// Affichage bloc par bloc
unsigned int bloc=0;
unsigned int nbloc=1;
unsigned int sqrnb=1;

// Affichage pixélisé de la julia
unsigned int maxgropix = 16; // taille du bloc maximal pixelisé de calcul rapide
unsigned int gropix = maxgropix; // taille du gros bloc pixelisé courant

// Starting stationary position and scale motion for Julia
double xJdOff = 0.0;
double yJdOff = 0.0;
double dscaleJ = 1.0;

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
unsigned int hTimer; // pour l'affichage des fps
unsigned int hETATimer; // pour l'affichage du temps restant a calculer
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

void openHelp()
{
	System::Windows::Forms::MessageBox::Show("Hello, Windows Forms");

//	Application::Run(gcnew KanParam());
//	ShellExecute(0, "open", "http://code.google.com/p/kanulia/wiki/Control", 0, 0, 1);
/*	SHELLEXECUTEINFO	shellInfo = { 0 };

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
	nbloc=1;
	sqrnb=1;
    cutResetTimer(hETATimer);
	totalETAtime = 0.f;
}

void computeFPS()
{
    frameCount++;
    fpsCount++;
    if (fpsCount == fpsLimit-1) {
        g_Verify = true;
    }
    if (fpsCount == fpsLimit) {
        char fps[256];
        float ifps = 1.f / (cutGetTimerValue(hTimer) / 1000.f);
		float ETAtimer = cutGetTimerValue(hETATimer) /1000.f;

		if ( pass < maxpass )
		{
			if (pass==0)
			{

				if (bloc == 0)
					sprintf(fps, "%Kanulia START %3.1f fps", ifps);
				else
				{
					float timeleft = (ETAtimer)*(float)(maxgropix*maxgropix)/(float)(bloc);
					float tottimeleft = timeleft*maxpass - ETAtimer;
					timeleft -= ETAtimer;

					int hh = (int)(tottimeleft)/3600;
					int mm = ((int)(tottimeleft)/60)%60;
					int ss = (int)(tottimeleft)%60;
					
					int mm2 = (int)(timeleft)/60;
					int ss2 = (int)(timeleft)%60;
					sprintf(fps, "%Kanulia %02d:%02d -- %02d:%02d:%02d -- %3.1f fps", mm2, ss2, hh, mm, ss, ifps);
				}
			}
			else
			{
				if (bloc==0)
					totalETAtime = (ETAtimer)*(float)(maxpass)/(float)(pass);
				float timeleft = totalETAtime - ETAtimer;
				int hh = (int)(timeleft)/3600;
				int mm = ((int)(timeleft)/60)%60;
				int ss = (int)(timeleft)%60;
				float perc = (ETAtimer/totalETAtime)*100.f;
				sprintf(fps, "%Kanulia %04.2f%% -- %02d:%02d:%02d -- %3.1f fps", perc, hh,mm,ss, ifps);
			}
		}
		else
			sprintf(fps, "%Kanulia DONE %3.1f fps", ifps);
 //         ((g_CheckRender && g_CheckRender->IsQAReadback()) ? "AutoTest: " : ""), ifps);

        glutSetWindowTitle(fps);
        fpsCount = 0;
//        if (g_CheckRender && !g_CheckRender->IsQAReadback()) fpsLimit = (int)MAX(ifps, 1.f);

        cutilCheckError(cutResetTimer(hTimer));

//        AutoQATest();
    }
}

// Get a sub-pixel sample location
void GetSample(int sampleIndex, float &x, float &y)
{
    static const unsigned char pairData[128][2] = {
        { 64,  64}, {  0,   0}, {  1,  63}, { 63,   1}, { 96,  32}, { 97,  95}, { 36,  96}, { 30,  31},
        { 95, 127}, {  4,  97}, { 33,  62}, { 62,  33}, { 31, 126}, { 67,  99}, { 99,  65}, {  2,  34},
        { 81,  49}, { 19,  80}, {113,  17}, {112, 112}, { 80,  16}, {115,  81}, { 46,  15}, { 82,  79},
        { 48,  78}, { 16,  14}, { 49, 113}, {114,  48}, { 45,  45}, { 18,  47}, { 20, 109}, { 79, 115},
        { 65,  82}, { 52,  94}, { 15, 124}, { 94, 111}, { 61,  18}, { 47,  30}, { 83, 100}, { 98,  50},
        {110,   2}, {117,  98}, { 50,  59}, { 77,  35}, {  3, 114}, {  5,  77}, { 17,  66}, { 32,  13},
        {127,  20}, { 34,  76}, { 35, 110}, {100,  12}, {116,  67}, { 66,  46}, { 14,  28}, { 23,  93},
        {102,  83}, { 86,  61}, { 44, 125}, { 76,   3}, {109,  36}, {  6,  51}, { 75,  89}, { 91,  21},
        { 60, 117}, { 29,  43}, {119,  29}, { 74,  70}, {126,  87}, { 93,  75}, { 71,  24}, {106, 102},
        {108,  58}, { 89,   9}, {103,  23}, { 72,  56}, {120,   8}, { 88,  40}, { 11,  88}, {104, 120},
        { 57, 105}, {118, 122}, { 53,   6}, {125,  44}, { 43,  68}, { 58,  73}, { 24,  22}, { 22,   5},
        { 40,  86}, {122, 108}, { 87,  90}, { 56,  42}, { 70, 121}, {  8,   7}, { 37,  52}, { 25,  55},
        { 69,  11}, { 10, 106}, { 12,  38}, { 26,  69}, { 27, 116}, { 38,  25}, { 59,  54}, {107,  72},
        {121,  57}, { 39,  37}, { 73, 107}, { 85, 123}, { 28, 103}, {123,  74}, { 55,  85}, {101,  41},
        { 42, 104}, { 84,  27}, {111,  91}, {  9,  19}, { 21,  39}, { 90,  53}, { 41,  60}, { 54,  26},
        { 92, 119}, { 51,  71}, {124, 101}, { 68,  92}, { 78,  10}, { 13, 118}, {  7,  84}, {105,   4}
    };

    x = (1.0f / (float)maxpass) * (0.5f + (float)pairData[sampleIndex][0]);
    y = (1.0f / (float)maxpass) * (0.5f + (float)pairData[sampleIndex][1]);
} // GetSample

void rotate4(float4 *p, const float4 angle)
{
	float t;
	if (angle.x != 0. ) {
		t  =   (*p).y * cos(angle.x) + (*p).z * sin(angle.x);
		(*p).z = - (*p).y * sin(angle.x) + (*p).z * cos(angle.x);
		(*p).y = t;
	};
	if (angle.y != 0. ) {
		t   =   (*p).x * cos(angle.y) + (*p).z * sin(angle.y);
		(*p).z = - (*p).x * sin(angle.y) + (*p).z * cos(angle.y);
		(*p).x = t;
	};
	if (angle.z != 0. ) {
		t   =   (*p).z * cos(angle.z) + (*p).w * sin(angle.z);
		(*p).w = - (*p).z * sin(angle.z) + (*p).w * cos(angle.z);
		(*p).z = t;
	};
	if (angle.w != 0. ) {
		t   =   (*p).y * cos(angle.w) + (*p).w * sin(angle.w);
		(*p).w = - (*p).y * sin(angle.w) + (*p).w * cos(angle.w);
		(*p).y = t;
	};
}

// OpenGL display function
void displayFunc(void)
{
	if (StepOffre < 1.)
	{
		JSOff.x = OriJSOff.x + (DesJSOff.x - OriJSOff.x)*StepOffre;
		JSOff.y = OriJSOff.y + (DesJSOff.y - OriJSOff.y)*StepOffre;
		
		StepOffre += 0.003;
//		printf("StepOffre\n");
		newpic();
	}
	if (StepOffim < 1.)
	{
		JSOff.z = OriJSOff.z + (DesJSOff.z - OriJSOff.z)*StepOffim;
		JSOff.w = OriJSOff.w + (DesJSOff.w - OriJSOff.w)*StepOffim;
		
		StepOffim += 0.003;
//		printf("StepOffim\n");
		newpic();
	}
	if (vangle.x != 0.)
	{
		angle.x += vangle.x;
		newpic();
	}
	if (vangle.y != 0.)
	{
		angle.y += vangle.y;
		newpic();
	}
	if (vangle.z != 0.)
	{
		angle.z += vangle.z;
		newpic();
	}
	if (vangle.w != 0.)
	{
		angle.w += vangle.w;
		newpic();
	}
    if ((xdOff != 0.0) || (ydOff != 0.0)) {
        Off.x += xdOff;
        Off.y += ydOff;
		printf("xdOff\n");
		newpic();
    }
    if (dscale != 1.0) {
        scale *= dscale;
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
        cutResetTimer(hTimer);
        cutilSafeCall(cudaGLMapBufferObject((void**)&d_dst, gl_PBO));

        // Render anti-aliasing passes until we run out time (60fps approximately)
        do {
            float xs, ys;

            // Get the anti-alias sub-pixel sample location
            GetSample(pass & 127, xs, ys);

            // Get the pixel scale and offset
            double s  = scale  / (float)imageW;
            double si = scalei / (float)imageW;
			float4 Pos={ (xs - (float)imageW * 0.5f) * s  + Off.x,
						 (ys - (float)imageH * 0.5f) * s  + Off.y,
						 (xs - (float)imageW * 0.5f) * si + Off.z,
						 (ys - (float)imageH * 0.5f) * si + Off.w
						};
            // same for Julia
			double sj = scaleJ / (float)imageW;
			double xj;
			double yj;
			if ( julia4D == JULIA2D ) // blury in 2D mode
			{
				xj = (xs - (double)imageW * 0.5f) * sj + xJOff;
				yj = (ys - (double)imageH * 0.5f) * sj + yJOff;
			}
			else // but not if in 4D mode
			{
				xj = (0.5f - (double)imageW * 0.5f) * sj + xJOff;
				yj = (0.5f - (double)imageH * 0.5f) * sj + yJOff;
			}
			
			int rebloc = 0;
			int bbloc = bloc;
			for (unsigned int mx = sqrnb/2;mx >= 1; mx/=2)
			{
				switch ( bbloc % 4 )
				{
		/*			case 0:
						rebloc += 0;
						break;*/
					case 1:
						rebloc += (sqrnb*mx)+mx; // une ligne taille mx et un mx pixel
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
					imageW, imageH, // windows size
					Pos, JSOff, angle, s, si, xj, yj, sj, // seed point and scale for mandel brod and julia
					( 2.0 * xs ) - 1.0, ( 2.0 * ys ) - 1.0 , // blur modification
					maxgropix, gropix, rebloc, crunch,
					colors, pass, // color palette, and pass number
					animationFrame,
					numSMs, julia, julia4D);
            cudaThreadSynchronize();

            // Estimate the total time of the frame if one more pass is rendered
            timeEstimate = 0.001f * cutGetTimerValue(hTimer) * ((float)(pass + 1 - startPass) / (float)((pass - startPass)?(pass-startPass):1));
			printf("startpass=%d pass=%d M=%d gropix=%d blc= %d rblc=%d nblc=%d Estimate=%5.8f\n",startPass,pass,maxgropix,gropix,bloc,rebloc,nbloc,timeEstimate);

			// ajustage du maxgropix en fonction du temps mis pour calculer
			if (gropix==maxgropix) // on est dans la pass la plus grossiere
			{
				if ((maxgropix>  1)&&(timeEstimate<1./30.))
				{
					maxgropix /= 2;
					newpic();
				}
				if ((maxgropix<16)&&(timeEstimate>1./8.))
				{
					maxgropix *= 2;
					newpic();
				}
			}
			
			bloc++;
			if (bloc==nbloc)
			{
				// si on viens de terminer une passe au niveau le plus fin, on incremente
				if ( gropix == 1 ) pass++;
				
				if (gropix>1) gropix/=2;
				
				bloc=0;

				// On calcul le nombre de bloc
				sqrnb=maxgropix/gropix;
				nbloc=sqrnb*sqrnb;
//				nbloc=1;
//				for (unsigned int mx=maxgropix;mx>gropix;mx/=2) nbloc*=4;
				// si on est dans la 1ere image, on éviteles cases deja affiché
				if (pass == 0) bloc = (nbloc/4);
			}
			

        } while ((pass < maxpass) && (timeEstimate < 1.0f / 60.0f) && !RUN_TIMING);
        cutilSafeCall(cudaGLUnmapBufferObject(gl_PBO));
#if RUN_TIMING
        printf("GPU = %5.8f\n", 0.001f * cutGetTimerValue(hTimer));
#endif
    }

    // display image
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, imageW, imageH, GL_RGBA, GL_UNSIGNED_BYTE, BUFFER_DATA(0));

    glBegin(GL_QUADS);
    glTexCoord2f(0.0f, 0.0f); glVertex2f(0.0f, 0.0f);
    glTexCoord2f(1.0f, 0.0f); glVertex2f(1.0f, 0.0f);
    glTexCoord2f(1.0f, 1.0f); glVertex2f(1.0f, 1.0f);
    glTexCoord2f(0.0f, 1.0f); glVertex2f(0.0f, 1.0f);
    glEnd();

 /*   if (g_CheckRender && g_CheckRender->IsQAReadback() && g_Verify) {
        printf("> (Frame %d) Readback BackBuffer\n", frameCount);
        g_CheckRender->readback( imageW, imageH, (GLuint)NULL );
        g_CheckRender->savePPM ( sOriginal[0], true, NULL);
        if (!g_CheckRender->PPMvsPPM(sOriginal[0], sReference[haveDoubles], MAX_EPSILON)) {
            g_TotalErrors++;
        }
        g_Verify = false;
        g_AutoQuit = true;
    }*/

    glutSwapBuffers();

    computeFPS();
} // displayFunc

void cleanup()
{
    glDeleteBuffers(1, &gl_PBO);
    glDeleteTextures(1, &gl_Tex);

    if (h_Src) {
        free(h_Src);
        h_Src = 0;
    }

//    if (g_CheckRender) {
//        delete g_CheckRender; g_CheckRender = NULL;
//    }
}


// OpenGL keyboard function
void keyboardFunc(unsigned char k, int, int)
{
    int seed;
    switch (k){
        case '\033':
        case 'q':
        case 'Q':
            printf("Shutting down...\n");
            cutilCheckError(cutStopTimer(hTimer) );
            cutilCheckError(cutDeleteTimer(hTimer));
            cutilCheckError(cutStopTimer(hETATimer) );
            cutilCheckError(cutDeleteTimer(hETATimer));
            cutilSafeCall(cudaGLUnregisterBufferObject(gl_PBO));
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
            printf("Shutdown done.\n");
            exit(0);
            break;

        case '?':
		case 'h':
		case 'H':
            printf("Off.x = %5.8f\n", Off.x);
            printf("Off.y = %5.8f\n", Off.y);
            printf("scale = %e\n", scale);
            printf("detail = %d\n", crunch);
            printf("color = %d\n", colorSeed);
            printf("\n");
			openHelp();
            break;

        case 'r': case 'R':
            // Reset all values to their defaults
            Off.x = -0.5;
            Off.y = 0.0;
            scale = 3.2;
            xdOff = 0.0;
            ydOff = 0.0;
            dscale = 1.0;
            colorSeed = 0;
            colors.x = 3;
            colors.y = 5;
            colors.z = 7;
            crunch = 10;
            animationFrame = 0;
            animationStep = 0;
			newpic();
			angle.x = 0.;
			angle.y = 0.;
			angle.z = 0.;
			angle.w = 0.;
			vangle.x = 0.;
			vangle.y = 0.;
			vangle.z = 0.;
			vangle.w = 0.;
            break;

        case 'c':
            seed = ++colorSeed;
            if (seed) {
                colors.x = RANDOMBITS(seed, 4);
                colors.y = RANDOMBITS(seed, 4);
                colors.z = RANDOMBITS(seed, 4);
            } else {
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
            } else {
                colors.x = 3;
                colors.y = 5;
                colors.z = 7;
            }
			newpic();
            break;

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
				if ( crunch < 32 ) crunch += 1;
                else crunch *= 2;
				newpic();
            }
            printf("detail = %d\n", crunch);
            break;

        case 'D':
            if (crunch > 2) {
				if ( crunch <= 32 ) crunch -= 1;
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
			Off.x -= 0.05f * scale;
			newpic();
		    break;

        case '8':	// Up arrow key
			Off.y += 0.05f * scale;
			newpic();
		    break;

        case '6':	// Right arrow key
			Off.x += 0.05f * scale;
			newpic();
		    break;

        case '2':	// Down arrow key
			Off.y -= 0.05f * scale;
			newpic();
		    break;

        case '+':
			scale /= 1.1f;
			newpic();
		    break;

        case '-':
			scale *= 1.1f;
			newpic();
		    break;

		case 'u':
			vangle.x += 0.001;
			break;
		case 'i':
			vangle.y += 0.001;
			break;
		case 'o':
			vangle.z += 0.001;
			break;
		case 'p':
			vangle.w += 0.001;
			break;
		case 'U':
			vangle.x -= 0.001;
			break;
		case 'I':
			vangle.y -= 0.001;
			break;
		case 'O':
			vangle.z -= 0.001;
			break;
		case 'P':
			vangle.w -= 0.001;
			break;
		case 'w':
			imageW = 2560;
			imageH = 1600;
			
			glutReshapeWindow(imageW,imageH);
			
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
		if ( (julia < 32) && (x < imageW / julia) && (y  > imageH - imageH / julia))
		{
			// real seed point
			StepOffre = 1.;
			JSOff.x = Off.x + ( x - (double) ( imageW / julia ) / 2. ) * ( scale / (double) (imageW / julia) );
			JSOff.y = Off.y - ( y - (double) ( imageH - imageH / (2 * julia) ) ) * ( scale / (double) (imageW / julia) );
			
			newpic();
		};
		if ( (julia < 32) && ((imageW - x) < imageW / julia) && (y  > imageH - imageH / julia))
		{
			// imaginary seed point
			StepOffim = 1.;
			JSOff.z = Off.z + ( (imageW - x) - (double) ( imageW / julia ) / 2. )           * ( scalei / (double) (imageW / julia) );
			JSOff.w = Off.w - (           y  - (double) ( imageH - imageH / (2 * julia) ) ) * ( scalei / (double) (imageW / julia) );

			newpic();
		};
	}
	// middle mouse button
	if (button == 1) {
		// in the mandelbro select button released
	    if ((state == GLUT_UP) && ( (julia < 32) && (x < imageW / julia) && (y  > imageH - imageH / julia))) {
			// printf("Middle Clicked %3.2f %3.2f \n" , ( x - (double) ( imageH ) / 2. ) , ( y - (double) ( imageH ) / 2. ) );
			
			OriJSOff.x = JSOff.x;
			OriJSOff.y = JSOff.y;
			DesJSOff.x = Off.x + ( x - (double) ( imageW / julia ) / 2. )           * ( scale / (double) (imageW / julia) );
			DesJSOff.y = Off.y - ( y - (double) ( imageH - imageH / (2 * julia) ) ) * ( scale / (double) (imageW / julia) );
			StepOffre = 0.;

			newpic();
		}
		// in the mandelbro select button released
	    if ((state == GLUT_UP) && ( (julia < 32) && ((imageW - x) < imageW / julia) && (y  > imageH - imageH / julia))) {
			// printf("Middle Clicked %3.2f %3.2f \n" , ( x - (double) ( imageH ) / 2. ) , ( y - (double) ( imageH ) / 2. ) );
			
			OriJSOff.z = JSOff.z;
			OriJSOff.w = JSOff.w;
			DesJSOff.z = Off.z + ( (imageW - x) - (double) ( imageW / julia ) / 2. )           * ( scalei / (double) (imageW / julia) );
			DesJSOff.w = Off.w - (           y  - (double) ( imageH - imageH / (2 * julia) ) ) * ( scalei / (double) (imageW / julia) );
			StepOffim = 0.;
			
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
		if ( button == GLUT_WHEEL_UP )
		{
			if ( (julia < 32) && (x < imageW / julia) && (y  > imageH - imageH / julia))
			{
				scale /= 1.1f;
				Off.x += ( x - (double) ( imageW / julia ) / 2. ) * 0.1 * ( scale / (double) (imageW / julia) );
				Off.y -= ( y - (double) ( imageH - imageH / (2* julia) ) ) * 0.1 * ( scale / (double) (imageW / julia) );
			} else if ( (julia < 32) && ((imageW - x) < imageW / julia) && (y  > imageH - imageH / julia))
			{
				scalei /= 1.1f;
				Off.z += ( (imageW - x) - (double) ( imageW / julia ) / 2. )          * 0.1 * ( scalei / (double) (imageW / julia) );
				Off.w -= (           y  - (double) ( imageH - imageH / (2* julia) ) ) * 0.1 * ( scalei / (double) (imageW / julia) );
			} else {
				scaleJ /= 1.1f;
				xJOff += ( x - (double) ( imageW ) / 2. ) * 0.1 * ( scaleJ / (double) imageW );
				yJOff -= ( y - (double) ( imageH ) / 2. ) * 0.1 * ( scaleJ / (double) imageW );
			};
			newpic();
			printf("Wheel Up\n");
		}
		else if( button == GLUT_WHEEL_DOWN )
		{
			if ( (julia < 32) && (x < imageW / julia) && (y  > imageH - imageH / julia))
			{
				Off.x -= ( x - (double) ( imageW / julia ) / 2. ) * 0.1 * ( scale / (double) (imageW / julia) );
				Off.y += ( y - (double) ( imageH - imageH / (2 * julia) ) ) * 0.1 * ( scale / (double) (imageW / julia) );
				scale *= 1.1f;
			} else if ( (julia < 32) && ((imageW - x) < imageW / julia) && (y  > imageH - imageH / julia))
			{
				Off.z -= ( (imageW - x) - (double) ( imageW / julia ) / 2. )           * 0.1 * ( scalei / (double) (imageW / julia) );
				Off.w += (           y  - (double) ( imageH - imageH / (2 * julia) ) ) * 0.1 * ( scalei / (double) (imageW / julia) );
				scalei *= 1.1f;
			} else {
				xJOff -= ( x - (double) ( imageW ) / 2. ) * 0.1 * ( scaleJ / (double) imageW );
				yJOff += ( y - (double) ( imageH ) / 2. ) * 0.1 * ( scaleJ / (double) imageW );
				scaleJ *= 1.1f;
			};
			newpic();
			printf("Wheel Down\n");
		}
	}

    lastx = x;
    lasty = y;
    xdOff = 0.0;
    ydOff = 0.0;
    dscale = 1.0;
} // clickFunc

// OpenGL mouse motion function
void motionFunc(int x, int y)
{
    double fx = (double)((lastx - x) / 10.0) / (double)(imageW);
    double fy = (double)((y - lasty) / 10.0) / (double)(imageW);
	
//    int modifiers = glutGetModifiers();

    if (leftClicked) {
		if ( (julia < 32) && (x < imageW / julia) && (y  > imageH - imageH / julia))
		{
			JSOff.x = Off.x + ( x - (double) ( imageW / julia ) / 2.0 ) * ( scale / (double) (imageW / julia) );
			JSOff.y = Off.y - ( y - (double) ( imageH - imageH / (2.0 * julia) ) ) * ( scale / (double) (imageW / julia) );
			newpic();
		} else if ( (julia < 32) && ((imageW - x) < imageW / julia) && (y  > imageH - imageH / julia))
		{
			JSOff.z = Off.z + ( (imageW - x) - (double) ( imageW / julia ) / 2.0 ) * ( scalei / (double) (imageW / julia) );
			JSOff.w = Off.w - ( y - (double) ( imageH - imageH / (2.0 * julia) ) ) * ( scalei / (double) (imageW / julia) );
			newpic();
		} else
		{
			if (leftClicked && (modifiers & GLUT_ACTIVE_SHIFT))
			{
				angle.z -= fx*100.0;
				angle.w -= fy*100.0;
				
	//			printf("Motion fx=%f fy=%f\n",fx,fy);
	//			printf("angle.z=%f angle.w=%f\n",angle.z,angle.w);
			}
			else
			{
				angle.y -= fx*100.0;
				angle.x -= fy*100.0;
			//	float4 delangle = {0.0,0.0,-fx*100.0,-fy*100.0};
			//	float4 delangle1st = {0.0,-fy*100.0,0.0,0.0};
			//	rotate4(&angle,delangle);
/*				angle.x += delangle.x;
				angle.y += delangle.y;
				angle.z += delangle.z;
				angle.w += delangle.w;*/
				printf("Motion fx=%f fy=%f\n",fx,fy);
				printf("angle.y=%f angle.x=%f\n",angle.y,angle.x);
			}
			newpic();

			};
    } else {
        xdOff = 0.0f;
        ydOff = 0.0f;
    }

    if (middleClicked)
		if ( (julia < 32) && (x < imageW / julia) && (y  > imageH - imageH / julia))
		{
			if (fy > 0.0f) {
				dscale = 1.0 - fy;
				dscale = dscale < 1.05 ? dscale : 1.05;
			} else {
				dscale = 1.0 / (1.0 + fy);
				dscale = dscale > (1.0 / 1.05) ? dscale : (1.0 / 1.05);
			}
		} else {
			angle.z -= fx*100.0;
			angle.w -= fy*100.0;
			newpic();
//			printf("Motion fx=%f fy=%f\n",fx,fy);
//			printf("angle.z=%f angle.w=%f\n",angle.z,angle.w);

		}
    else
        dscale = 1.0;

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
			angle.x = 0.;
			angle.y = 0.;
			angle.z = 0.;
			angle.w = 0.;
            crunch = 16;
			julia4D = CLOUDJULIA; //"Cloudy Julia4D"
			break;
		case 5:
			angle.x = 0.;
			angle.y = 0.;
			angle.z = 0.;
			angle.w = 0.;
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
            } else {
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
            } else {
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
	glutAddSubMenu("Julia",juliamenu);
	glutAddSubMenu("Color",colormenu);
	glutAddMenuEntry("Help", 10);
    glutAttachMenu(GLUT_RIGHT_BUTTON);
}

void createBuffers(int w, int h)
{
    if (h_Src) {
        free(h_Src);
        h_Src = 0;
    }
	h_Src = (uchar4*)malloc(w * h * 4);

    if (gl_Tex) {
        glDeleteTextures(1, &gl_Tex);
        gl_Tex = 0;
    }
    if (gl_PBO) {
        cudaGLUnregisterBufferObject(gl_PBO);
        glDeleteBuffers(1, &gl_PBO);
        gl_PBO = 0;
    }
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
        cutilSafeCall( cudaGLRegisterBufferObject(gl_PBO) );
    printf("PBO created.\n");

    // This is the buffer we use to readback results into
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
    imageW = w;
    imageH = h;

}

////////////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
	//::MessageBox(NULL,"pouet","coucou",MB_OK);

    // use command-line specified CUDA device, otherwise use device with highest Gflops/s
    if( cutCheckCmdLineFlag(argc, (const char**)argv, "device") )
        cutilDeviceInit(argc, argv);
    else
        cudaSetDevice( cutGetMaxGflopsDeviceId() );

    // check for hardware double precision support
    int dev = 0;
    cutGetCmdLineArgumenti(argc, (const char **) argv, "device", &dev);

    cudaDeviceProp deviceProp;
    cutilSafeCall(cudaGetDeviceProperties(&deviceProp, dev));
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

    if (argc > 1) {
        if (cutCheckCmdLineFlag(argc, (const char **)argv, "qatest")) {
            bQAReadback = true;
            fpsLimit = frameCheckNumber;
        }
        if (cutCheckCmdLineFlag(argc, (const char **)argv, "fbo")) {
            bFBODisplay = true;
            fpsLimit = frameCheckNumber;
        }
    }

    float x;
    if (cutGetCmdLineArgumentf(argc, (const char **)argv, "Off.x", &x)) {
        Off.x = x;
    }
    if (cutGetCmdLineArgumentf(argc, (const char **)argv, "Off.y", &x)) {
        Off.y = x;
    }
    if (cutGetCmdLineArgumentf(argc, (const char **)argv, "scale", &x)) {
        scale = x;
    }

	imageW = 1152;
	imageH = 720;
//	imageW = 1920;
//	imageH = 1200;

    colors.w = 0;
    colors.x = 3;
    colors.y = 5;
    colors.z = 7;
    printf("Data init done.\n");

    printf("Initializing GLUT...\n");
        glutInit(&argc, argv);
        glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
        glutInitWindowSize(imageW, imageH);
        glutInitWindowPosition(0, 0);
        glutCreateWindow(argv[0]);

        printf("Loading extensions: %s\n", glewGetErrorString(glewInit()));
	    if (bFBODisplay) {
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
	    }
    printf("OpenGL window created.\n");

    // Creating the Auto-Validation Code
/*    if (bQAReadback) {
        if (bFBODisplay) {
            g_CheckRender = new CheckFBO(imageW, imageH, 4);
        } else {
            g_CheckRender = new CheckBackBuffer(imageW, imageH, 4);
        }
        g_CheckRender->setPixelFormat(GL_RGBA);
        g_CheckRender->setExecPath(argv[0]);
        g_CheckRender->EnableQAReadback(true);
    }*/

    printf("Starting GLUT main loop...\n");
    printf("\n");

    glutDisplayFunc(displayFunc);
    glutIdleFunc(idleFunc);
    glutKeyboardFunc(keyboardFunc);
    glutMouseFunc(mouseFunc);
    glutMotionFunc(motionFunc);
    glutReshapeFunc(reshapeFunc);
    initMenus();

	// timer pour fps
    cutilCheckError(cutCreateTimer(&hTimer));
    cutilCheckError(cutStartTimer(hTimer));
	// timer pour total time
    cutilCheckError(cutCreateTimer(&hETATimer));
    cutilCheckError(cutStartTimer(hETATimer));

    atexit(cleanup);

    glutMainLoop();

    cudaThreadExit();

    cutilExit(argc, argv);
} // main
