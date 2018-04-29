#ifndef _KANULIACALC_CU_
#define _KANULIACALC_CU_

#include <stdio.h>
//#include "cutil_inline.h"
#include "kanulia.h"

// The dimensions of the thread block
#define BLOCKDIM_X 16
#define BLOCKDIM_Y 16

//#define ABS(n) ((n) < 0 ? -(n) : (n))
#define MAX_CRN_IN 2560

// return the argument of a complex number
__device__ inline float arg( float re, float im )
{
	float pi = 3.14159;
	float a = 0.;
	if ((re>0.)&&(im>0.)) a= atan(im/re);
	if ((re>0.)&&(im<0.)) a=2.*pi-atan(-im/re);
	if ((re<0.)&&(im>0.)) a=pi-atan(-im/re);
	if ((re<0.)&&(im<0.)) a=pi+atan(im/re);
	if ((re>0.)&&(im==0.)) a= 0.;
	if ((re==0.)&&(im>0.)) a= pi / 2.;
	if ((re<0.)&&(im==0.)) a= pi;
	if ((re==0.)&&(im<0.)) a= ( 3. * pi ) / 2.;
	return a/(2.*pi);
}

// Given H,S,L in range of 0-1
// Returns a Color (RGB struct) in range of 0-255
__device__ inline void HSL2RGB(float h, const float sl, const float ll, int *rc, int *gc, int *bc)
{
	float v,r,g,b,l = ll;
	if ( ll < 0. ) l = 0;
	if ( ll > 1. ) l = 1;
	r = l;   // default to gray
	g = l;
	b = l;
	v = (l <= 0.5) ? (l * (1.0 + sl)) : (l + sl - l * sl);
	if (v > 0)
	{
		float m;
		float sv;
		int sextant;
		float fract, vsf, mid1, mid2;

		m = l + l - v;
		sv = (v - m ) / v;
		h *= 6.0;
		sextant = (int)h;
		fract = h - sextant;
		vsf = v * sv * fract;
		mid1 = m + vsf;
		mid2 = v - vsf;
		switch (sextant)
		{
		case 0:
			r = v;
			g = mid1;
			b = m;
			break;
		case 1:
			r = mid2;
			g = v;
			b = m;
			break;
		case 2:
			r = m;
			g = v;
			b = mid1;
			break;
		case 3:
			r = m;
			g = mid2;
			b = v;
			break;
		case 4:
			r = mid1;
			g = m;
			b = v;
			break;
		case 5:
			r = v;
			g = m;
			b = mid2;
			break;
		}
	}
	*rc = r * 255.0;
	*gc = g * 255.0;
	*bc = b * 255.0;
}

// The core Mandelbrot CUDA GPU calculation function
// Unrolled version
__device__ inline int CalcMandelbrot(const float xPos, const float yPos, const int crn)
{
	float y = yPos;
	float x = xPos;
	float yy = y * y;
	float xx = x * x;
	int i = crn;

	do {
		// Iteration 1
		if (xx + yy > float(4.0))
		return i - 1;
		y = x * y * float(2.0) + yPos;
		x = xx - yy + xPos;
		yy = y * y;
		xx = x * x;

		// Iteration 2
		if (xx + yy > float(4.0))
		return i - 2;
		y = x * y * float(2.0) + yPos;
		x = xx - yy + xPos;
		yy = y * y;
		xx = x * x;

		// Iteration 3
		if (xx + yy > float(4.0))
		return i - 3;
		y = x * y * float(2.0) + yPos;
		x = xx - yy + xPos;
		yy = y * y;
		xx = x * x;

		// Iteration 4
		if (xx + yy > float(4.0))
		return i - 4;
		y = x * y * float(2.0) + yPos;
		x = xx - yy + xPos;
		yy = y * y;
		xx = x * x;

		// Iteration 5
		if (xx + yy > float(4.0))
		return i - 5;
		y = x * y * float(2.0) + yPos;
		x = xx - yy + xPos;
		yy = y * y;
		xx = x * x;

		// Iteration 6
		if (xx + yy > float(4.0))
		return i - 6;
		y = x * y * float(2.0) + yPos;
		x = xx - yy + xPos;
		yy = y * y;
		xx = x * x;

		// Iteration 7
		if (xx + yy > float(4.0))
		return i - 7;
		y = x * y * float(2.0) + yPos;
		x = xx - yy + xPos;
		yy = y * y;
		xx = x * x;

		// Iteration 8
		if (xx + yy > float(4.0))
		return i - 8;
		y = x * y * float(2.0) + yPos;
		x = xx - yy + xPos;
		yy = y * y;
		xx = x * x;

		// Iteration 9
		if (xx + yy > float(4.0))
		return i - 9;
		y = x * y * float(2.0) + yPos;
		x = xx - yy + xPos;
		yy = y * y;
		xx = x * x;

		// Iteration 10
		if (xx + yy > float(4.0))
		return i - 10;
		y = x * y * float(2.0) + yPos;
		x = xx - yy + xPos;
		yy = y * y;
		xx = x * x;

		// Iteration 11
		if (xx + yy > float(4.0))
		return i - 11;
		y = x * y * float(2.0) + yPos;
		x = xx - yy + xPos;
		yy = y * y;
		xx = x * x;

		// Iteration 12
		if (xx + yy > float(4.0))
		return i - 12;
		y = x * y * float(2.0) + yPos;
		x = xx - yy + xPos;
		yy = y * y;
		xx = x * x;

		// Iteration 13
		if (xx + yy > float(4.0))
		return i - 13;
		y = x * y * float(2.0) + yPos;
		x = xx - yy + xPos;
		yy = y * y;
		xx = x * x;

		// Iteration 14
		if (xx + yy > float(4.0))
		return i - 14;
		y = x * y * float(2.0) + yPos;
		x = xx - yy + xPos;
		yy = y * y;
		xx = x * x;

		// Iteration 15
		if (xx + yy > float(4.0))
		return i - 15;
		y = x * y * float(2.0) + yPos;
		x = xx - yy + xPos;
		yy = y * y;
		xx = x * x;

		// Iteration 16
		if (xx + yy > float(4.0))
		return i - 16;
		y = x * y * float(2.0) + yPos;
		x = xx - yy + xPos;
		yy = y * y;
		xx = x * x;

		// Iteration 17
		if (xx + yy > float(4.0))
		return i - 17;
		y = x * y * float(2.0) + yPos;
		x = xx - yy + xPos;
		yy = y * y;
		xx = x * x;

		// Iteration 18
		if (xx + yy > float(4.0))
		return i - 18;
		y = x * y * float(2.0) + yPos;
		x = xx - yy + xPos;
		yy = y * y;
		xx = x * x;

		// Iteration 19
		if (xx + yy > float(4.0))
		return i - 19;
		y = x * y * float(2.0) + yPos;
		x = xx - yy + xPos;
		yy = y * y;
		xx = x * x;

		// Iteration 20
		i -= 20;
		if ((i <= 0) || (xx + yy > float(4.0)))
		return i;
		y = x * y * float(2.0) + yPos;
		x = xx - yy + xPos;
		yy = y * y;
		xx = x * x;
	} while (1);
} // CalcMandelbrot

// The core Julia CUDA GPU calculation function

// Unrolled version
__device__ inline int CalcJulia(const float xPos, const float yPos, const float4 JS, const unsigned int crn)
{
	float y = yPos;
	float x = xPos;
	float yy = y * y;
	float xx = x * x;
	int i = crn;

	do {
		// Iteration 1
		if (xx + yy > float(4.0))
		return i - 1;
		y = x * y * float(2.0) + JS.y;
		x = xx - yy + JS.x;
		yy = y * y;
		xx = x * x;

		// Iteration 2
		if (xx + yy > float(4.0))
		return i - 2;
		y = x * y * float(2.0) + JS.y;
		x = xx - yy + JS.x;
		yy = y * y;
		xx = x * x;

		// Iteration 3
		if (xx + yy > float(4.0))
		return i - 3;
		y = x * y * float(2.0) + JS.y;
		x = xx - yy + JS.x;
		yy = y * y;
		xx = x * x;

		// Iteration 4
		if (xx + yy > float(4.0))
		return i - 4;
		y = x * y * float(2.0) + JS.y;
		x = xx - yy + JS.x;
		yy = y * y;
		xx = x * x;

		// Iteration 5
		if (xx + yy > float(4.0))
		return i - 5;
		y = x * y * float(2.0) + JS.y;
		x = xx - yy + JS.x;
		yy = y * y;
		xx = x * x;

		// Iteration 6
		if (xx + yy > float(4.0))
		return i - 6;
		y = x * y * float(2.0) + JS.y;
		x = xx - yy + JS.x;
		yy = y * y;
		xx = x * x;

		// Iteration 7
		if (xx + yy > float(4.0))
		return i - 7;
		y = x * y * float(2.0) + JS.y;
		x = xx - yy + JS.x;
		yy = y * y;
		xx = x * x;

		// Iteration 8
		if (xx + yy > float(4.0))
		return i - 8;
		y = x * y * float(2.0) + JS.y;
		x = xx - yy + JS.x;
		yy = y * y;
		xx = x * x;

		// Iteration 9
		if (xx + yy > float(4.0))
		return i - 9;
		y = x * y * float(2.0) + JS.y;
		x = xx - yy + JS.x;
		yy = y * y;
		xx = x * x;

		// Iteration 10
		if (xx + yy > float(4.0))
		return i - 10;
		y = x * y * float(2.0) + JS.y;
		x = xx - yy + JS.x;
		yy = y * y;
		xx = x * x;

		// Iteration 11
		if (xx + yy > float(4.0))
		return i - 11;
		y = x * y * float(2.0) + JS.y;
		x = xx - yy + JS.x;
		yy = y * y;
		xx = x * x;

		// Iteration 12
		if (xx + yy > float(4.0))
		return i - 12;
		y = x * y * float(2.0) + JS.y;
		x = xx - yy + JS.x;
		yy = y * y;
		xx = x * x;

		// Iteration 13
		if (xx + yy > float(4.0))
		return i - 13;
		y = x * y * float(2.0) + JS.y;
		x = xx - yy + JS.x;
		yy = y * y;
		xx = x * x;

		// Iteration 14
		if (xx + yy > float(4.0))
		return i - 14;
		y = x * y * float(2.0) + JS.y;
		x = xx - yy + JS.x;
		yy = y * y;
		xx = x * x;

		// Iteration 15
		if (xx + yy > float(4.0))
		return i - 15;
		y = x * y * float(2.0) + JS.y;
		x = xx - yy + JS.x;
		yy = y * y;
		xx = x * x;

		// Iteration 16
		if (xx + yy > float(4.0))
		return i - 16;
		y = x * y * float(2.0) + JS.y;
		x = xx - yy + JS.x;
		yy = y * y;
		xx = x * x;

		// Iteration 17
		if (xx + yy > float(4.0))
		return i - 17;
		y = x * y * float(2.0) + JS.y;
		x = xx - yy + JS.x;
		yy = y * y;
		xx = x * x;

		// Iteration 18
		if (xx + yy > float(4.0))
		return i - 18;
		y = x * y * float(2.0) + JS.y;
		x = xx - yy + JS.x;
		yy = y * y;
		xx = x * x;

		// Iteration 19
		if (xx + yy > float(4.0))
		return i - 19;
		y = x * y * float(2.0) + JS.y;
		x = xx - yy + JS.x;
		yy = y * y;
		xx = x * x;

		// Iteration 20
		i -= 20;
		if ((i <= 0) || (xx + yy > float(4.0)))
		return i;
		y = x * y * float(2.0) + JS.y;
		x = xx - yy + JS.x;
		yy = y * y;
		xx = x * x;
	} while (1);
} // CalcJulia
// The core Julia CUDA GPU calculation function

__device__ inline int CalcJulia4D(const float xPos, const float yPos, const float zPos, const float wPos, float4 JS, const unsigned int crn)
{
	float x = xPos;float y = yPos;float z = zPos;float w = wPos;
	float xx = x * x;
	float yy = y * y;
	float zz = z * z;
	float ww = w * w;
	int i = crn;

	// if (y>0) return i;
	do {
		i--;
		if (xx + yy + zz + ww > float(4.0))
		return i;
		z = x * z * float(2.0) + JS.z;
		w = x * w * float(2.0) + JS.w;
		y = x * y * float(2.0) + JS.y;
		x = xx - yy - zz - ww + JS.x;
		xx = x * x;
		yy = y * y;
		zz = z * z;
		ww = w * w;
	} while (i);
	return 0;
} // CalcJulia4D

/*__device__ inline int CalcJulia4Dstep(const float xPos, const float yPos, const float zPos, const float wPos, float4 JS, const unsigned int crn,float *step)
{
	float x = xPos;float y = yPos;float z = zPos;float w = wPos;
	float x2 = x * x;
	float y2 = y * y;
	float z2 = z * z;
	float w2 = w * w;
	int i = crn;
	float mx = 0.;
	float mz = 0.;
	float zx =1.0;float zy =0.0;float zz =0.0;float zw =0.0;
	float zzx =1.0;float zzy =0.0;float zzz =0.0;float zzw =0.0;

	// if (y>0) return i;
	do {
		i--;
		if (x2 + y2 + z2 + w2 > float(4.0))
		return i;
		z = x * z * float(2.0) + JS.z;
		w = x * w * float(2.0) + JS.w;
		y = x * y * float(2.0) + JS.y;
		x = x2 - y2 - z2 - w2 + JS.x;
		x2 = x * x;
		y2 = y * y;
		z2 = z * z;
		w2 = w * w;
		
		// Outbounding
		zzz = zx * z * float(4.0);
		zzw = zx * w * float(4.0);
		zzy = zx * y * float(4.0);
		zzx = (zx*x - zy*y - zz*z - zw*w)* float(2.0);
		zx = zzx;
		zy = zzy;
		zz = zzz;
		zw = zzw;
		mz = sqrt(zx*zx+zy*zy+zz*zz+zw*zw);
		mx = sqrt( x2+ y2+ z2+ w2);
		*step = log(mx)*float(0.5)*mx/mz;
		if (*step<RAYSTEP) *step=RAYSTEP;
	} while (i);
	return 0;
} // CalcJulia4Dstep*/

__device__ inline int CalcJulia4Dhue(const float xPos, const float yPos, const float zPos, const float wPos, float4 JS, float *hue, const unsigned int crn)
{
	float x = xPos;float y = yPos;float z = zPos;float w = wPos;
	float xx = x * x;
	float yy = y * y;
	float zz = z * z;
	float ww = w * w;

	int i = crn;
	int huenb = 7;

	if (huenb>i) huenb = i;

	do {
		i--;
		huenb--;
		if (huenb==0) *hue = arg(y-w,z+x);

		if (xx + yy + zz + ww > float(4.0))
		{
			// hue = 0.5 + cos((x+y+z+w)/4.)/2.;
			return i;
		}
		z = x * z * float(2.0) + JS.z;
		w = x * w * float(2.0) + JS.w;
		y = x * y * float(2.0) + JS.y;
		x = xx - yy - zz - ww + JS.x;
		xx = x * x;
		yy = y * y;
		zz = z * z;
		ww = w * w;
	} while (i);
	return 0;
} // CalcJulia4Dhue

/*__device__ inline int CalcMandelBox3D(const float xPos, const float yPos, const float zPos, const unsigned int crn)
{
	float x = xPos;float y = yPos;float z = zPos;
	//    float x = 0.;float y = 0.;float z = 0.;
	float xx = x*x;
	float yy = y*y;
	float zz = z*z;
	float m2 = xx+yy+zz;

	int i = crn;

	// if (y>0) return i;
	do {
		i--;
		// boxFold(x)
	
		// ballFold(r,x)
	
		float scale = 2;
		//fold box onto itself  
		if (x > 1)
		x = 2 - x;
		else if (x < -1)
		x = -2 - x;
		if (y > 1)
		y = 2 - y;
		else if (y < -1)
		y = -2 - y;
		if (z > 1)
		z = 2 - z;
		else if (z < -1)
		z = -2 - z;
		//fold sphere onto itself
		float fixedRadius = 1;
		float minRadius = 0.5;
		float length = sqrt( x*x + y*y + z*z );
		if (length < minRadius)
		{
			float fact = (fixedRadius*fixedRadius/minRadius*minRadius);
			x *= fact;
			y *= fact;
			z *= fact;
		}
		else if (length < fixedRadius)
		{
			float fact = (fixedRadius*fixedRadius/length*length);
			x *= fact;
			y *= fact;
			z *= fact;
		}
		x *= scale;
		y *= scale;
		z *= scale;

		x += xPos;
		y += yPos;
		z += zPos;

		if (x*x + y*y + z*z > float(OUTMANDELBOX))
		return i;
	} while (i);
	return 0;
} // CalcMandelbox4D*/

/*__device__ inline int CalcMandelBox3Dhue(const float xPos, const float yPos, const float zPos, float *hue, const unsigned int crn)
{
	float x = xPos;float y = yPos;float z = zPos;
	//    float x = 0.;float y = 0.;float z = 0.;

	float xx = x*x;
	float yy = y*y;
	float zz = z*z;
	float m2 = xx+yy+zz;
	int i = crn;
	int huenb = 7;

	if (huenb>i) huenb = i;

	do {
		i--;
		huenb--;
		if (huenb==0) *hue = arg(y,x);

		// boxFold(x)
		if      (x> 1.) x= 2.-x;
		else if (x<-1.) x=-2.-x;
		if      (y> 1.) y= 2.-y;
		else if (y<-1.) y=-2.-y;
		if      (z> 1.) z= 2.-z;
		else if (z<-1.) z=-2.-z;
		// ballFold(r,x)
		

		x =  x * float(2.0) + xPos;
		y =  y * float(2.0) + yPos;
		z =  z * float(2.0) + zPos;
		xx = x * x;
		yy = y * y;
		zz = z * z;
		if (xx + yy + zz > float(OUTMANDELBOX))
		{
			*hue = 0.5;
			return i;
		}
	} while (i);
	*hue = 0.9;
	return 0;
} // CalcMandelBox3Dhue*/

__device__ inline int CalcJulia4Dcore(const float xPos, const float yPos, const float zPos, const float wPos, const float4 JS, float *hue)
{
	float x = xPos;float y = yPos;float z = zPos;float w = wPos;
	float xx = x * x;
	float yy = y * y;
	float zz = z * z;
	float ww = w * w;

	int i = 0;

	do {
		i++;

		if (xx + yy + zz + ww > float(4.0))
		{
			*hue =(float)(i)/(float)(256);
			while (*hue>1.0) *hue -= 1.0;
			// if (*hue < 0.05) *hue = 0.05;
			return i;
		}
		z = x * z * float(2.0) + JS.z;
		w = x * w * float(2.0) + JS.w;
		y = x * y * float(2.0) + JS.y;
		x = xx - yy - zz - ww + JS.x;
		xx = x * x;
		yy = y * y;
		zz = z * z;
		ww = w * w;
	} while (i<=MAX_CRN_IN);
	*hue = 0.05;
	return i;
} // CalcJulia4Dcore

__device__ inline int CalcMandel4Dcore(const float xPos, const float yPos, const float zPos, const float wPos, float *hue)
{
	float x = 0.;float y = 0.;float z = 0./*JS.z*/;float w = 0./*JS.w*/;
	float xx = x * x;
	float yy = y * y;
	float zz = z * z;
	float ww = w * w;

	int i = 0;

	do {
		i++;

		if (xx + yy + zz + ww > float(4.0))
		{
			*hue =(float)(i)/256.0;
			while (*hue>1.0) *hue -= 1.0;
			return i;
		}
		z = x * z * float(2.0) + zPos;
		w = x * w * float(2.0) + wPos;
		y = x * y * float(2.0) + yPos;
		x = xx - yy - zz - ww + xPos;
		xx = x * x;
		yy = y * y;
		zz = z * z;
		ww = w * w;
	} while (i<=256);
	return i;
} // CalcMandel4Dcore


// The core Julia CUDA GPU calculation function
/*__device__ int SolidMandelBox3D(const int ix, const int iy, const float4 JS, const float4 angle, const int d_imageW, const int d_imageH, const float scaleJ,
const float xblur, const float yblur, int *r, int *g, int *b, const float xJOff, const float yJOff, const unsigned int crn)
{
	//hue color
	float hue;
	float dist = 6.0;
	float step = RAYSTEP;

	float x = ((float)ix + (xblur)) * scaleJ + xJOff;
	float y = ((float)iy + (yblur)) * scaleJ + yJOff;
	float z = - 3.0;
	float dx = sin( 0.7 * step *scaleJ* ( (float) ix  - (d_imageW/2.)) / ((float) d_imageW) );
	float dy = sin( 0.7 * step *scaleJ* ( (float) iy  - (d_imageH/2.)) / ((float) d_imageW) );
	float dz = step;
	rotate3(&x,&y,&z,angle);
	rotate3(&dx,&dy,&dz,angle);
	int nb = (dist/step);

	float x0 = 0.0;float y0 = -1.0;float z0 = 0.0;// normal is the secant plan's normal
	float x1 = step;float y1 = 0.0;float z1 = 0.0;
	float x2 = 0.0;float y2 = step;float z2 = 0.0;

	rotate3(&x1,&y1,&z1,angle);
	rotate3(&x2,&y2,&z2,angle);

	float xl = -1.;
	float yl = 1.;
	float zl = -1.;
	rotate3(&xl,&yl,&zl,angle);

	float ddx=dx;
	float ddy=dy;
	float ddz=dz;
	int c=nb;
	bool out = true; // if ray is out main c=0
	do {
		x += dx;y += dy;z += dz;

		if (CalcMandelBox3D(x, y, z, crn)==0)
		{
			// ray is not out. we ll see if normal is out now
			out=false;
			c=12;

			// for normal 3D
			x1=x + x1;
			y1=y + y1;
			z1=z + z1;

			x2=x + x2;
			y2=y + y2;
			z2=z + z2;

			ddx=dx;ddy= dy;ddz=dz;
			float d1x=dx;float d1y=dy;float d1z=dz;
			float d2x=dx;float d2y=dy;float d2z=dz;
			int in=0,in1=0,in2=0;//,in3=0;

			// place les 2 rayons pour les normales contre la forme
			if (CalcMandelBox3D(x1, y1, z1, crn)==0)
			{
				do {
					x1 -= d1x;y1 -= d1y;z1 -= d1z;
					if (x1*x1 + y1*y1 + z1*z1 > float(OUTMANDELBOX)) out=true;
					//if (x1 + y1 + z1 + w1 > float(OUTMANDELBOX)) out=true;
				} while ((CalcMandelBox3D(x1, y1, z1, crn) == 0) && (!out) );
			} else {
				do {
					x1 += d1x;y1 += d1y;z1 += d1z;
					if (x1*x1 + y1*y1 + z1*z1 > float(OUTMANDELBOX)) out=true;
					//if (x1 + y1 + z1 + w1 > float(OUTMANDELBOX)) out=true;
				} while ((CalcMandelBox3D(x1, y1, z1, crn) != 0) && (!out) );
			}
			//if (CalcJulia4D(x2, y2, z2, w2, JS, crn)==0)
			if (CalcMandelBox3D(x2, y2, z2, crn)==0)
			{
				do {
					x2 -= d2x;y2 -= d2y;z2 -= d2z;
					if (x2*x2 + y2*y2 + z2*z2 > float(OUTMANDELBOX)) out=true;
					//if (x2 + y2 + z2 + w2 > float(OUTMANDELBOX)) out=true;
				} while ((CalcMandelBox3D(x2, y2, z2, crn) == 0) && (!out) );
			} else {
				do {
					x2 += d2x;y2 += d2y;z2 += d2z;
					if (x2*x2 + y2*y2 + z2*z2 > float(OUTMANDELBOX)) out=true;
					//if (x2 + y2 + z2 + w2 > float(OUTMANDELBOX)) out=true;
				} while ((CalcMandelBox3D(x2, y2, z2, crn) != 0) && (!out) );
			}

			if (!out) {
				do {
					in  = CalcMandelBox3Dhue(x,  y,  z, &hue, crn);
					in1 = CalcMandelBox3D(x1, y1, z1, crn);
					in2 = CalcMandelBox3D(x2, y2, z2, crn);
					if (in==0) {
						x -= ddx;y -= ddy;z -= ddz;
					} else {
						x += ddx;y += ddy;z += ddz;
					}
					if (in1==0) {
						x1 -= d1x;y1 -= d1y;z1 -= d1z;
					} else {
						x1 += d1x;y1 += d1y;z1 += d1z;
					}
					if (in2==0) {
						x2 -= d2x;y2 -= d2y;z2 -= d2z;
					} else {
						x2 += d2x;y2 += d2y;z2 += d2z;
					}
					ddx /= 2.0;ddy /= 2.0;ddz /= 2.0;
					d1x /= 2.0;d1y /= 2.0;d1z /= 2.0;
					d2x /= 2.0;d2y /= 2.0;d2z /= 2.0;
				} while (c-->0);
			} else c=1;
		}
	} while (c-->0);

	if (out) {
		*r = 1;
		*g = 1;
		*b = 1;
	} else {
		// computing vector
		x1 -= x;y1 -= y;z1 -= z;
		x2 -= x;y2 -= y;z2 -= z;
		// vector product for normal
		// 3D Normal in space vue
		// x0 = x1 * x2 - y1 * y2 - z1 * z2 - w1* w2;
		// y0 = x1 * y2 + y1 * x2 + z1 * w2 - w1* z2;
		// z0 = x1 * z2 + z1 * x2 + w1 * y2 - y1* w2;
		// w0 = x1 * w2 + w1 * x2 + y1 * z2 - z1* y2;
		// 4D Normal
		// x0 = y1*(w2*z3-z2*w3)+y2*(z1*w3-w1*z3)+y3*(w1*z2-z1*w2);
		// y0 = x1*(z2*w3-w2*z3)+x2*(w1*z3-z1*w3)+x3*(z1*w2-w1*z2);
		// z0 = x1*(w2*y3-y2*w3)+x2*(y1*w3-w1*y3)+x3*(w1*y2-y1*w2);
		// w0 = x1*(y2*z3-z2*y3)+x2*(z1*y3-y1*z3)+x3*(y1*z2-z1*y2);
		// 3D Normal in space xyz
		x0 = y1 * z2 - z1 * y2;
		y0 = z1 * x2 - x1 * z2;
		z0 = x1 * y2 - y1 * x2;
		// w0 = 0.;

		// Normalisation
		float nd=sqrt(dx*dx+dy*dy+dz*dz);
		float n0=sqrt(x0*x0+y0*y0+z0*z0);
		float nl=sqrt(xl*xl+yl*yl+zl*zl);
		dx/=nd;dy/=nd;dz/=nd;
		x0/=n0;y0/=n0;z0/=n0;
		xl/=nl;yl/=nl;zl/=nl;

		// angle of direction / normal
		float anv = ( x0 * dx + y0 *dy + z0 *dz );
		if (anv<0.) anv=0.;

		// angle of light direction / normal
		float anl = -( x0* xl + y0* yl + z0*zl );
		if (anl<0.) anl=0.;

		// radiance
		float anr = 0.;
		if ( xl*x0 + yl*y0 + zl*z0 < 0. )
		{
			float xr=xl+2.*x0;float yr=yl+2.*y0;float zr=zl+2.*z0;
			float nr=sqrt(xr*xr+yr*yr+zr*zr);
			xr/=nr;yr/=nr;zr/=nr;
			anr = -0.85 -(xr*dx + yr*dy + zr*dz);
		}
		if ( anr < 0. ) anr=0.;
		anr *= 9.;
		if ( anr > 1. ) anr=1.;
		// shadow
		float sh = 1.0;
		out=true;
		do {
			x += xl*step;y += yl*step;z += zl*step; //sh+=0.1;
			if (CalcMandelBox3D(x, y, z, crn)==0) out = false;
		} while ( (x*x + y*y + z*z < float(OUTMANDELBOX)) &&(out));

		float li = anl*0.7+0.1;
		if (!out)
		{
			sh=0.5;
			anr=0.0;
		}
		float L = (li + (1. - li)*anr*anr) * sh;
		// if ( L < 0.0 ) L = 0.0;
		HSL2RGB(hue, 0.6, L, r, g, b);
	}
	return out;
} // SolidMandelBox3D*/


// Determine if two pixel colors are within tolerance
__device__ inline int CheckColors(const uchar4 &color0, const uchar4 &color1)
{
	int x = color1.x - color0.x;
	int y = color1.y - color0.y;
	int z = color1.z - color0.z;
	return (ABS(x) > 10) || (ABS(y) > 10) || (ABS(z) > 10);
} // CheckColors


// Increase the grid size by 1 if the image width or height does not divide evenly
// by the thread block dimensions
inline int iDivUp(int a, int b)
{
	return ((a % b) != 0) ? (a / b + 1) : (a / b);
} // iDivUp

#endif