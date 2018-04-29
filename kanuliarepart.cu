#ifndef _KANULIAREPART_CU_
#define _KANULIAREPART_CU_

#include <stdio.h>
//#include "cutil_inline.h"
#include "kanulia.h"
#include "kanuliacalc.cu"


//  Rotation de quaternion

__device__ inline void rotate4(float *px, float *py, float *pz, float *pw, const float4 angle)
{
	float t;
	if (angle.x != 0. ) {
		t  =   *py * cos(angle.x) + *pz * sin(angle.x);
		*pz = - *py * sin(angle.x) + *pz * cos(angle.x);
		*py = t;
	};
	if (angle.y != 0. ) {
		t   =   *px * cos(angle.y) + *pz * sin(angle.y);
		*pz = - *px * sin(angle.y) + *pz * cos(angle.y);
		*px = t;
	};
	if (angle.z != 0. ) {
		t   =   *pz * cos(angle.z) + *pw * sin(angle.z);
		*pw = - *pz * sin(angle.z) + *pw * cos(angle.z);
		*pz = t;
	};
	if (angle.w != 0. ) {
		t   =   *py * cos(angle.w) + *pw * sin(angle.w);
		*pw = - *py * sin(angle.w) + *pw * cos(angle.w);
		*py = t;
	};
}
__device__ inline void rotate4inv(float *px, float *py, float *pz, float *pw, const float4 angle)
{
	float t;

	if (angle.w != 0. ) {
		t   =   *py * cos(-angle.w) + *pw * sin(-angle.w);
		*pw = - *py * sin(-angle.w) + *pw * cos(-angle.w);
		*py = t;
	};
	if (angle.z != 0. ) {
		t   =   *pz * cos(-angle.z) + *pw * sin(-angle.z);
		*pw = - *pz * sin(-angle.z) + *pw * cos(-angle.z);
		*pz = t;
	};
	if (angle.y != 0. ) {
		t   =   *px * cos(-angle.y) + *pz * sin(-angle.y);
		*pz = - *px * sin(-angle.y) + *pz * cos(-angle.y);
		*px = t;
	};
	if (angle.x != 0. ) {
		t  =   *py * cos(-angle.x) + *pz * sin(-angle.x);
		*pz = - *py * sin(-angle.x) + *pz * cos(-angle.x);
		*py = t;
	};
}

__device__ inline void rotate3(float *px, float *py, float *pz, const float4 angle)
{
	float t;
	if (angle.x != 0. ) {
		t  =    *py * cos(angle.x) + *pz * sin(angle.x);
		*pz = - *py * sin(angle.x) + *pz * cos(angle.x);
		*py =   t;
	};
	if (angle.y != 0. ) {
		t   =   *px * cos(angle.y) + *pz * sin(angle.y);
		*pz = - *px * sin(angle.y) + *pz * cos(angle.y);
		*px =   t;
	};
	if (angle.z != 0. ) {
		t   =   *px * cos(angle.z) - *py * sin(angle.z);
		*py =   *px * sin(angle.z) + *py * cos(angle.z);
		*px =   t;
	};
/*	if (angle.w != 0. ) {
		t   =   *py * cos(angle.w) + *pw * sin(angle.w);
		*pw = - *py * sin(angle.w) + *pw * cos(angle.w);
		*py = t;
	};*/
}

// The Julia4D CUDA GPU thread function

/*
    Version using software scheduling of thread blocks.

    The idea here is to launch of fixed number of worker blocks to fill the
    machine, and have each block loop over the available work until it is all done.

    We use a counter in global memory to keep track of which blocks have been
    completed. The counter is incremented atomically by each worker block.

    This method can achieve higher performance when blocks take a wide range of
    different times to complete.
*/


// The core Julia CUDA GPU calculation function
/**/
__device__ int CloudJulia4D(const float ox, const float oy, const float oz, const float ow, const float4 JS, const float dx, const float dy, const float dz, const float dw, int *r, int *g, int *b, int nb, const unsigned int crn)
{
	float ret = 0;
	float x = ox;
	float y = oy;
	float z = oz;
	float w = ow;
	int c = nb;
	do {
		x += dx;
		y += dy;
		z += dz;
		w += dw;

		if (CalcJulia4D(x, y, z, w, JS, crn) == 0) ret += 1;
	} while (c--);

	if (ret>255) ret = 255;
	if (ret == 0) {
		*r = 0;
		*g = 0;
		*b = 0;
	}
	else {
		*r = ret;
		*g = ret;
		*b = 155;
	}
	return ret;
} // CloudJulia4D

/** return if y is cutted by the cutjulia option*/
/* if non 0.0 it is cutted out and the return value is the distance to the next un cutted plan */
__device__ bool iscuttedout(bool cutjulia,float y){
	float d = 0.15f;// distance between 2 layer start
	float h = 0.02f;// width of the layer, ( h < d )
	if (!cutjulia) return false;
//	float ymodd = abs(y) - (int(abs(y) / d))*d; // = y % d
//	return (ymodd > h / 2.) && (ymodd < d - h / 2.0);
	float ymodd = (y/*+10.0*d*/) - (int((y/*+10.0*d*/) / d))*d; // = y % d
	return (ymodd > h );
}

/** return distance factor to next uncutted out plan */
__device__ float getstepstonextplan(float y,float dy) {
	float d = 0.15f;// distance between 2 layer start
	//float h = 0.02f;// width of the layer, ( h < d )
	float ymodd = y - (int(y / d))*d; // = y % d
/*	if (dy > 0.)
		return (d - (h / 2.0) - ymodd) / dy;
	else
		return ((h / 2.0) - ymodd ) / dy;*/
	return (d - ymodd) / dy;
}

// The core Julia CUDA GPU calculation function
__device__ int SolidJulia4D(const int ix, const int iy, const float4 JS, const float4 angle,
	const int d_imageW, const int d_imageH, const float scaleJ,
	const float xblur, const float yblur, int *r, int *g, int *b, const float xJOff, const float yJOff, const unsigned int crn, int julia4D,
	const bool cutjulia)
{
	//hue color
	float hue;
	float dist = 6.0;
	float step = RAYSTEP;

	float x = ((float)ix + (xblur)) * scaleJ + xJOff;
	float y = ((float)iy + (yblur)) * scaleJ + yJOff;
	float z = ZOBSERVER;
	float w = 0.0;

	if (julia4D & CROSSEYE)
	{
		if (ix < (d_imageW / 2.)) // image gauche
			x = ((float)ix + (d_imageW / 4.) + (xblur)) * scaleJ + xJOff + SPACEEYE;
		else // image droite
			x = ((float)ix - (d_imageW / 4.) + (xblur)) * scaleJ + xJOff - SPACEEYE;
	}
	else
	{
		x = ((float)ix + (xblur)) * scaleJ + xJOff;
	}

	float dx = sin(KANULFOV * step * scaleJ * ((float)ix + (xblur)-(d_imageW / 2.)) / ((float)d_imageW));
	float dy = sin(KANULFOV * step * scaleJ * ((float)iy + (yblur)-(d_imageH / 2.)) / ((float)d_imageW));
	float dz = step;
	float dw = 0.;
	if (julia4D & CROSSEYE)
	{
		if (ix < (d_imageW / 2.)) // image gauche
			dx -= CROSSANGLE;
		else // image droite
			dx += CROSSANGLE;
	}

	rotate4(&x, &y, &z, &w, angle);
	rotate4(&dx, &dy, &dz, &dw, angle);
	float nd = sqrt(dx*dx + dy*dy + dz*dz + dw*dw);
	//	float mx = 0.;
	//	float ndx =dx/nd;float ndy =dy/nd;float ndz =dz/nd;float ndw =dw/nd;
	int nb = (dist / step);

	// hum sert a rien ?
	float x0 = 0.0; float y0 = -1.0; float z0 = 0.0; float w0 = 0.0;// normal is the secant plan's normal

	// Les trois rays qui vont servir a calculer la normale
	float x1 = step; float y1 = 0.0; float z1 = 0.0; float w1 = 0.0;
	float x2 = 0.0; float y2 = step; float z2 = 0.0; float w2 = 0.0;
	//float x3 = 0.0;float y3 = 0.0;float z3 = 0.0;float w3 = step;

	rotate4(&x1, &y1, &z1, &w1, angle);
	rotate4(&x2, &y2, &z2, &w2, angle);

	// light source direction
	float xl = 1.;
	float yl = -1.;
	float zl = 1.;
	float wl = 0.;

	float ddx = dx;
	float ddy = dy;
	float ddz = dz;
	float ddw = dw;
	int c = nb;
	bool out = true; // if ray is out main c=0
	bool cutplan = false; // if ray hit cutting plan
//	int logout = 0;
	do {
		// if inside empty aera
		if (iscuttedout(cutjulia,y))
			{
				// hit the surface
				float dhit = getstepstonextplan(y,dy);
				//if (logout == 1000) {
				//	printf("%f pouet ", dhit);
				//	logout = 0;
				//}
				//else {
				//	logout++;
				//}
				x += dx * dhit;
				y += dy * dhit;
				z += dz * dhit;
				w += dw * dhit;
				if (CalcJulia4Dcore(x, y, z, w, JS, &hue) >= crn)
				{
					c = 0; // stop, we hit the inside
					// for normal 3D
					x1 = x + x1;
					y1 = y + y1;
					z1 = z + z1;
					w1 = w + w1;
					dhit = -y1 / dy;
					x1 += dx * dhit;
					y1 += dy * dhit;
					z1 += dz * dhit;
					w1 += dw * dhit;

					x2 = x + x2;
					y2 = y + y2;
					z2 = z + z2;
					w2 = w + w2;
					dhit = -y2 / dy;
					x2 += dx * dhit;
					y2 += dy * dhit;
					z2 += dz * dhit;
					w2 += dw * dhit;

					//x3=x + x3;
					//y3=y + y3;
					//z3=z + z3;
					//w3=w + w3;
					//dhit = -y3/dy;
					//x3 += dx * dhit;
					//y3 += dy * dhit;
					//z3 += dz * dhit;
					//w3 += dw * dhit;
					cutplan = true;
					out = false;
//				}
			}
		}
		else
		{

/*			if (logout == 1000) {
				printf("%f pouIIIt ",dy);
				logout = 0;
			}
			else {
				logout++;
			}*/
			x += dx; y += dy; z += dz; w += dw;
			//			x += ndx*step;y += ndy*step;z += ndz*step;w += ndw*step;

			//			if (CalcJulia4Dstep(x, y, z, w, JS, crn,&step)==0)
			if (CalcJulia4D(x, y, z, w, JS, crn) == 0)
			{
				// ray is not out. we ll see if normal is out now
				out = false;
				c = 12;

				// for normal 3D
				x1 = x + x1;
				y1 = y + y1;
				z1 = z + z1;
				w1 = w + w1;

				x2 = x + x2;
				y2 = y + y2;
				z2 = z + z2;
				w2 = w + w2;

				//x3=x + x3;
				//y3=y + y3;
				//z3=z + z3;
				//w3=w + w3;

				ddx = dx; ddy = dy; ddz = dz; ddw = dw;
				float d1x = dx*2.0; float d1y = dy*2.0; float d1z = dz*2.0; float d1w = dw*2.0;
				float d2x = dx*2.0; float d2y = dy*2.0; float d2z = dz*2.0; float d2w = dw*2.0;
				//float d3x=dx*2.0;float d3y=dy*2.0;float d3z=dz*2.0;float d3w=dw*2.0;
				int in = 0, in1 = 0, in2 = 0;//,in3=0;

				// place les 3 rayons pour les normales contre la forme
				if (CalcJulia4D(x1, y1, z1, w1, JS, crn) == 0)
				{
					do {
						x1 -= d1x; y1 -= d1y; z1 -= d1z; w1 -= d1w;
						if (x1*x1 + y1*y1 + z1*z1 + w1*w1 > OUTMANDELBOX) out = true;
					} while ((CalcJulia4D(x1, y1, z1, w1, JS, crn) == 0) && (!out));
				}
				else {
					do {
						x1 += d1x; y1 += d1y; z1 += d1z; w1 += d1w;
						if (x1*x1 + y1*y1 + z1*z1 + w1*w1 > OUTMANDELBOX) out = true;
					} while ((CalcJulia4D(x1, y1, z1, w1, JS, crn) != 0) && (!out));
				}
				if (CalcJulia4D(x2, y2, z2, w2, JS, crn) == 0)
				{
					do {
						x2 -= d2x; y2 -= d2y; z2 -= d2z; w2 -= d2w;
						if (x2*x2 + y2*y2 + z2*z2 + w2*w2 > OUTMANDELBOX) out = true;
					} while ((CalcJulia4D(x2, y2, z2, w2, JS, crn) == 0) && (!out));
				}
				else {
					do {
						x2 += d2x; y2 += d2y; z2 += d2z; w2 += d2w;
						if (x2*x2 + y2*y2 + z2*z2 + w2*w2 > OUTMANDELBOX) out = true;
					} while ((CalcJulia4D(x2, y2, z2, w2, JS, crn) != 0) && (!out));
				}
				//if (CalcJulia4D(x3, y3, z3, w3, JS, crn)==0)
				//{
				//	do {
				//		x3 -= d3x;y3 -= d3y;z3 -= d3z;w2 -= d3w;
				//		if (x3*x3 + y3*y3 + z3*z3 + w3*w3 > OUTMANDELBOX) out=true;
				//	} while ((CalcJulia4D(x3, y3, z3, w3, JS, crn) == 0) && (!out) );
				//} else {
				//	do {
				//		x3 += d3x;y3 += d3y;z3 += d3z;w3 += d3w;
				//		if (x3*x3 + y3*y3 + z3*z3 + w3*w3 > OUTMANDELBOX) out=true;
				//	} while ((CalcJulia4D(x3, y3, z3, w3, JS, crn) != 0) && (!out) );
				//}

				if (!out) {
					do {
						in = CalcJulia4Dhue(x, y, z, w, JS, &hue, crn);
						in1 = CalcJulia4D(x1, y1, z1, w1, JS, crn);
						in2 = CalcJulia4D(x2, y2, z2, w2, JS, crn);
						//in3 = CalcJulia4D(x3, y3, z3, w3, JS, crn);
						if (in == 0) {
							x -= ddx; y -= ddy; z -= ddz; w -= ddw;
						}
						else {
							x += ddx; y += ddy; z += ddz; w += ddw;
						}
						if (in1 == 0) {
							x1 -= d1x; y1 -= d1y; z1 -= d1z; w1 -= d1w;
						}
						else {
							x1 += d1x; y1 += d1y; z1 += d1z; w1 += d1w;
						}
						if (in2 == 0) {
							x2 -= d2x; y2 -= d2y; z2 -= d2z; w2 -= d2w;
						}
						else {
							x2 += d2x; y2 += d2y; z2 += d2z; w2 += d2w;
						}
						//if (in3==0) {
						//	x3 -= d3x;y3 -= d3y;z3 -= d3z;w3 -= d3w;
						//} else {
						//	x3 += d3x;y3 += d3y;z3 += d3z;w3 += d3w;
						//}
						ddx /= 2.0; ddy /= 2.0; ddz /= 2.0; ddw /= 2.0;
						d1x /= 2.0; d1y /= 2.0; d1z /= 2.0; d1w /= 2.0;
						d2x /= 2.0; d2y /= 2.0; d2z /= 2.0; d2w /= 2.0;
						//d3x /= 2.0;d3y /= 2.0;d3z /= 2.0;d3w /= 2.0;
					} while (c-->0);
				}
				else c = 1;
			}
		}
		//		if (mx>4.0) c=1;
	} while (c-->0);

	if (out) {
		//		while (x*x+y*y+z*z+w*w<OUTBOX)
		/*		while ((x<OUTBOX)&&(x>-OUTBOX)
		&&(y<OUTBOX)&&(y>-OUTBOX)
		&&(z<OUTBOX)&&(z>-OUTBOX)
		&&(w<OUTBOX)&&(w>-OUTBOX))*/
		/*		{
		x+=dx;y+=dy;z+=dz;w+=dw;
		}*/
		*r = 1;
		*g = 1;
		*b = 1;
		//		if ((x-(float)((int)(x*1.))/1.<0.01)
		//		  ||(y-(float)((int)(y*1.))/1.<0.01)
		//		  ||(z-(float)((int)(z*10.))/10.<0.01)
		//		  ||(w-(float)((int)(w*10.))/10.<0.01)
		//			)
		/*		if (
		(ABS(x-(float)((int)(x*7.))/7.)<0.01)
		||(ABS(y-(float)((int)(y*7.))/7.)<0.01)
		||(ABS(z-(float)((int)(z*7.))/7.)<0.01)
		||(ABS(w-(float)((int)(w*7.))/7.)<0.01)
		)
		{
		*r = 255;
		*g = 255;
		*b = 255;
		}*/
	}
	else {
		// computing vector
		x1 -= x; y1 -= y; z1 -= z; w1 -= w;
		x2 -= x; y2 -= y; z2 -= z; w2 -= w;
		//x3 -= x;y3 -= y;z3 -= z;w3 -= w;
		// vector product for normal
		// 3D Normal in space vue
		//x0 = x1 * x2 - y1 * y2 - z1 * z2 - w1* w2;
		//y0 = x1 * y2 + y1 * x2 + z1 * w2 - w1* z2;
		//z0 = x1 * z2 + z1 * x2 + w1 * y2 - y1* w2;
		//w0 = x1 * w2 + w1 * x2 + y1 * z2 - z1* y2;
		// 4D Normal
		//x0 = y1*(w2*z3-z2*w3)+y2*(z1*w3-w1*z3)+y3*(w1*z2-z1*w2);
		//y0 = x1*(z2*w3-w2*z3)+x2*(w1*z3-z1*w3)+x3*(z1*w2-w1*z2);
		//z0 = x1*(w2*y3-y2*w3)+x2*(y1*w3-w1*y3)+x3*(w1*y2-y1*w2);
		//w0 = x1*(y2*z3-z2*y3)+x2*(z1*y3-y1*z3)+x3*(y1*z2-z1*y2);

		// retour dans le repere de la cam
		rotate4inv(&dx, &dy, &dz, &dw, angle);
		rotate4inv(&x1, &y1, &z1, &w1, angle);
		rotate4inv(&x2, &y2, &z2, &w2, angle);

		// 3D Normal in space xyz
		x0 = z1 * y2 - y1 * z2;
		y0 = x1 * z2 - z1 * x2;
		z0 = y1 * x2 - x1 * y2;
		w0 = 0.;
		if (cutplan)
		{
			x0 = 0.0; y0 = -1.0; z0 = 0.0; w0 = 0.0;// normal is the secant plan's normal
			rotate4inv(&x0, &y0, &z0, &w0, angle);
			float n0 = sqrt(x0*x0 + y0*y0 + z0*z0);//+w0*w0);
			x0 /= n0; y0 /= n0; z0 /= n0;//w0/=n0;
		}

		// Normalisation
		float n0 = sqrt(x0*x0 + y0*y0 + z0*z0);//+w0*w0);
		float nl = sqrt(xl*xl + yl*yl + zl*zl);//+wl*wl);
		float nd = sqrt(dx*dx + dy*dy + dz*dz);//+dw*dw);
		x0 /= n0; y0 /= n0; z0 /= n0;//w0/=n0;
		xl /= nl; yl /= nl; zl /= nl;//wl/=nl;
		dx /= nd; dy /= nd; dz /= nd;//dw/=nd;

		// angle of direction / normal
		/*		float anv = (x0 * dx + y0 *dy + z0 *dz + w0 *dw);
		if (anv<0.) anv=0.;*/

		// angle of light direction / normal
		float anl = -(x0* xl + y0* yl + z0*zl);// + w0*wl);
		if (anl<0.) anl = 0.;
		//		dx=0.;dy=0.;dz=1.;dw=0.;

		// radiance	
		float anr = 0.;
		float pscal = (xl*x0 + yl*y0 + zl*z0);// + wl*w0);
		if (pscal < 0.)
		{
			float xr = xl - x0*2.*pscal; float yr = yl - y0*2.*pscal; float zr = zl - z0*2.*pscal;//float wr=wl-w0*2.*pscal;
			float nr = sqrt(xr*xr + yr*yr + zr*zr);//+wr*wr);
			xr /= nr; yr /= nr; zr /= nr;//wr/=nr;
			anr = -(xr*dx + yr*dy + zr*dz);// + wr*dw);
			//			anr = -pscal;
			//			anr = -(x0*dx + y0*dy + z0*dz + w0*dw);
			anr = anr * 8.5 - 7.;
			//if ( anr < 0.8 ) anr=0.;
			if (anr > 1.) anr = 1.;
			if (anr < 0.) anr = 0.;
			anr = anr*anr;
		}

		// shadow computation
		float sh = 1.0;
		bool shadow = false;
		// light source rotate with camera
		rotate4(&xl, &yl, &zl, &wl, angle);
		do {
			x -= xl*step; y -= yl*step; z -= zl*step; w -= wl*step;
			//if ((y > 0.) || (!cutjulia))
			if (!iscuttedout(cutjulia,y))
					if (CalcJulia4D(x, y, z, w, JS, crn) == 0) shadow = true;
		} while ((x*x + y*y + z*z + w*w < OUTMANDELBOX) && (!shadow) && (iscuttedout(cutjulia,y)));

		float li = anl*0.7 + 0.1;
		if (shadow)
		{
			sh = 0.5;
			anr = 0.0;
		}
		float L = (li + (1. - li)*anr) * sh;
		// if ( L < 0.0 ) L = 0.0;
		HSL2RGB(hue, 0.5, L, r, g, b);
	}
	return out;
} // SolidJulia4D


__device__ unsigned int blockCounter;   // global counter, initialized to zero before kernel launch

__global__ void Julia4Drepart(uchar4 *dst, const int imageW, const int imageH,
 const float4 Off, const float4 JS, const float4 angle, const float scale, const float scalei,
 const float xJOff, const float yJOff, const float scaleJ,
 const float xblur, const float yblur,
 const unsigned int maxgropix,
 const unsigned int gropix, const unsigned int bloc, const unsigned int crn,
 const uchar4 colors, const int frame,
 const int animationFrame, const int gridWidth, const int numBlocks, const int julia, const int julia4D,
 const bool cutjulia)
{
    __shared__ unsigned int blockIndex;
    __shared__ unsigned int blockX, blockY;
	
    // loop until all blocks completed
    while(1) {
        if ((threadIdx.x==0) && (threadIdx.y==0)) {
            // get block to process
            blockIndex = atomicAdd(&blockCounter, 1);
            //blockIndex++;
            blockX = blockIndex % gridWidth;            // note: this is slow, but only called once per block here
            blockY = blockIndex / gridWidth;
        }
#ifndef __DEVICE_EMULATION__        // device emu doesn't like syncthreads inside while()
        __syncthreads();
#endif

//        if (blockIndex >= ((numBlocks/nbloc)+1)*(bloc+1)) break;  // finish
        if (blockIndex >= numBlocks) break;  // finish

        // process this block
        const int ix = blockDim.x * blockX * maxgropix + threadIdx.x * maxgropix + ((bloc * gropix) % maxgropix);
        const int iy = blockDim.y * blockY * maxgropix + threadIdx.y * maxgropix + ((bloc * gropix) / maxgropix) * gropix;

		int r = 0;int g = 0;int b = 0;
		bool seedre = false;bool seedim = false;

		if ((ix < imageW) && (iy < imageH)) {
			int m = 0;
	        if ( (julia<32) && (ix < imageW / julia) && (iy < imageH / julia)) {
			    // Calculate the location
			    const float xPos = (float)ix * scale * julia + Off.x;
				const float yPos = (float)iy * scale * julia + Off.y;

				// Calculate the Mandelbrot index for the current location
				if (abs(JS.x-xPos)+abs(JS.y-yPos) < 2.1 * scale * julia )
				{
					seedre = true; 
				}
				if (!seedre)
				{
					float hue;
//					m = CalcMandelbrot(xPos , yPos);
					m = CalcMandel4Dcore(xPos,  yPos,  JS.z,  JS.w, &hue);
					if (m<=256) HSL2RGB(hue, 0.6, 0.5, &r, &g, &b);
				}
    		} else if (julia4D&& (julia<32) &&((imageW - ix < imageW / julia) && (iy < imageH / julia))) {
			    // Calculate the location
			    const float zPos = (float)(imageW - ix) * scalei * julia + Off.z;
				const float wPos = (float)iy           * scalei * julia  + Off.w;

				// Calculate the Mandelbrot index for the current location
				if (abs(JS.z-zPos)+abs(JS.w-wPos) < 2.1 * scalei * julia )
				{
					seedim = true; 
				}
				if (!seedim)
				{
					float hue;
//					m = CalcMandelbrot(zPos , wPos);
					m = CalcMandel4Dcore(JS.x,  JS.y,  zPos,  wPos, &hue);
					if (m<=256) HSL2RGB(hue, 0.6, 0.5, &r, &g, &b);
				}
			} else {
			    // Calculate the location
			    const float xPos = (float)ix * scaleJ + xJOff;
				const float yPos = (float)iy * scaleJ + yJOff;
/*				const float zPos = (float)0.;
				const float wPos = (float)0.;*/
				// Calculate the Mandelbrot index for the current location
				if (julia4D == JULIA2D)
				{
					m = CalcJulia(xPos, yPos, JS, crn);
				}
				if (julia4D == CLOUDJULIA)
				{
					float dist = 6.0;
					float step = 0.009;

					float ox = (float)ix * scaleJ + xJOff;
					float oy = (float)iy * scaleJ + yJOff;
					float oz = - 3.0;
					float ow = 0.0;
					float dx = sin( 0.7 * step * ( (float)ix + xblur - (imageW/2.)) / ((float) imageW) );
					float dy = sin( 0.7 * step * ( (float)iy + yblur - (imageH/2.)) / ((float) imageW) );
					float dz = step;
					float dw = 0.;
					rotate4(&ox,&oy,&oz,&ow,angle);
					rotate4(&dx,&dy,&dz,&dw,angle);
					int nb = (dist/step);
					m = CloudJulia4D(ox, oy, oz, ow, JS, dx, dy, dz, dw, &r, &g, &b, nb, crn);
				}
				if (julia4D & JULIA4D)
				{
/*					if ((julia4D & CROSSEYE)&&
					   (  (sqrt( (float)((ix-  imageW/4)*(ix-  imageW/4) + (iy-(imageH)/5)*(iy-(imageH)/5) )) < 20.)						// si viseur
						||(sqrt( (float)((ix-3*imageW/4)*(ix-3*imageW/4) + (iy-(imageH)/5)*(iy-(imageH)/5) )) < 20.)))
					{
						r = 255;
						g = 255;
						b = 255;
					}
					else*/
						m = SolidJulia4D(ix-1,iy-1,JS,angle,imageW,imageH,scaleJ,xblur,yblur,&r,&g,&b,xJOff,yJOff,crn,julia4D,cutjulia);
	//				m = SolidMandelBox3D(ix-1,iy-1,JS,angle,imageW,imageH,scaleJ,xblur,yblur,&r,&g,&b,xJOff,yJOff,crn);
				}
    		}
//			m = blockIdx.x;         // uncomment to see scheduling order

            // Convert the Mandelbrot index into a color
            uchar4 color;
//			m = m > 0 ? crn - m : 0;

            if ((julia4D)&&((ix >= imageW / julia) || (iy >= imageH / julia))) {
				color.x = r;
				color.y = g;
				color.z = b;
			} else
			{
				if (seedim||seedre)
				{
					color.x = 150;
					color.y = 250;
					color.z = 250;
				} else {
					color.x = r;
					color.y = g;
					color.z = b;

/*					if (m) {
						m += animationFrame;
						color.x = m * colors.x;
						color.y = m * colors.y;
						color.z = m * colors.z;
					} else {
						color.x = 0;
						color.y = 0;
						color.z = 0;
					}*/
					
				}
			}
			
			// activer pour voir le calcul progressif
//			if (gropix==1) color.z += 120;
//			if (gropix==2) color.y += 120;
//			if (gropix==4) color.x += 120;
//				
					
			// Output the pixel
			int pixel = imageW * iy + ix;
			if (frame == 0) {
			    color.w = 0;
				if (gropix==1)
					dst[pixel] = color;
				else
					for (int i=0;i<gropix;i++) for (int j=0;j<gropix;j++)
						if ((ix+i<imageW)&&(iy+j<imageH))
							dst[pixel+i+imageW*j] = color;
			} else {
			    int frame1 = frame + 1;
			    int frame2 = frame1 / 2;
			    dst[pixel].x = (dst[pixel].x * frame + color.x + frame2) / frame1;
			    dst[pixel].y = (dst[pixel].y * frame + color.y + frame2) / frame1;
			    dst[pixel].z = (dst[pixel].z * frame + color.z + frame2) / frame1;
			}
        }

    }

} // Julia4D0


// The host CPU Mandebrot thread spawner
void RunJulia4Drepart(uchar4 *dst, const int imageW, const int imageH,
 const float4 Off,
 const float4 JS,
 const float4 angle,
 const double scale, const double scalei,
 const double xJOff, const double yJOff, const double scaleJ,
 const float xblur, const float yblur,
 const unsigned int maxgropix,
 const unsigned int gropix, const unsigned int bloc, const unsigned int crn,
 const uchar4 colors, const int frame, const int animationFrame, const int numSMs, const int julia, const int julia4D,
 const bool cutjulia)
{
    dim3 threads(BLOCKDIM_X, BLOCKDIM_Y);
    dim3 grid(iDivUp(imageW/maxgropix, BLOCKDIM_X), iDivUp(imageH/(maxgropix), BLOCKDIM_Y));

    // zero block counter
//    unsigned int hBlockCounter = (((grid.x)*(grid.y)/nbloc)+1)*(bloc);
    unsigned int hBlockCounter = 0;
    /*cutilSafeCall( */
	cudaMemcpyToSymbol(blockCounter, &hBlockCounter, sizeof(unsigned int), 0, cudaMemcpyHostToDevice /*)*/ );

	int numWorkUnit = numSMs;
	
	Julia4Drepart<<<numWorkUnit, threads>>>(dst, imageW, imageH,
						Off, JS, angle, (float)scale, (float)scalei,
						(float)xJOff, (float)yJOff, (float)scaleJ,
						xblur, yblur,
						maxgropix, gropix, bloc, crn,
						colors, frame, animationFrame, grid.x, (grid.x)*(grid.y), julia, julia4D,
						cutjulia);

//    cutilCheckMsg("Julia4D0_sm13 kernel execution failed.\n");
} // RunJulia4D0


// check if we're running in emulation mode
int inEmulationMode()
{
#ifdef __DEVICE_EMULATION__
    return 1;
#else
    return 0;
#endif
}

#endif
