#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <cuda.h>
#include <vector>
#include <thrust/sort.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/device_malloc.h>
#include <thrust/device_ptr.h>
#include "common.h"

#define NUM_THREADS 256

extern double size;

static texture<int2, 1, cudaReadModeElementType> old_pos_tex;
static texture<int2, 1, cudaReadModeElementType> old_vel_tex;
static texture<int2, 1, cudaReadModeElementType> old_acc_tex;
static texture<int,  1, cudaReadModeElementType> bin_index_tex;
static texture<int,  1, cudaReadModeElementType> particle_index_tex;
static texture<int,  1, cudaReadModeElementType> bin_start_tex;
static texture<int,  1, cudaReadModeElementType> bin_end_tex;

static __inline__ __device__ double fetch_double(texture<int2, 1> t, int i)
{
	int2 v = tex1Dfetch(t, i);
	return __hiloint2double(v.y, v.x);
}

//
//  benchmarking program
//
void init_particles_gpu(int n, double *pos, double *vel)
{
    srand48( time( NULL ) );

    int sx = (int)ceil(sqrt((double)n));
    int sy = (n+sx-1)/sx;

    int *shuffle = (int*)malloc( n * sizeof(int) );
    for( int i = 0; i < n; i++ )
        shuffle[i] = i;

    for( int i = 0; i < n; i++ )
    {
        //
        //  make sure particles are not spatially sorted
        //
        int j = lrand48()%(n-i);
        int k = shuffle[j];
        shuffle[j] = shuffle[n-i-1];

        //
        //  distribute particles evenly to ensure proper spacing
        //
//        p[i].x = size*(1.+(k%sx))/(1+sx);
//        p[i].y = size*(1.+(k/sx))/(1+sy);
	pos[2*i] = size*(1.+(k%sx))/(1+sx);
	pos[2*i+1] = size*(1.+(k/sx))/(1+sy);
        //
        //  assign random velocities within a bound
        //
//        p[i].vx = drand48()*2-1;
//        p[i].vy = drand48()*2-1;
	vel[2*i] = drand48()*2-1;
	vel[2*i+1] = drand48()*2-1;
    }
    free( shuffle );
}

void sort_particles(int *bin_index, int *particle_index, int n)
{
	thrust::sort_by_key(thrust::device_ptr<int>(bin_index),
			thrust::device_ptr<int>(bin_index + n),
			thrust::device_ptr<int>(particle_index));
}

// calculate particle's bin number
static __inline__ __device__ int binNum(double &d_x, double &d_y, int bpr) 
{
	return ( floor(d_x/cutoff) + bpr*floor(d_y/cutoff) );
}

__global__ void reorder_data_calc_bin(int *bin_start, int *bin_end, double *sorted_pos, double *sorted_vel, double *sorted_acc, int *bin_index, int *particle_index, double *d_pos, double *d_vel, double *d_acc, int n, int num_bins)
{
	extern __shared__ int sharedHash[];    // blockSize + 1 elements
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int bi;
	if (index < n) {
		bi = bin_index[index];
		sharedHash[threadIdx.x+1] = bi;
		if (index > 0 && threadIdx.x == 0)
		{
			// first thread in block must load neighbor particle hash
			sharedHash[0] = bin_index[index-1];
		}
	}

	__syncthreads();

	if (index < n) {
		// If this particle has a different cell index to the previous
		// particle then it must be the first particle in the cell,
		// so store the index of this particle in the cell.
		// As it isn't the first particle, it must also be the cell end of
		// the previous particle's cell

		if (index == 0 || bi != sharedHash[threadIdx.x])
		{
			bin_start[bi] = index;
			if (index > 0)
				bin_end[sharedHash[threadIdx.x]] = index;
		}

		if (index == n - 1)
		{
			bin_end[bi] = index + 1;
		}

		// Now use the sorted index to reorder the pos and vel data
		int sortedIndex = particle_index[index];
		sorted_pos[2*index]   = d_pos[2*sortedIndex];
		sorted_pos[2*index+1] = d_pos[2*sortedIndex+1];
		sorted_vel[2*index]   = d_vel[2*sortedIndex];
		sorted_vel[2*index+1] = d_vel[2*sortedIndex+1];
		sorted_acc[2*index]   = d_acc[2*sortedIndex];
		sorted_acc[2*index+1] = d_acc[2*sortedIndex+1];
	}
}

__global__ void calculate_bin_index(int *bin_index, int *particle_index, double *d_pos, int n, int bpr)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if(index >= n) return;
	double pos_x = fetch_double(old_pos_tex, 2*index);
	double pos_y = fetch_double(old_pos_tex, 2*index+1);
	int cbin = binNum( pos_x,pos_y,bpr );
	bin_index[index] = cbin;
	particle_index[index] = index;
}

static __inline__ __device__ void apply_force_gpu(double &particle_x, double &particle_y, double &particle_ax, double &particle_ay, double &neighbor_x, double &neighbor_y)
{
	double dx = neighbor_x - particle_x;
	double dy = neighbor_y - particle_y;
	double r2 = dx * dx + dy * dy;
	if( r2 > cutoff*cutoff )
		return;
	//r2 = fmax( r2, min_r*min_r );
	r2 = (r2 > min_r*min_r) ? r2 : min_r*min_r;
	double r = sqrt( r2 );

	//
	//  very simple short-range repulsive force
	//
	double coef = ( 1 - cutoff / r ) / r2 / mass;
	particle_ax += coef * dx;
	particle_ay += coef * dy;
}

__global__ void compute_forces_gpu(double *pos, double *acc, int n, int bpr, int *bin_start, int *bin_end)
{
	// Get thread (particle) ID
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if(tid >= n) return;

	double pos_1x = fetch_double(old_pos_tex, 2*tid);
	double pos_1y = fetch_double(old_pos_tex, 2*tid+1);

	// find current particle's in, handle boundaries
//	int cbin = binNum( pos[2*tid], pos[2*tid+1], bpr );
	int cbin = binNum( pos_1x, pos_1y, bpr );
	int lowi = -1, highi = 1, lowj = -1, highj = 1;
	if (cbin < bpr)
		lowj = 0;
	if (cbin % bpr == 0)
		lowi = 0;
	if (cbin % bpr == (bpr-1))
		highi = 0;
	if (cbin >= bpr*(bpr-1))
		highj = 0;

//	acc[2*tid] = acc[2*tid+1] = 0;
	double acc_x;
	double acc_y;
	acc_x = acc_y = 0;

	for (int i = lowi; i <= highi; i++)
		for (int j = lowj; j <= highj; j++)
		{
			int nbin = cbin + i + bpr*j;
			int bin_st = tex1Dfetch(bin_start_tex, nbin);
			if (bin_st != 0xffffffff) {
				int bin_et = tex1Dfetch(bin_end_tex, nbin);
				for (int k = bin_st; k < bin_et; k++ ) {
					double pos_2x = fetch_double(old_pos_tex, 2*k);
					double pos_2y = fetch_double(old_pos_tex, 2*k+1);
					apply_force_gpu( pos_1x, pos_1y, acc_x, acc_y, pos_2x, pos_2y );
				}
			}
		}
	acc[2*tid] = acc_x;
	acc[2*tid+1] = acc_y;
}

__global__ void move_gpu (double *pos, double *vel, double *acc, int n, double size)
{

	// Get thread (particle) ID
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if(tid >= n) return;

//	particle_t * p = &particles[tid];
	
	//
	//  slightly simplified Velocity Verlet integration
	//  conserves energy better than explicit Euler method
	//
//	p->vx += p->ax * dt;
//	p->vy += p->ay * dt;
//	p->x  += p->vx * dt;
//	p->y  += p->vy * dt;
	vel[2*tid]   += acc[2*tid] * dt;
	vel[2*tid+1] += acc[2*tid+1] * dt;
	pos[2*tid]   += vel[2*tid] * dt;
	pos[2*tid+1] += vel[2*tid+1] * dt;

	//
	//  bounce from walls
	//
/*
	while( p->x < 0 || p->x > size )
	{
		p->x  = p->x < 0 ? -(p->x) : 2*size-p->x;
		p->vx = -(p->vx);
	}
	while( p->y < 0 || p->y > size )
	{
		p->y  = p->y < 0 ? -(p->y) : 2*size-p->y;
		p->vy = -(p->vy);
	}
*/
	while( pos[2*tid] < 0 || pos[2*tid] > size )
	{
		pos[2*tid] = pos[2*tid] < 0 ? -(pos[2*tid]) : 2*size-pos[2*tid];
		vel[2*tid] = -(vel[2*tid]);
	}
	while( pos[2*tid+1] < 0 || pos[2*tid+1] > size )
	{
		pos[2*tid+1] = pos[2*tid+1] < 0 ? -(pos[2*tid+1]) : 2*size-pos[2*tid+1];
		vel[2*tid+1] = -(vel[2*tid+1]);
	}
}


int main( int argc, char **argv )
{    
	// This takes a few seconds to initialize the runtime
	cudaThreadSynchronize(); 

	if( find_option( argc, argv, "-h" ) >= 0 )
	{
		printf( "Options:\n" );
		printf( "-h to see this help\n" );
		printf( "-n <int> to set the number of particles\n" );
		printf( "-o <filename> to specify the output file name\n" );
		printf( "-s <filename> to specify the summary output file name\n" );
		return 0;
	}

	int n = read_int( argc, argv, "-n", 1000 );

	char *savename = read_string( argc, argv, "-o", NULL );
	char *sumname = read_string( argc, argv, "-s", NULL );

	FILE *fsave = savename ? fopen( savename, "w" ) : NULL;
	FILE *fsum = sumname ? fopen(sumname,"a") : NULL;
//	particle_t *particles = (particle_t*) malloc( n * sizeof(particle_t) );
	double *pos = (double *) malloc( 2*n * sizeof(double) );
	double *vel = (double *) malloc( 2*n * sizeof(double) );
	double *acc = (double *) malloc( 2*n * sizeof(double) );

	// GPU particle data structure
//	particle_t * d_particles;
	double *d_pos;
	double *d_vel;
	double *d_acc;
	
	cudaMalloc((void **) &d_pos, 2*n * sizeof(double));
	cudaMalloc((void **) &d_vel, 2*n * sizeof(double));
	cudaMalloc((void **) &d_acc, 2*n * sizeof(double));
//	particle_t * sorted_particles;
	double *sorted_pos;
	double *sorted_vel;
	double *sorted_acc;
	cudaMalloc((void **) &sorted_pos, 2*n * sizeof(double));
	cudaMalloc((void **) &sorted_vel, 2*n * sizeof(double));
	cudaMalloc((void **) &sorted_acc, 2*n * sizeof(double));

	int *bin_index;
	cudaMalloc((void **) &bin_index, n * sizeof(int));
	cudaMemset(bin_index, 0x0, n * sizeof(int));
	int *particle_index;
	cudaMalloc((void **) &particle_index, n * sizeof(int));
	cudaMemset(particle_index, 0x0, n * sizeof(int));

	set_size( n );

//	init_particles( n, particles );
	init_particles_gpu(n, pos, vel);
/*
	for(int i=0; i<n; ++i)
		printf("1.pos[%d].x=%f,pos[%d].y=%f,vel[%d].x=%f,vel[%d].y=%f,acc[%d].x=%f,acc[%d].y=%f\n",i,pos[2*i],i,pos[2*i+1],i,vel[2*i],i,vel[2*i+1],i,acc[2*i],i,acc[2*i+1]);
*/

	// create spatial bins (of size cutoff by cutoff)
	double size = sqrt( density*n );
	int bpr = ceil(size/cutoff);
	int num_bins = bpr*bpr;
	//	thrust::pointer< host_vector<particle_t*> > bins = new thrust::host_vector<particle_t*>[numbins];
	//	thrust::device_ptr< thrust::device_vector<particle_t*> > d_bins = thrust::device_malloc< thrust::device_vector<particle_t*> >(numbins);

	int *bin_start;
	int *bin_end;
	cudaMalloc((void **) &bin_start, num_bins * sizeof(int));
	cudaMalloc((void **) &bin_end, num_bins * sizeof(int));
	cudaMemset(bin_start, 0x0, num_bins * sizeof(int));
	cudaMemset(bin_end, 0x0, num_bins * sizeof(int));

	cudaThreadSynchronize();
	double copy_time = read_timer( );

	// Copy the particles to the GPU
//	cudaMemcpy(d_particles, particles, n * sizeof(particle_t), cudaMemcpyHostToDevice);
	cudaMemcpy(d_pos, pos, 2*n * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_vel, vel, 2*n * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_acc, acc, 2*n * sizeof(double), cudaMemcpyHostToDevice);

	int *bin_index_host = (int *) malloc( n * sizeof(int) );//##
	int *particle_index_host = (int *) malloc( n * sizeof(int) );//##
	int *bin_start_host = (int *) malloc( num_bins * sizeof(int) );//##
	int *bin_end_host = (int *) malloc( num_bins * sizeof(int) );//##
	double *d_pos_host = (double *) malloc( 2*n * sizeof(double) );//##
	double *d_vel_host = (double *) malloc( 2*n * sizeof(double) );//##
	double *d_acc_host = (double *) malloc( 2*n * sizeof(double) );//##
	double *sorted_pos_host = (double *) malloc( 2*n * sizeof(double) );//##
	double *sorted_vel_host = (double *) malloc( 2*n * sizeof(double) );//##
	double *sorted_acc_host = (double *) malloc( 2*n * sizeof(double) );//##

	cudaThreadSynchronize();
	copy_time = read_timer( ) - copy_time;

	//
	//  simulate a number of time steps
	//
	cudaThreadSynchronize();
	double simulation_time = read_timer( );
	for( int step = 0; step < NSTEPS; step++ )
	{

		int blks = (n + NUM_THREADS - 1) / NUM_THREADS;

		cudaBindTexture(0, old_pos_tex, d_pos, 2*n * sizeof(int2));
//		calculate_bin_index <<< blks, NUM_THREADS >>> (bin_index, particle_index, d_particles, n, bpr);
		calculate_bin_index <<< blks, NUM_THREADS >>> (bin_index, particle_index, d_pos, n, bpr);
		cudaUnbindTexture(old_pos_tex);
/*
		cudaMemcpy(d_pos_host, d_pos, 2*n * sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(bin_index_host, bin_index, n * sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(particle_index_host, particle_index, n * sizeof(int), cudaMemcpyDeviceToHost);
		for(int i=0; i<n; ++i){
			printf("0.0.bin_index[%d]=%d,particle_index[%d]=%d,d_pos[%d].x=%f,d_pos[%d].y=%f\n",i,bin_index_host[i],i,particle_index_host[i],i,d_pos_host[2*i],i,d_pos_host[2*i+1]);
		}
*/
		cudaBindTexture(0, bin_index_tex, bin_index, n * sizeof(int));
		cudaBindTexture(0, particle_index_tex, particle_index, n * sizeof(int));
		sort_particles(bin_index, particle_index, n);
		cudaUnbindTexture(bin_index_tex);
		cudaUnbindTexture(particle_index_tex);
/*
		cudaMemcpy(bin_index_host, bin_index, n * sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(particle_index_host, particle_index, n * sizeof(int), cudaMemcpyDeviceToHost);
		for(int i=0; i<n; ++i){
			printf("0.5.bin_index[%d]=%d,particle_index[%d]=%d,d_pos[%d].x=%f,d_pos[%d].y=%f\n",i,bin_index_host[i],i,particle_index_host[i],i,d_pos_host[2*i],i,d_pos_host[2*i+1]);
		}
*/
		cudaMemset(bin_start, 0xffffffff, num_bins * sizeof(int));
		int smemSize = sizeof(int)*(NUM_THREADS+1);
		reorder_data_calc_bin <<< blks, NUM_THREADS, smemSize >>> (bin_start, bin_end, sorted_pos, sorted_vel, sorted_acc, bin_index, particle_index, d_pos, d_vel, d_acc, n, num_bins);
/*
		cudaMemcpy(bin_index_host, bin_index, n * sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(particle_index_host, particle_index, n * sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(sorted_pos_host, sorted_pos, 2*n * sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(sorted_vel_host, sorted_vel, 2*n * sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(sorted_acc_host, sorted_acc, 2*n * sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(d_pos_host, d_pos, 2*n * sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(d_vel_host, d_vel, 2*n * sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(d_acc_host, d_acc, 2*n * sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(bin_start_host, bin_start, num_bins * sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(bin_end_host, bin_end, num_bins * sizeof(int), cudaMemcpyDeviceToHost);
		for(int i=0; i<n; ++i){
			printf("2.bin_start[%d]=%d,bin_end[%d]=%d\n",i,bin_start_host[i],i,bin_end_host[i]);
			printf("2.d_pos[%d].x=%f,d_pos[%d].y=%f,d_vel[%d].x=%f,d_vel[%d].y=%f,d_acc[%d].x=%f,d_acc[%d].y=%f\n",i,d_pos_host[2*i],i,d_pos_host[2*i+1],i,d_vel_host[2*i],i,d_vel_host[2*i+1],i,d_acc_host[2*i],i,d_acc_host[2*i+1]);
			printf("2.sorted_pos[%d].x=%f,sorted_pos[%d].y=%f,sorted_vel[%d].x=%f,sorted_vel[%d].y=%f,sorted_acc[%d].x=%f,sorted_acc[%d].y=%f\n",i,sorted_pos_host[2*i],i,sorted_pos_host[2*i+1],i,sorted_vel_host[2*i],i,sorted_vel_host[2*i+1],i,sorted_acc_host[2*i],i,sorted_acc_host[2*i+1]);
		}
*/
		cudaBindTexture(0, old_pos_tex, sorted_pos, 2*n * sizeof(int2));
		cudaBindTexture(0, bin_start_tex, bin_start, num_bins * sizeof(int));
		cudaBindTexture(0, bin_end_tex, bin_end, num_bins * sizeof(int));

		compute_forces_gpu <<< blks, NUM_THREADS >>> (sorted_pos, sorted_acc, n, bpr, bin_start, bin_end);

		cudaUnbindTexture(old_pos_tex);
		cudaUnbindTexture(bin_start_tex);
		cudaUnbindTexture(bin_end_tex);
/*
		cudaMemcpy(sorted_pos_host, sorted_pos, 2*n * sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(sorted_vel_host, sorted_vel, 2*n * sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(sorted_acc_host, sorted_acc, 2*n * sizeof(double), cudaMemcpyDeviceToHost);
		for(int i=0; i<n; ++i){
			printf("3.sorted_pos[%d].x=%f,sorted_pos[%d].y=%f,sorted_vel[%d].x=%f,sorted_vel[%d].y=%f,sorted_acc[%d].x=%f,sorted_acc[%d].y=%f\n",i,sorted_pos_host[2*i],i,sorted_pos_host[2*i+1],i,sorted_vel_host[2*i],i,sorted_vel_host[2*i+1],i,sorted_acc_host[2*i],i,sorted_acc_host[2*i+1]);
		}
*/

		//
		//  move particles
		//
		move_gpu <<< blks, NUM_THREADS >>> (sorted_pos, sorted_vel, sorted_acc, n, size);
/*
		cudaMemcpy(sorted_pos_host, sorted_pos, 2*n * sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(sorted_vel_host, sorted_vel, 2*n * sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(sorted_acc_host, sorted_acc, 2*n * sizeof(double), cudaMemcpyDeviceToHost);
		for(int i=0; i<n; ++i){
			printf("4.sorted_pos[%d].x=%f,sorted_pos[%d].y=%f,sorted_vel[%d].x=%f,sorted_vel[%d].y=%f,sorted_acc[%d].x=%f,sorted_acc[%d].y=%f\n",i,sorted_pos_host[2*i],i,sorted_pos_host[2*i+1],i,sorted_vel_host[2*i],i,sorted_vel_host[2*i+1],i,sorted_acc_host[2*i],i,sorted_acc_host[2*i+1]);
		}
*/
		//
		// Swap particles between d_particles and sorted_particles
		//
//        	particle_t *temp = sorted_particles;
//        	sorted_particles = d_particles;
//        	d_particles = temp;
		double *temp_pos = sorted_pos;
		double *temp_vel = sorted_vel;
		double *temp_acc = sorted_acc;
		sorted_pos = d_pos;
		sorted_vel = d_vel;
		sorted_acc = d_acc;
		d_pos = temp_pos;
		d_vel = temp_vel;
		d_acc = temp_acc;

	}

	cudaThreadSynchronize();

	simulation_time = read_timer( ) - simulation_time;

	particle_t *particles = (particle_t*) malloc( n * sizeof(particle_t) );
	if( fsave ) {
		// Copy the particles back to the CPU
		cudaMemcpy(pos, d_pos, 2*n * sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(vel, d_vel, 2*n * sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(acc, d_acc, 2*n * sizeof(double), cudaMemcpyDeviceToHost);
		for(int i=0; i<n; ++i){
			particles[i].x  = pos[2*i];particles[i].y  = pos[2*i+1];
			particles[i].vx = vel[2*i];particles[i].vy = vel[2*i+1];
			particles[i].ax = acc[2*i];particles[i].ay = acc[2*i+1];
		}
		save( fsave, n, particles);
	}

	printf( "CPU-GPU copy time = %g seconds\n", copy_time);
	printf( "n = %d, simulation time = %g seconds\n", n, simulation_time );

	if (fsum)
		fprintf(fsum,"%d %lf \n",n,simulation_time);

	if (fsum)
		fclose( fsum );    
	free( particles );
//	cudaFree(d_particles);
//	cudaFree(sorted_particles);
	free(pos);
	free(vel);
	free(acc);
	free(d_pos_host);
	free(d_vel_host);
	free(d_acc_host);
	free(bin_start_host);
	free(bin_end_host);
	free(sorted_pos_host);
	free(sorted_vel_host);
	free(sorted_acc_host);
	free(bin_index_host);
	free(particle_index_host);
	cudaFree(d_pos);
	cudaFree(d_vel);
	cudaFree(d_acc);
	cudaFree(sorted_pos);
	cudaFree(sorted_vel);
	cudaFree(sorted_acc);
	cudaFree(bin_index);
	cudaFree(particle_index);
	cudaFree(bin_start);
	cudaFree(bin_end);

	if( fsave )
		fclose( fsave );

	return 0;
}
