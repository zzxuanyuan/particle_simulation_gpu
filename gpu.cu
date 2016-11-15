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
//
//  benchmarking program
//
/*
void swap(particle_t *sorted_particles, particle_t *d_particles)
{
	particle_t *temp = sorted_particles;
	sorted_particles = d_particles;
	d_particles = temp;
}
*/
void sort_particles(int *bin_index, int *particle_index, int n)
{
	thrust::sort_by_key(thrust::device_ptr<int>(bin_index),
			thrust::device_ptr<int>(bin_index + n),
			thrust::device_ptr<int>(particle_index));
}

// calculate particle's bin number
__device__ int binNum(particle_t &p, int bpr) 
{
	return ( floor(p.x/cutoff) + bpr*floor(p.y/cutoff) );
}

__global__ void reorder_data_calc_bin(int *bin_start, int *bin_end, particle_t *sorted_particles, int *bin_index, int *particle_index, particle_t *d_particles, int n, int num_bins)
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
		sorted_particles[index].x = d_particles[sortedIndex].x;
		sorted_particles[index].y = d_particles[sortedIndex].y;
		sorted_particles[index].vx = d_particles[sortedIndex].vx;
		sorted_particles[index].vy = d_particles[sortedIndex].vy;
		sorted_particles[index].ax = d_particles[sortedIndex].ax;
		sorted_particles[index].ay = d_particles[sortedIndex].ay;
	}
}

__global__ void calculate_bin_index(int *bin_index, int *particle_index, particle_t * particles, int n, int bpr)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if(index >= n) return;
	int cbin = binNum( particles[index], bpr );
	bin_index[index] = cbin;
	particle_index[index] = index;
}

__device__ void apply_force_gpu(particle_t &particle, particle_t &neighbor)
{
	double dx = neighbor.x - particle.x;
	double dy = neighbor.y - particle.y;
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
	particle.ax += coef * dx;
	particle.ay += coef * dy;
}

__global__ void compute_forces_gpu(particle_t * particles, int n, int bpr, int *bin_start, int *bin_end)
{
	// Get thread (particle) ID
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if(tid >= n) return;

	// find current particle's in, handle boundaries
	int cbin = binNum( particles[tid], bpr );
	int lowi = -1, highi = 1, lowj = -1, highj = 1;
	if (cbin < bpr)
		lowj = 0;
	if (cbin % bpr == 0)
		lowi = 0;
	if (cbin % bpr == (bpr-1))
		highi = 0;
	if (cbin >= bpr*(bpr-1))
		highj = 0;

	particles[tid].ax = particles[tid].ay = 0;

	for (int i = lowi; i <= highi; i++)
		for (int j = lowj; j <= highj; j++)
		{
			int nbin = cbin + i + bpr*j;
			for (int k = bin_start[nbin]; k < bin_end[nbin]; k++ ) {
				if (bin_start[nbin] >= 0) {
					apply_force_gpu( particles[tid], particles[k] );
				}
			}
		}
}

__global__ void move_gpu (particle_t * particles, int n, double size)
{

	// Get thread (particle) ID
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if(tid >= n) return;

	particle_t * p = &particles[tid];
	//
	//  slightly simplified Velocity Verlet integration
	//  conserves energy better than explicit Euler method
	//
	p->vx += p->ax * dt;
	p->vy += p->ay * dt;
	p->x  += p->vx * dt;
	p->y  += p->vy * dt;

	//
	//  bounce from walls
	//
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
	particle_t *particles = (particle_t*) malloc( n * sizeof(particle_t) );

	// GPU particle data structure
	particle_t * d_particles;
	cudaMalloc((void **) &d_particles, n * sizeof(particle_t));
	particle_t * sorted_particles;
	cudaMalloc((void **) &sorted_particles, n * sizeof(particle_t));
	int *bin_index;
	cudaMalloc((void **) &bin_index, n * sizeof(int));
	cudaMemset(bin_index, 0x0, n * sizeof(int));
	int *particle_index;
	cudaMalloc((void **) &particle_index, n * sizeof(int));
	cudaMemset(particle_index, 0x0, n * sizeof(int));

	set_size( n );

	init_particles( n, particles );

	// create spatial bins (of size cutoff by cutoff)
	double size = sqrt( density*n );
	int bpr = ceil(size/cutoff);
	int num_bins = bpr*bpr;
	printf("n=%d, size=%f, bpr=%d, num_bins=%d\n",n,size,bpr,num_bins);//##
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
	cudaMemcpy(d_particles, particles, n * sizeof(particle_t), cudaMemcpyHostToDevice);

	cudaThreadSynchronize();
	copy_time = read_timer( ) - copy_time;

	//
	//  simulate a number of time steps
	//
	cudaThreadSynchronize();
	double simulation_time = read_timer( );
/*
	int *bin_index_host = (int *)malloc(n * sizeof(int));//##
	int *particle_index_host = (int *)malloc(n * sizeof(int));//##
	double *r2_host = (double *)malloc(n * sizeof(double));//##
	double *coef_host = (double *)malloc(n * sizeof(double));//##
	int *bin_start_host = (int *)malloc(num_bins * sizeof(int));//##
	int *bin_end_host = (int *)malloc(num_bins * sizeof(int));//##
	particle_t *d_particles_host = (particle_t *)malloc(n * sizeof(particle_t));//##
	particle_t *sorted_particles_host = (particle_t *)malloc(n * sizeof(particle_t));//##
*/
	for( int step = 0; step < NSTEPS; step++ )
	{

		// clear bins at each time step
		//		for (int m = 0; m < numbins; m++)
		//			bins[m].clear();

		// place particles in bins
		//		for (int i = 0; i < n; i++)
		//			bins[binNum(particles[i],bpr)].push_back(particles + i);

		//
		//  compute forces
		//

		//		cudaMemcpy(d_bins, bins, numbins * sizeof(thrust::host_vector<particle_t*>), cudaMemcpyHostToDevice);
		int blks = (n + NUM_THREADS - 1) / NUM_THREADS;

		calculate_bin_index <<< blks, NUM_THREADS >>> (bin_index, particle_index, d_particles, n, bpr);
//		cudaMemcpy(bin_index_host, bin_index, n * sizeof(int), cudaMemcpyDeviceToHost);//##
//		cudaMemcpy(particle_index_host, particle_index, n * sizeof(int), cudaMemcpyDeviceToHost);//##
/*
		for(int i = 0; i < n; ++i) {
			printf("1.bin_index_host[%d]=%d, particle_index[%d]=%d\n",i,bin_index_host[i],i,particle_index_host[i]);//##
		}
*/
		sort_particles(bin_index, particle_index, n);

//		cudaMemcpy(bin_index_host, bin_index, n * sizeof(int), cudaMemcpyDeviceToHost);//##
//		cudaMemcpy(particle_index_host, particle_index, n * sizeof(int), cudaMemcpyDeviceToHost);//##
/*
		for(int i = 0; i < n; ++i) {
			printf("2.bin_index_host[%d]=%d, particle_index[%d]=%d\n",i,bin_index_host[i],i,particle_index_host[i]);//##
		}
*/
//		cudaMemcpy(d_particles_host, d_particles, n * sizeof(particle_t), cudaMemcpyDeviceToHost);//##
/*
		for(int i = 0; i < n; ++i) {
			printf("d_particles_host[%d]=%f,%f,%f,%f,%f,%f(x,y,vx,vy,ax,ay)\n",i,d_particles_host[i].x,d_particles_host[i].y,d_particles_host[i].vx,d_particles_host[i].vy,d_particles_host[i].ax,d_particles_host[i].ay);//##
		}
*/
		cudaMemset(bin_start, 0xffffffff, num_bins * sizeof(int));
		int smemSize = sizeof(int)*(NUM_THREADS+1);
		reorder_data_calc_bin <<< blks, NUM_THREADS, smemSize >>> (bin_start, bin_end, sorted_particles, bin_index, particle_index, d_particles, n, num_bins);

//		cudaMemcpy(bin_start_host, bin_start, num_bins * sizeof(int), cudaMemcpyDeviceToHost);//##
//		cudaMemcpy(bin_end_host, bin_end, num_bins * sizeof(int), cudaMemcpyDeviceToHost);//##
/*
		for(int i = 0; i < num_bins; ++i) {
			printf("bin_start_host[%d]=%d, bin_end_host[%d]=%d\n",i,bin_start_host[i],i,bin_end_host[i]);//##
		}
*/
//		cudaMemcpy(sorted_particles_host, sorted_particles, n * sizeof(particle_t), cudaMemcpyDeviceToHost);//##
/*
		for(int i = 0; i < n; ++i) {
			printf("1.sorted_particles_host[%d]=%f,%f,%f,%f,%f,%f(x,y,vx,vy,ax,ay)\n",i,sorted_particles_host[i].x,sorted_particles_host[i].y,sorted_particles_host[i].vx,sorted_particles_host[i].vy,sorted_particles_host[i].ax,sorted_particles_host[i].ay);//##
		}
*/
		compute_forces_gpu <<< blks, NUM_THREADS >>> (sorted_particles, n, bpr, bin_start, bin_end);

//		cudaMemcpy(sorted_particles_host, sorted_particles, n * sizeof(particle_t), cudaMemcpyDeviceToHost);//##
/*
		for(int i = 0; i < n; ++i) {
			printf("2.sorted_particles_host[%d]=%f,%f,%f,%f,%f,%f(x,y,vx,vy,ax,ay)\n",i,sorted_particles_host[i].x,sorted_particles_host[i].y,sorted_particles_host[i].vx,sorted_particles_host[i].vy,sorted_particles_host[i].ax,sorted_particles_host[i].ay);//##
		}
*/
//		cudaMemcpy(r2_host, r2, n * sizeof(double), cudaMemcpyDeviceToHost);//##
/*
		for(int i = 0; i < n; ++i) {
			printf("r2_host[%d]=%f\n",i,r2_host[i]);//##
		}
*/
//		cudaMemcpy(coef_host, coef, n * sizeof(double), cudaMemcpyDeviceToHost);//##
/*
		for(int i = 0; i < n; ++i) {
			printf("coef_host[%d]=%f\n",i,coef_host[i]);//##
		}
*/

		//
		//  move particles
		//
		move_gpu <<< blks, NUM_THREADS >>> (sorted_particles, n, size);

//		cudaMemcpy(sorted_particles_host, sorted_particles, n * sizeof(particle_t), cudaMemcpyDeviceToHost);//##
/*
		for(int i = 0; i < n; ++i) {
			printf("3.sorted_particles_host[%d]=%f,%f,%f,%f,%f,%f(x,y,vx,vy,ax,ay)\n",i,sorted_particles_host[i].x,sorted_particles_host[i].y,sorted_particles_host[i].vx,sorted_particles_host[i].vy,sorted_particles_host[i].ax,sorted_particles_host[i].ay);//##
		}
*/

//		swap(d_particles, sorted_particles);
		//
		// Swap particles between d_particles and sorted_particles
		//
        	particle_t *temp = sorted_particles;
        	sorted_particles = d_particles;
        	d_particles = temp;

		//
		//  save if necessary
		//
/*
		if( fsave && (step%SAVEFREQ) == 0 ) {
			// Copy the particles back to the CPU
			cudaMemcpy(particles, d_particles, n * sizeof(particle_t), cudaMemcpyDeviceToHost);
			save( fsave, n, particles);
		}
*/
	}
	if( fsave ) {
		// Copy the particles back to the CPU
		cudaMemcpy(particles, d_particles, n * sizeof(particle_t), cudaMemcpyDeviceToHost);
		save( fsave, n, particles);
	}

	cudaThreadSynchronize();
	simulation_time = read_timer( ) - simulation_time;

	printf( "CPU-GPU copy time = %g seconds\n", copy_time);
	printf( "n = %d, simulation time = %g seconds\n", n, simulation_time );

	if (fsum)
		fprintf(fsum,"%d %lf \n",n,simulation_time);

	if (fsum)
		fclose( fsum );    
	free( particles );
	cudaFree(d_particles);
	cudaFree(sorted_particles);
	cudaFree(bin_index);
	cudaFree(particle_index);
	cudaFree(bin_start);
	cudaFree(bin_end);

	if( fsave )
		fclose( fsave );

	return 0;
}
