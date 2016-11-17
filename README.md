////////////////////////////////////////////////////////////////////////
///                                                                  ///	
///  Assignment 3. Particle Simulation with CUDA                     ///
///            Author: Zhe Zhang                                     ///
////////////////////////////////////////////////////////////////////////

1. I used the second method mentioned in [1]. The algorithm sorts all particles based on their bin indices each iteration.

2. Texture memory is designed to optimize spatial locality. Therefore, sorting particles can help texture memory determine
   the access pattern and automatically optimize performance.

3. Since the grid is very sparse with the given density, we only need to query texture memory when the requested bin is
   non-empty. This trick gives me a performance boost.

Reference:
[1] Particle Simulation using CUDA. Simon Green.
