#include "mex.h"
#include <cmath>
#include <cstddef>
#include <vector>

// Same alternating idx_good / idx_bad rule as NAOMi's sampleDenseNeurons.
// The location draw is performed in compiled code to avoid allocating and
// scanning several full 3D Octave arrays for every placed cell.
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  if (nrhs != 5 || nlhs != 2 || !mxIsLogical(prhs[0]) ||
      !mxIsDouble(prhs[1]) || !mxIsDouble(prhs[2]) ||
      !mxIsDouble(prhs[3]) || !mxIsDouble(prhs[4])) {
    mexErrMsgTxt("Expected logical allowed 3D grid, double XY density, double volume size, double minimum distance, double uniform draws; two outputs.");
  }
  const mwSize *dims = mxGetDimensions(prhs[0]);
  if (mxGetNumberOfDimensions(prhs[0]) != 3) {
    mexErrMsgTxt("Allowed grid must be 3D.");
  }
  const mwSize nx = dims[0], ny = dims[1], nz = dims[2];
  const mwSize nxy = nx * ny;
  const mwSize nxyz = nxy * nz;
  if (nx < 2 || ny < 2 || nz < 2 ||
      mxGetNumberOfElements(prhs[1]) != nxy ||
      mxGetNumberOfElements(prhs[2]) != 3 ||
      mxGetNumberOfElements(prhs[3]) != 1) {
    mexErrMsgTxt("Grid, XY density, or volume dimensions mismatch.");
  }
  const double *density = mxGetPr(prhs[1]);
  const double *size_um = mxGetPr(prhs[2]);
  const double min_dist = mxGetScalar(prhs[3]);
  const double *draws = mxGetPr(prhs[4]);
  const mwSize wanted = mxGetNumberOfElements(prhs[4]);
  if (!(min_dist > 0)) mexErrMsgTxt("Minimum distance must be positive.");

  const mxLogical *initial = mxGetLogicals(prhs[0]);
  std::vector<unsigned char> good(nxyz), bad(nxyz);
  mwSize good_count = 0;
  for (mwSize i = 0; i < nxyz; ++i) {
    good[i] = bad[i] = initial[i] ? 1 : 0;
    good_count += good[i];
  }
  std::vector<float> xcoord(nx), ycoord(ny), zcoord(nz);
  for (mwSize i = 0; i < nx; ++i)
    xcoord[i] = static_cast<float>(size_um[0] * static_cast<double>(i) / (nx - 1));
  for (mwSize i = 0; i < ny; ++i)
    ycoord[i] = static_cast<float>(size_um[1] * static_cast<double>(i) / (ny - 1));
  for (mwSize i = 0; i < nz; ++i)
    zcoord[i] = static_cast<float>(size_um[2] * static_cast<double>(i) / (nz - 1));

  plhs[0] = mxCreateDoubleMatrix(wanted, 3, mxREAL);
  plhs[1] = mxCreateDoubleScalar(0);
  double *locations = mxGetPr(plhs[0]);
  mwSize placed = 0;
  const float outer2 = static_cast<float>(1.1 * min_dist * 1.1 * min_dist);
  const float inner2 = static_cast<float>(min_dist * min_dist);
  while (placed < wanted && good_count > 1) {
    double total = 0;
    for (mwSize i = 0; i < nxyz; ++i)
      if (good[i]) total += density[i % nxy];
    if (!(total > 0)) break;
    const double target = draws[placed] * total;
    double cumulative = 0;
    mwSize selected = nxyz;
    for (mwSize i = 0; i < nxyz; ++i) {
      if (!good[i]) continue;
      cumulative += density[i % nxy];
      if (cumulative >= target) { selected = i; break; }
    }
    if (selected == nxyz) break;
    const mwSize ix = selected % nx;
    const mwSize iy = (selected / nx) % ny;
    const mwSize iz = selected / nxy;
    const float cx = xcoord[ix], cy = ycoord[iy], cz = zcoord[iz];
    locations[placed] = cx;
    locations[placed + wanted] = cy;
    locations[placed + 2 * wanted] = cz;
    ++placed;

    good_count = 0;
    mwSize linear = 0;
    for (mwSize z = 0; z < nz; ++z) {
      const float dz = zcoord[z] - cz;
      for (mwSize y = 0; y < ny; ++y) {
        const float dy = ycoord[y] - cy;
        for (mwSize x = 0; x < nx; ++x, ++linear) {
          const float dx = xcoord[x] - cx;
          const float dist2 = dx * dx + dy * dy + dz * dz;
          if (dist2 <= outer2) good[linear] = 0;
          if (dist2 <= inner2) bad[linear] = 0;
          good[linear] = (!good[linear] && bad[linear]) ? 1 : 0;
          good_count += good[linear];
        }
      }
    }
  }
  *mxGetPr(plhs[1]) = static_cast<double>(placed);
}
