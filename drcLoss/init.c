#include "TH.h"
#include "luaT.h"

// For the Tensor functions
#include "rayPotSem.h"
#include "rayPotColor.h"
#include "mleDepthSem.h"
#include "rayPotGeom.h"
#include "rayFusion.h"
#include "grid.h"

static int computeRayFusion(lua_State *L){
    //printf("Starting Computation\n");
    THDoubleTensor *origins = luaT_checkudata(L, 2, "torch.DoubleTensor");
    THDoubleTensor *directions = luaT_checkudata(L, 3, "torch.DoubleTensor");
    THDoubleTensor *depths = luaT_checkudata(L, 4, "torch.DoubleTensor");

    THDoubleTensor *emptyCount = luaT_getfieldcheckudata(L, 1, "emptyCount", "torch.DoubleTensor");
    THDoubleTensor *occCount = luaT_getfieldcheckudata(L, 1, "occCount", "torch.DoubleTensor");
    THIntTensor *useProj = luaT_getfieldcheckudata(L, 1, "useProj", "torch.IntTensor");
    THIntTensor *maskOnly = luaT_getfieldcheckudata(L, 1, "maskOnly", "torch.IntTensor");

    THDoubleTensor *minBounds = luaT_getfieldcheckudata(L, 1, "minBounds", "torch.DoubleTensor");
    THDoubleTensor *maxBounds = luaT_getfieldcheckudata(L, 1, "maxBounds", "torch.DoubleTensor");
    THDoubleTensor *focal = luaT_getfieldcheckudata(L, 1, "focal", "torch.DoubleTensor");
    //printf("Declared Vars\n");

    int batchsize = origins->size[0];
    int nRays = origins->size[1];
    int nDims = origins->size[2];
    //printf("Problem Size (%d, %d, %d)\n",batchsize, nRays, nDims);

    grid* g = grid_init(nDims, THDoubleTensor_data(minBounds), THDoubleTensor_data(maxBounds), THDoubleTensor_data(focal));
    //printf("Grid init, nCells = %d\n",grid_nCells(g));

    int* useProj_data = THIntTensor_data(useProj);
    int* useMask_data = THIntTensor_data(maskOnly);
    rpFuse* rp = rpFuse_init(g, useProj_data[0]);

    //printf("Initialized rp\n");
    rpFuse_forward(rp, THDoubleTensor_data(emptyCount), THDoubleTensor_data(occCount), THDoubleTensor_data(origins), THDoubleTensor_data(directions), THDoubleTensor_data(depths), batchsize, nRays, useMask_data[0]);

    free(g);free(rp);
    return 0;
}

static int computeRpGeom(lua_State *L){
    //printf("Starting Computation\n");
    THDoubleTensor *origins = luaT_checkudata(L, 2, "torch.DoubleTensor");
    THDoubleTensor *directions = luaT_checkudata(L, 3, "torch.DoubleTensor");
    THDoubleTensor *depths = luaT_checkudata(L, 4, "torch.DoubleTensor");

    THDoubleTensor *predsGeom = luaT_getfieldcheckudata(L, 1, "predsGeom", "torch.DoubleTensor");
    THDoubleTensor *gradPredsGeom = luaT_getfieldcheckudata(L, 1, "gradPredsGeom", "torch.DoubleTensor");
    THDoubleTensor *E_psi = luaT_getfieldcheckudata(L, 1, "E_psi", "torch.DoubleTensor");
    THDoubleTensor *maxDepth = luaT_getfieldcheckudata(L, 1, "maxDepth", "torch.DoubleTensor");
    THIntTensor *useProj = luaT_getfieldcheckudata(L, 1, "useProj", "torch.IntTensor");
    THIntTensor *maskOnly = luaT_getfieldcheckudata(L, 1, "maskOnly", "torch.IntTensor");

    THDoubleTensor *minBounds = luaT_getfieldcheckudata(L, 1, "minBounds", "torch.DoubleTensor");
    THDoubleTensor *maxBounds = luaT_getfieldcheckudata(L, 1, "maxBounds", "torch.DoubleTensor");
    THDoubleTensor *focal = luaT_getfieldcheckudata(L, 1, "focal", "torch.DoubleTensor");
    //printf("Declared Vars\n");

    int batchsize = origins->size[0];
    int nRays = origins->size[1];
    int nDims = origins->size[2];
    //printf("Problem Size (%d, %d, %d, %d)\n",batchsize, nRays, nDims, nClasses);

    grid* g = grid_init(nDims, THDoubleTensor_data(minBounds), THDoubleTensor_data(maxBounds), THDoubleTensor_data(focal));
    //printf("Grid init, nCells = %d\n",grid_nCells(g));

    double* maxD_data = THDoubleTensor_data(maxDepth);
    int* useProj_data = THIntTensor_data(useProj);
    int* useMask_data = THIntTensor_data(maskOnly);
    rpGeom* rp = rpGeom_init(g, maxD_data[0], useProj_data[0]);

    //printf("Initialized rp\n");
    rpGeom_forward(rp, THDoubleTensor_data(predsGeom), THDoubleTensor_data(gradPredsGeom), THDoubleTensor_data(E_psi), THDoubleTensor_data(origins), THDoubleTensor_data(directions), THDoubleTensor_data(depths), batchsize, nRays, useMask_data[0]);

    free(g);free(rp);
    return 0;
}

static int computeRpGeomWeighted(lua_State *L){
    //printf("Starting Computation\n");
    THDoubleTensor *origins = luaT_checkudata(L, 2, "torch.DoubleTensor");
    THDoubleTensor *directions = luaT_checkudata(L, 3, "torch.DoubleTensor");
    THDoubleTensor *depths = luaT_checkudata(L, 4, "torch.DoubleTensor");
    THDoubleTensor *weights = luaT_checkudata(L, 5, "torch.DoubleTensor");

    THDoubleTensor *predsGeom = luaT_getfieldcheckudata(L, 1, "predsGeom", "torch.DoubleTensor");
    THDoubleTensor *gradPredsGeom = luaT_getfieldcheckudata(L, 1, "gradPredsGeom", "torch.DoubleTensor");
    THDoubleTensor *E_psi = luaT_getfieldcheckudata(L, 1, "E_psi", "torch.DoubleTensor");
    THDoubleTensor *maxDepth = luaT_getfieldcheckudata(L, 1, "maxDepth", "torch.DoubleTensor");
    THIntTensor *useProj = luaT_getfieldcheckudata(L, 1, "useProj", "torch.IntTensor");
    THIntTensor *maskOnly = luaT_getfieldcheckudata(L, 1, "maskOnly", "torch.IntTensor");

    THDoubleTensor *minBounds = luaT_getfieldcheckudata(L, 1, "minBounds", "torch.DoubleTensor");
    THDoubleTensor *maxBounds = luaT_getfieldcheckudata(L, 1, "maxBounds", "torch.DoubleTensor");
    THDoubleTensor *focal = luaT_getfieldcheckudata(L, 1, "focal", "torch.DoubleTensor");
    //printf("Declared Vars\n");

    int batchsize = origins->size[0];
    int nRays = origins->size[1];
    int nDims = origins->size[2];
    //printf("Problem Size (%d, %d, %d, %d)\n",batchsize, nRays, nDims, nClasses);

    grid* g = grid_init(nDims, THDoubleTensor_data(minBounds), THDoubleTensor_data(maxBounds), THDoubleTensor_data(focal));
    //printf("Grid init, nCells = %d\n",grid_nCells(g));

    double* maxD_data = THDoubleTensor_data(maxDepth);
    int* useProj_data = THIntTensor_data(useProj);
    int* useMask_data = THIntTensor_data(maskOnly);
    rpGeom* rp = rpGeom_init(g, maxD_data[0], useProj_data[0]);

    //printf("Initialized rp\n");
    rpGeom_weighted_forward(rp, THDoubleTensor_data(predsGeom), THDoubleTensor_data(gradPredsGeom), THDoubleTensor_data(E_psi), THDoubleTensor_data(origins), THDoubleTensor_data(directions), THDoubleTensor_data(depths), THDoubleTensor_data(weights), batchsize, nRays, useMask_data[0]);

    free(g);free(rp);
    return 0;
}

static int computeRpSem(lua_State *L){
    //printf("Starting Computation\n");
    THDoubleTensor *origins = luaT_checkudata(L, 2, "torch.DoubleTensor");
    THDoubleTensor *directions = luaT_checkudata(L, 3, "torch.DoubleTensor");
    THDoubleTensor *depths = luaT_checkudata(L, 4, "torch.DoubleTensor");
    THIntTensor *classIds = luaT_checkudata(L, 5, "torch.IntTensor");

    THDoubleTensor *predsGeom = luaT_getfieldcheckudata(L, 1, "predsGeom", "torch.DoubleTensor");
    THDoubleTensor *predsSem = luaT_getfieldcheckudata(L, 1, "predsSem", "torch.DoubleTensor");
    THDoubleTensor *gradPredsGeom = luaT_getfieldcheckudata(L, 1, "gradPredsGeom", "torch.DoubleTensor");
    THDoubleTensor *gradPredsSem = luaT_getfieldcheckudata(L, 1, "gradPredsSem", "torch.DoubleTensor");
    THDoubleTensor *E_psi = luaT_getfieldcheckudata(L, 1, "E_psi", "torch.DoubleTensor");
    THDoubleTensor *maxDepth = luaT_getfieldcheckudata(L, 1, "maxDepth", "torch.DoubleTensor");
    THIntTensor *useProj = luaT_getfieldcheckudata(L, 1, "useProj", "torch.IntTensor");
    THDoubleTensor *bgDist = luaT_getfieldcheckudata(L, 1, "bgDist", "torch.DoubleTensor");

    THDoubleTensor *minBounds = luaT_getfieldcheckudata(L, 1, "minBounds", "torch.DoubleTensor");
    THDoubleTensor *maxBounds = luaT_getfieldcheckudata(L, 1, "maxBounds", "torch.DoubleTensor");
    THDoubleTensor *focal = luaT_getfieldcheckudata(L, 1, "focal", "torch.DoubleTensor");
    //printf("Declared Vars\n");

    int batchsize = origins->size[0];
    int nRays = origins->size[1];
    int nDims = origins->size[2];
    int nClasses = bgDist->size[0];
    //printf("Problem Size (%d, %d, %d, %d)\n",batchsize, nRays, nDims, nClasses);

    grid* g = grid_init(nDims, THDoubleTensor_data(minBounds), THDoubleTensor_data(maxBounds), THDoubleTensor_data(focal));
    //printf("Grid init, nCells = %d\n",grid_nCells(g));

    double* maxD_data = THDoubleTensor_data(maxDepth);
    int* useProj_data = THIntTensor_data(useProj);
    rpSem* rp = rpSem_init(g, maxD_data[0], nClasses, THDoubleTensor_data(bgDist), useProj_data[0]);

    //printf("Initialized rp\n");
    rpSem_forward(rp, THDoubleTensor_data(predsGeom), THDoubleTensor_data(gradPredsGeom), THDoubleTensor_data(predsSem), THDoubleTensor_data(gradPredsSem), THDoubleTensor_data(E_psi), THDoubleTensor_data(origins), THDoubleTensor_data(directions), THDoubleTensor_data(depths), THIntTensor_data(classIds), batchsize, nRays);
    free(g);free(rp);

    return 0;
}

static int computeRpColor(lua_State *L){
    //printf("Starting Computation\n");
    THDoubleTensor *origins = luaT_checkudata(L, 2, "torch.DoubleTensor");
    THDoubleTensor *directions = luaT_checkudata(L, 3, "torch.DoubleTensor");
    THDoubleTensor *colors = luaT_checkudata(L, 4, "torch.DoubleTensor");

    THDoubleTensor *predsGeom = luaT_getfieldcheckudata(L, 1, "predsGeom", "torch.DoubleTensor");
    THDoubleTensor *predsColor = luaT_getfieldcheckudata(L, 1, "predsColor", "torch.DoubleTensor");
    THDoubleTensor *gradPredsGeom = luaT_getfieldcheckudata(L, 1, "gradPredsGeom", "torch.DoubleTensor");
    THDoubleTensor *gradPredsColor = luaT_getfieldcheckudata(L, 1, "gradPredsColor", "torch.DoubleTensor");
    THDoubleTensor *E_psi = luaT_getfieldcheckudata(L, 1, "E_psi", "torch.DoubleTensor");
    THIntTensor *useProj = luaT_getfieldcheckudata(L, 1, "useProj", "torch.IntTensor");
    THDoubleTensor *bgColor = luaT_getfieldcheckudata(L, 1, "bgColor", "torch.DoubleTensor");

    THDoubleTensor *minBounds = luaT_getfieldcheckudata(L, 1, "minBounds", "torch.DoubleTensor");
    THDoubleTensor *maxBounds = luaT_getfieldcheckudata(L, 1, "maxBounds", "torch.DoubleTensor");
    THDoubleTensor *focal = luaT_getfieldcheckudata(L, 1, "focal", "torch.DoubleTensor");
    //printf("Declared Vars\n");

    int batchsize = origins->size[0];
    int nRays = origins->size[1];
    int nDims = origins->size[2];
    //printf("Problem Size (%d, %d, %d, %d)\n",batchsize, nRays, nDims, nClasses);

    grid* g = grid_init(nDims, THDoubleTensor_data(minBounds), THDoubleTensor_data(maxBounds), THDoubleTensor_data(focal));
    //printf("Grid init, nCells = %d\n",grid_nCells(g));

    int* useProj_data = THIntTensor_data(useProj);
    rpColor* rp = rpColor_init(g, THDoubleTensor_data(bgColor), useProj_data[0]);

    //printf("Initialized rp\n");
    rpColor_forward(rp, THDoubleTensor_data(predsGeom), THDoubleTensor_data(gradPredsGeom), THDoubleTensor_data(predsColor), THDoubleTensor_data(gradPredsColor), THDoubleTensor_data(E_psi), THDoubleTensor_data(origins), THDoubleTensor_data(directions), THDoubleTensor_data(colors), batchsize, nRays);
    free(g);free(rp);

    return 0;
}

static int computeDepthSemMLE(lua_State *L){
    //printf("Starting Computation\n");
    THDoubleTensor *origins = luaT_checkudata(L, 2, "torch.DoubleTensor");
    THDoubleTensor *directions = luaT_checkudata(L, 3, "torch.DoubleTensor");

    THDoubleTensor *predsGeom = luaT_getfieldcheckudata(L, 1, "predsGeom", "torch.DoubleTensor");
    THIntTensor *predsSem = luaT_getfieldcheckudata(L, 1, "predsSem", "torch.IntTensor");
    THDoubleTensor *inferredDepth = luaT_getfieldcheckudata(L, 1, "inferredDepth", "torch.DoubleTensor");
    THIntTensor *inferredClass = luaT_getfieldcheckudata(L, 1, "inferredClass", "torch.IntTensor");
    THDoubleTensor *maxDepth = luaT_getfieldcheckudata(L, 1, "maxDepth", "torch.DoubleTensor");
    THIntTensor *useProj = luaT_getfieldcheckudata(L, 1, "useProj", "torch.IntTensor");
    //THDoubleTensor *bgDist = luaT_getfieldcheckudata(L, 1, "bgDist", "torch.DoubleTensor");

    THDoubleTensor *minBounds = luaT_getfieldcheckudata(L, 1, "minBounds", "torch.DoubleTensor");
    THDoubleTensor *maxBounds = luaT_getfieldcheckudata(L, 1, "maxBounds", "torch.DoubleTensor");
    THDoubleTensor *focal = luaT_getfieldcheckudata(L, 1, "focal", "torch.DoubleTensor");
    //printf("Declared Vars\n");

    int batchsize = origins->size[0];
    int nRays = origins->size[1];
    int nDims = origins->size[2];
    //int nClasses = bgDist->size[0];
    //printf("Problem Size (%d, %d, %d, %d)\n",batchsize, nRays, nDims, nClasses);

    grid* g = grid_init(nDims, THDoubleTensor_data(minBounds), THDoubleTensor_data(maxBounds), THDoubleTensor_data(focal));
    //printf("Grid init, nCells = %d\n",grid_nCells(g));

    double* maxD_data = THDoubleTensor_data(maxDepth);
    int* useProj_data = THIntTensor_data(useProj);
    mleDepthSem* rp = mleDepthSem_init(g, maxD_data[0], useProj_data[0]);

    //printf("Initialized rp\n");
    mleDepthSem_forward(rp, THDoubleTensor_data(predsGeom), THIntTensor_data(predsSem), THDoubleTensor_data(origins), THDoubleTensor_data(directions), THDoubleTensor_data(inferredDepth), THIntTensor_data(inferredClass), batchsize, nRays);
    free(g);free(rp);

    return 0;
}

static const struct luaL_Reg routines [] = {
    {"computeRpSem", computeRpSem},
    {"computeRpColor", computeRpColor},
    {"computeDepthSemMLE", computeDepthSemMLE},
    {"computeRpGeom", computeRpGeom},
    {"computeRpGeomWeighted", computeRpGeomWeighted},
    {"computeRayFusion", computeRayFusion},
    {NULL, NULL}
};

// This function should be called "luaopen_libmodule" for a module called "module", here "rpSem"
int luaopen_librpsem(lua_State *L)
{
  // Create a new table
  lua_newtable(L);
  // Add your functions as elements of the table
  luaL_register(L, NULL, routines);

  // This function returns one element (the table containing the functions)
  return 1;
}
