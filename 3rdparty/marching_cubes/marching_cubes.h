#ifndef MARCHING_CUBES_H
#define MARCHING_CUBES_H

#include <map>
#include <vector>

typedef float Point[3];
typedef unsigned int index_t;

struct IdPoint
{
	index_t id;
	float x, y, z;
};

typedef std::map<index_t, IdPoint> PointIdMapping;

struct Triangle
{
	index_t pointId[3];
};

/**
 * the primary struct used to pass around the components of a mesh
 * vertices:    the vertex positions as an array of points
 * normals:     the normal direction of each vertex as an array of points
 * vertexCount: the number of vertices and normals
 * faces:       the faces given by 3 vertex indices (length = faceCount * 3)
 * faceCount:   the number of faces
 */
struct Mesh
{
	Point offset;
	index_t vertexCount;
	Point* vertices;
	Point* normals;
	index_t faceCount;
	index_t* faces;

	Mesh(index_t, Point*, Point*, index_t, index_t*);
	Mesh();
};

/**
 * the marching cubes algorithm as described here: http://paulbourke.net/geometry/polygonise/
 * volume:   contains the data (size = xDim * yDim * zDim)
 * [xyz]Dim: the dimensions in each direction
 * isoLevel: the minimum isoLevel, all values >= isoLevel will contribute to the mesh
 * the mesh is returned, the caller takes ownership over the pointers
 */
template<typename T>
Mesh march(const T* volume, index_t xDim, index_t yDim, index_t zDim, T isoLevel);
#endif