#include "GeometryGen.hpp"
#include <iostream>

//following code taken from:
//https://schneide.blog/2016/07/15/generating-an-icosphere-in-c/#:~:text=for%20(%20int%20i=0;,marching%20cubes%20or%20marching%20tetrahedrons.
//except the "make spiky part"


//it returns the index of the newly inserted midpoint vertex of the edge(if insertion succeeds), if it fails, it returns what's already there
//which is a midpoint already inserted before
Index vertex_for_edge(Lookup& lookup,
VertexList& vertices, Index first, Index second)
{
    Lookup::key_type key(first, second);
    if (key.first>key.second){
        std::swap(key.first, key.second);
    }
    //lookup is inserted with the new {edge,midpoint} pair
    auto inserted=lookup.insert({key, (uint32_t) vertices.size()});
    if (inserted.second)
    {
        auto& edge0=vertices[first];
        auto& edge1=vertices[second];
        auto point = normalized(edge0 + edge1);
        vertices.push_back(point);
    }
    return inserted.first->second;
}

TriangleList subdivide(VertexList& vertices,
    TriangleList triangles)
{
    Lookup lookup;
    TriangleList result;

    for (auto&& each:triangles)
    {
        std::array<Index, 3> mid;
        for (int edge=0; edge<3; ++edge)
        {
            mid[edge]=vertex_for_edge(lookup, vertices,
            each.vertex[edge], each.vertex[(edge+1)%3]);
        }

        result.push_back({each.vertex[0], mid[0], mid[2]});
        result.push_back({each.vertex[1], mid[1], mid[0]});
        result.push_back({each.vertex[2], mid[2], mid[1]});
        result.push_back({mid[0], mid[1], mid[2]});
    }

    return result;
}


IndexedMesh make_icosphere(int subdivisions)
{
    VertexList vertices=icosahedron::vertices;
    TriangleList triangles=icosahedron::triangles;

    for (int i=0; i<subdivisions; ++i)
    {
    triangles=subdivide(vertices, triangles);
    }

    return{vertices, triangles};
}

//this is my addition
TriangleList make_spiky(VertexList &vertices, TriangleList triangles){

    TriangleList ret;
    for (auto&& each:triangles)
    {
        vec3 mid = (vertices[each.vertex[0]] + vertices[each.vertex[1]] +vertices[each.vertex[2]])/3;
        //make it stick out
        mid = mid * 1.6f;
        //insert vertex(I don't think there will be overlap so no lookup memorization needed)
        Index mid_ind = (uint32_t)vertices.size();
        vertices.push_back(mid);
        //now insert triangle faces
        ret.push_back({each.vertex[0], each.vertex[1], mid_ind});
        ret.push_back({each.vertex[1], each.vertex[2], mid_ind});
        ret.push_back({each.vertex[2], each.vertex[0], mid_ind});
    }
    return ret;
}

IndexedMesh make_spiky_icosphere(int subdivisions)
{
    VertexList vertices=icosahedron::vertices;
    TriangleList triangles=icosahedron::triangles;

    for (int i=0; i<subdivisions; ++i)
    {
    triangles=subdivide(vertices, triangles);
    }

    triangles = make_spiky(vertices, triangles);

    return{vertices, triangles};
}