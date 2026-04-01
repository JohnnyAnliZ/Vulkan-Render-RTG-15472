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


//could be useful in future but eh
void make_tree_with_fruits(){
    { // lines stuff
	  //  if(starts.size() > 64){
	  //  	std::cout<<"too many nodes"<<std::endl;
	  //  	object_instances.clear();
	  //  	starts.clear();
	  //  	lines_vertices.clear();
	  //  	iters = 0;
	  //  }
	  //  if(lines_vertices.size() > 1024){
	  //  	std::cout<<"too many vertices"<<std::endl;
	  //  	object_instances.clear();
	  //  	starts.clear();
	  //  	lines_vertices.clear();
	  //  	iters = 0;
	  //  }

		//----tree stuff----
		// if(starts.empty())starts.emplace_back(vec3(0.0f,0.0f,0.0f));
		// if(time_elapsed > 0.5f && growing){
		// 	size_t num_nodes = starts.size();

		// 	auto verts = meshes.begin();//go through the mesh vertcies (like with poker cards) to grow them on trees
		// 	for(size_t i = 0; i < num_nodes; ++i){
		// 		vec3 cur_node = starts[i];
		// 		starts.emplace_back(emplace_random_line(cur_node,iters));

		// 		if(dist(engine) > 0.2f){
		// 			vec3 fruit_node = emplace_random_line(cur_node,iters);
		// 			//make fruits
		// 			if(dist(engine) > 1.0f-(iters-5) * 0.07f && iters>5){

		// 				{//spiky ball shrunken by a factor

		// 					float scaling_factor = 0.5f;
		// 					mat4 WORLD_FROM_LOCAL{
		// 						scaling_factor, 0.0f,  0.0f, 0.0f,
		// 						0.0f,scaling_factor, 0.0f, 0.0f,
		// 						0.0f, 0.0f,   scaling_factor, 0.0f,
		// 						fruit_node.x,fruit_node.y,fruit_node.z, 1.0f,
		// 					};
		// 					object_instances.emplace_back(ObjectInstance{
		// 						.vertices = fruit_vertices,//which vertices to use
		// 						.transform{
		// 							.CLIP_FROM_LOCAL = CLIP_FROM_WORLD * WORLD_FROM_LOCAL,
		// 							.WORLD_FROM_LOCAL = WORLD_FROM_LOCAL,
		// 							.WORLD_FROM_LOCAL_NORMAL = WORLD_FROM_LOCAL,
		// 						},
		// 						.texture = 1,
		// 					});
		// 				}
		// 				verts++;
		// 				if(verts == meshes.end())verts = meshes.begin();
		// 			}
		// 			//else branch
		// 			else starts.emplace_back(fruit_node);
		// 		}
		// 	}
		// 	// Remove the old nodes (keep only the new branches)
		// 	starts.erase(starts.begin(), starts.begin() + num_nodes);
		// 	time_elapsed = 0.0f;
		// 	iters++;
		// }
	}

	{ // make some objects

		// { //plane translated +x by one unit:
		// 	mat4 WORLD_FROM_LOCAL{
		// 		1.0f, 0.0f, 0.0f, 0.0f,
		// 		0.0f, 1.0f, 0.0f, 0.0f,
		// 		0.0f, 0.0f, 1.0f, 0.0f,
		// 		1.0f, 0.0f, 0.0f, 1.0f,
		// 	};

		// 	object_instances.emplace_back(ObjectInstance{
		// 		.vertices = plane_vertices,
		// 		.transform{
		// 			.CLIP_FROM_LOCAL = CLIP_FROM_WORLD * WORLD_FROM_LOCAL,
		// 			.WORLD_FROM_LOCAL = WORLD_FROM_LOCAL,
		// 			.WORLD_FROM_LOCAL_NORMAL = WORLD_FROM_LOCAL,
		// 		},
		// 		.texture = 1,
		// 	});
		// }
		// { //torus translated -x by one unit and rotated CCW around +y:
		// 	float ang = time / 60.0f * 2.0f * float(M_PI) * 10.0f;
		// 	float ca = std::cos(ang);
		// 	float sa = std::sin(ang);
		// 	mat4 WORLD_FROM_LOCAL{
		// 		  ca, 0.0f,  -sa, 0.0f,
		// 		0.0f, 1.0f, 0.0f, 0.0f,
		// 		  sa, 0.0f,   ca, 0.0f,
		// 		-1.0f,0.0f, 0.0f, 1.0f,
		// 	};

		// 	object_instances.emplace_back(ObjectInstance{
		// 		.vertices = torus_vertices,
		// 		.transform{
		// 			.CLIP_FROM_LOCAL = CLIP_FROM_WORLD * WORLD_FROM_LOCAL,
		// 			.WORLD_FROM_LOCAL = WORLD_FROM_LOCAL,
		// 			.WORLD_FROM_LOCAL_NORMAL = WORLD_FROM_LOCAL,
		// 		},
		// 	});
		// }
		// {//spiky ball shrunken by a factor of 0.5
		// 	float scaling_factor = 0.5f;
		// 	mat4 WORLD_FROM_LOCAL{
		// 		scaling_factor, 0.0f,  0.0f, 0.0f,
		// 		0.0f,scaling_factor, 0.0f, 0.0f,
		// 		0.0f, 0.0f,   scaling_factor, 0.0f,
		// 		0.0f,0.0f, 0.0f, 1.0f,
		// 	};
		// 	object_instances.emplace_back(ObjectInstance{
		// 		.vertices = fruit_vertices,
		// 		.transform{
		// 			.CLIP_FROM_LOCAL = CLIP_FROM_WORLD * WORLD_FROM_LOCAL,
		// 			.WORLD_FROM_LOCAL = WORLD_FROM_LOCAL,
		// 			.WORLD_FROM_LOCAL_NORMAL = WORLD_FROM_LOCAL,
		// 		},
		// 		.texture = 1,
		// 	});
		// }
	}
}
    