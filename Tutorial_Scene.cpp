// Tutorial_scene.cpp
// This file contains functions that concerns the loading and updating of the application's scene representation layer
#include "Tutorial.hpp"

#include "VK.hpp"

#include <GLFW/glfw3.h>

#include <array>
#include <cassert>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <corecrt_math_defines.h>

#include <random>
#include <chrono>



void Tutorial::load_scene() {
    // the BFS traversal, mesh loading, etc.


    //load the scene file
	try {
		scene72 = S72::load(rtg.configuration.scene_file);
	} catch (std::exception &e) {
		std::cerr << "Failed to load s72-format scene from '" << rtg.configuration.scene_file << "':\n" << e.what() << std::endl;
		return ;
	}

	{//load cameras in the scene file and get references to one of them as the current camera, which is an iterator 
		std::deque<Item> current_nodes;
		for(auto n : scene72.scene.roots){//start with the root nodes
			current_nodes.emplace_back(Item{n,mat4::identity()});
		}

		while(!current_nodes.empty()){//go through the graph using this queue bfs setup (this creates two instances of a child if two nodes have it as one of the children)
			auto [node, world_from_parent] = current_nodes.front();
			current_nodes.pop_front();

			//this node's world transform
			mat4 world_from_local = world_from_parent * node->parent_from_local();//accumulate transform
		 
			if(node->camera != nullptr){//there could be numerous cameras, but every camera has only one instance 
				//put every unique camera in the unordered_map, if there is a duplicate, print err and exit
				
				if(loaded_cameras.find(node->camera->name) != loaded_cameras.end()){
					std::cerr<<"duplicate camera "<<node->camera->name<<std::endl;
					throw;
				}
				assert(!node->camera->projection.valueless_by_exception());
				std::cout<<"loading camera: "<<node->camera->name<<std::endl;;
				S72::Camera::Perspective perspective = get<S72::Camera::Perspective>(node->camera->projection);
				loaded_cameras[node->camera->name] = BasicCamera{
					.eye = world_from_local.translation(),
					.dir = (world_from_local * vec4{0,0,-1,0}).xyz(),
					.up = (world_from_local * vec4{0,1,0,0}).xyz(),
					.aspect = perspective.aspect,
					.vfov = perspective.vfov,
					.near = perspective.near,
					.far = perspective.far,
				};
				assert(loaded_cameras.find(node->camera->name) != loaded_cameras.end());

			}
			for(S72::Node *child : node->children){
				current_nodes.emplace_back(child, world_from_local);
			}
		}
		if(rtg.configuration.required_camera != "") {// if there is a command-line specified camera
			if(loaded_cameras.find(rtg.configuration.required_camera) == loaded_cameras.end()){//and you can't find it *~*
				throw std::runtime_error(
					"Required camera named '" + rtg.configuration.required_camera +
					"' was not found");
			}
			//however if you do find it !o!
			current_camera = loaded_cameras.find(rtg.configuration.required_camera);
		}
		else{//if there is no camera specified
			current_camera = loaded_cameras.begin();
		}
		assert(!loaded_cameras.empty());
	}


    {//load(and create) objects vertices
		std::vector<PosNorTanTexVertex> vertices;

		{//load vertices from s72 file, so that all meshes' vertex data(attributes) are in one big pool
			meshes.clear();
			size_t base = 0;//base offset for writing in each mesh's vertices into std::vector<PosNorTanTexVertex> vertices
			uint32_t count = 0;//the number of vertices for each mesh
			for(auto const &mesh : scene72.meshes ){
				std::cout<<"loading mesh: "<< mesh.first<<std::endl;
				//assuming there aren't duplicate meshes, no indices property, use PosNorTanTex attributes
				assert(meshes.find(mesh.first) == meshes.end());
				//get the indices
				count = mesh.second.count;
				meshes[mesh.first].verts.first = uint32_t(vertices.size());
				meshes[mesh.first].verts.count = count;
				//copy the vertices from file into vertices
				base = vertices.size();
				vertices.resize(base + count);//resize for copying
				S72::Mesh::Attribute const &att_pos = mesh.second.attributes.at("POSITION");//12bytes 
				S72::Mesh::Attribute const &att_nor = mesh.second.attributes.at("NORMAL");//12bytes
				S72::Mesh::Attribute const &att_tan = mesh.second.attributes.at("TANGENT");//16bytes
				S72::Mesh::Attribute const &att_tex= mesh.second.attributes.at("TEXCOORD");//8bytes

				//copy into vertices while computing bounding box
				vec3 max_pos = vec3{-INFINITY,-INFINITY,-INFINITY};
				vec3 min_pos = vec3{INFINITY,INFINITY,INFINITY};
				for(uint32_t i = 0; i<count; ++i){
					std::memcpy(&vertices[base + i].Position, att_pos.src.data.data() + att_pos.offset + att_pos.stride * i, sizeof(vec3));
					std::memcpy(&vertices[base + i].Normal, att_nor.src.data.data() + att_nor.offset + att_nor.stride * i, sizeof(vec3));
					std::memcpy(&vertices[base + i].Tangent, att_tan.src.data.data() + att_tan.offset + att_tan.stride * i, sizeof(vec4));
					std::memcpy(&vertices[base + i].TexCoord, att_tex.src.data.data() + att_tex.offset + att_tex.stride * i, sizeof(float) * 2);
					if(vertices[base + i].Position.x > max_pos.x) max_pos.x = vertices[base + i].Position.x;
					if(vertices[base + i].Position.y > max_pos.y) max_pos.y = vertices[base + i].Position.y;
					if(vertices[base + i].Position.z > max_pos.z) max_pos.z = vertices[base + i].Position.z;
					if(vertices[base + i].Position.x < min_pos.x) min_pos.x = vertices[base + i].Position.x;
					if(vertices[base + i].Position.y < min_pos.y) min_pos.y = vertices[base + i].Position.y;
					if(vertices[base + i].Position.z < min_pos.z) min_pos.z = vertices[base + i].Position.z;
				}
				meshes[mesh.first].bbox.max = max_pos;
				meshes[mesh.first].bbox.min = min_pos;
			}
		}


		{//a spiky fruit (durian?)
			fruit_vertices.first = uint32_t(vertices.size());

			//this code to generate an ico sphere is taken from:
			//https://schneide.blog/2016/07/15/generating-an-icosphere-in-c/#:~:text=for%20(%20int%20i=0;,marching%20cubes%20or%20marching%20tetrahedrons.
			IndexedMesh ball_mesh = make_spiky_icosphere(1);
			//now, use the indexed mesh to emplace vertices
			auto emplace_triangle = [&](Triangle tri){
				for(int32_t i = 2; i>-1; i--){
					vertices.emplace_back(PosNorTanTexVertex{
						.Position{
							.x = ball_mesh.first[tri.vertex[i]].x,
							.y = ball_mesh.first[tri.vertex[i]].y,
							.z = ball_mesh.first[tri.vertex[i]].z,
						},
						.Normal{
							.x = ball_mesh.first[tri.vertex[i]].x,
							.y = ball_mesh.first[tri.vertex[i]].y,
							.z = ball_mesh.first[tri.vertex[i]].z,
						},
						.Tangent{
							.x = - ball_mesh.first[tri.vertex[i]].y,
							.y = ball_mesh.first[tri.vertex[i]].x,
							.z = 0.0f,
							.w = 1.0f,
						},
						.TexCoord{
							.s = ball_mesh.first[tri.vertex[i]].x,
							.t = ball_mesh.first[tri.vertex[i]].y,
						}
					});
				}
				
			};
			for(Triangle const & tri: ball_mesh.second){
				emplace_triangle(tri);
			}
			fruit_vertices.count = uint32_t(vertices.size()) - fruit_vertices.first;
		}
		

		size_t bytes = sizeof(vertices[0]) * vertices.size();
		object_vertices = rtg.helpers.create_buffer(bytes, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT,
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, Helpers::Unmapped);
		//copy data to buffer
		rtg.helpers.transfer_to_buffer(vertices.data(), bytes, object_vertices);
	}
}




void Tutorial::update_scene(float dt) {
    // animation drivers, per-frame graph walk
    {//go through the animation drivers to update nodes' transforms
		for(S72::Driver const &d: scene72.drivers){
			if(scene72.nodes.find(d.node.name) == scene72.nodes.end()){
				std::cerr<<"can't find "<< d.node.name<< " from the scene's nodes"<<std::endl;
			}		
			S72::Node &n = scene72.nodes.at(d.node.name);
			//get the start keyframe index
			auto it = std::upper_bound(d.times.begin(), d.times.end(), time_elapsed)-1;//assuming d.times start with zero, upper_bound should never return the first iterator 
			uint32_t offset = uint32_t(it - d.times.begin()); 
			float t = time_elapsed - *it;
			float t_percentage;
			//if it is the last keyframe, just set 
			if(it == d.times.end()-1){
				switch (d.channel){
					case  S72::Driver::Channel::translation:
						n.translation = vec3(&d.values[offset * 3]);
						break;
					case  S72::Driver::Channel::rotation:
						n.rotation = quat(&d.values[offset * 4]);
						break;
					case  S72::Driver::Channel::scale:
						n.scale = vec3(&d.values[offset * 3]);
						break;
				}
				continue;
			}
			else{
				float t_total = *(it+1) - *it;
				t_percentage = t/t_total;//between 0 and 1
			}
			
			

			switch (d.channel){
				case  S72::Driver::Channel::translation:{
					vec3 trans0(&d.values[offset * 3]);
					vec3 trans1(&d.values[(offset + 1) * 3]);
					n.translation = (trans1 - trans0) * t_percentage + trans0;
					break;
				}	
				case  S72::Driver::Channel::rotation:
				{
					quat r0(&d.values[offset * 4]);//quats have a block size of 4 floats
					quat r1(&d.values[(offset + 1) * 4]);	
					n.rotation = quat::slerp(r0, r1, t_percentage);
					break;
				}
				case  S72::Driver::Channel::scale:{
					vec3 scl0(&d.values[offset * 3]);
					vec3 scl1(&d.values[(offset + 1) * 3]);
					n.scale = (scl1 - scl0) * t_percentage + scl0;
					break;
				}
			}	
		}
	}

	{//load s72 into the object instances, two lights, and cameras
		
		std::deque<Item> current_nodes;
		for(auto n : scene72.scene.roots){//start with the root nodes
			current_nodes.emplace_back(Item{n,mat4::identity()});
		}

		while(!current_nodes.empty()){//go through the graph using this queue bfs setup (this creates two instances of a child if two nodes have it as one of the children)
			auto [node, world_from_parent] = current_nodes.front();
			current_nodes.pop_front();

			//this node's world transform
			mat4 world_from_local = world_from_parent * node->parent_from_local();//accumulate transform

			if(node->mesh != nullptr){//if the node has mesh, instance it
				assert(meshes.find(node->mesh->name) != meshes.end());//check that the mesh's vertices has been loaded
				//std::cout<<"instancing "<<node->mesh->name<<". material is"<<node->mesh->material->name<<std::endl;

				//Try to cull it 
				AABB &b = meshes[node->mesh->name].bbox;
				std::array<vec3, 8> bbox_corners;				
				std::array<vec3, 8> frustrum_corners;
				b.get_box_corners(world_from_local, bbox_corners);
				if(camera_mode == CameraMode::Free || prev_camera_mode == CameraMode::Free){
					frustrum_corners = free_camera.get_frustum_corners(rtg.swapchain_extent.width / float(rtg.swapchain_extent.height));
				}
				else if(camera_mode == CameraMode::Scene || prev_camera_mode == CameraMode::Scene){
					frustrum_corners = current_camera->second.get_frustum_corners();
				}
				//cull the box				

				if(!rtg.configuration.cull || !do_cull(frustrum_corners, bbox_corners)){
					auto v = node->mesh->material->brdf;
					
			
					if(std::holds_alternative<S72::Material::Lambertian>(v)){
						lambertian_object_instances.emplace_back(
							LambertianObjectInstance{
								.vertices = meshes[node->mesh->name].verts,
								.transform{
									.CLIP_FROM_LOCAL = CLIP_FROM_WORLD * world_from_local,
									.WORLD_FROM_LOCAL = world_from_local,
									.WORLD_FROM_LOCAL_NORMAL = world_from_local,//not correct, TODO	
								},
								//if the scenefile doesn't specify material just use the 0 debug material
								.texture = (node->mesh->material == nullptr) ? Texture_Indices{.albedo_index=0} : material_textures_table.at(node->mesh->material->name),
							}
						);
					}
					else if(std::holds_alternative<S72::Material::Mirror>(v) || std::holds_alternative<S72::Material::Environment>(v)){
						bool _is_env = std::holds_alternative<S72::Material::Environment>(v);
						env_mirror_object_instances.emplace_back(
							EnvMirrorObjectInstance{
								.vertices = meshes[node->mesh->name].verts,
								.transform{
									.CLIP_FROM_LOCAL = CLIP_FROM_WORLD * world_from_local,
									.WORLD_FROM_LOCAL = world_from_local,
									.WORLD_FROM_LOCAL_NORMAL = world_from_local,//not correct, TODO	
								},
								//if the scenefile doesn't specify material just use the 0 debug material
								.texture = (node->mesh->material == nullptr) ? Texture_Indices{.albedo_index=0} : material_textures_table.at(node->mesh->material->name),
								.is_env = _is_env ? 1 : 0,
							}
						);
					}
				}			
				if(camera_mode == CameraMode::Debug){
					add_debug_lines_bbox(b, world_from_local);	
				}	
			}
		
			if(node->camera != nullptr){
				//now that all the cameras are loaded
				assert(!node->camera->projection.valueless_by_exception());
				
				assert(loaded_cameras.find(node->camera->name) != loaded_cameras.end());

				S72::Camera::Perspective perspective = get<S72::Camera::Perspective>(node->camera->projection);
				loaded_cameras[node->camera->name] = BasicCamera{
					.eye = world_from_local.translation(),
					.dir = (world_from_local * vec4{0,0,-1,0}).xyz(),
					.up = (world_from_local * vec4{0,1,0,0}).xyz(),
					.aspect = perspective.aspect,
					.vfov = perspective.vfov,
					.near = perspective.near,
					.far = perspective.far,
				};
			}

			if(node->light != nullptr){//two lights for A1
				//resolve the variant				
				std::variant< S72::Light::Sun, S72::Light::Sphere, S72::Light::Spot > &v = node->light->source;
				vec3 world_dir = normalized((world_from_local * vec4{0,0,1,0}).xyz());//local z axis direction(not -z)
				if(std::holds_alternative<S72::Light::Sun>(v)){
					default_world_lights = false;
					S72::Light::Sun &sun = get<S72::Light::Sun>(v);
					if(std::abs(sun.angle - 3.14156926) < 0.001){// a hemisphere light
						world.SKY_DIRECTION.x = world_dir.x;
						world.SKY_DIRECTION.y = world_dir.y;
						world.SKY_DIRECTION.z = world_dir.z;
						world.SKY_ENERGY.r = (node->light->tint * sun.strength).x;
						world.SKY_ENERGY.g = (node->light->tint * sun.strength).y;
						world.SKY_ENERGY.b = (node->light->tint * sun.strength).z;
						///std::cout<<"loaded sky light with strength "<< sun.strength<< "and tint"<< node->light->tint.data<<std::endl; 
					}
					else{
						world.SUN_DIRECTION.x = world_dir.x;
						world.SUN_DIRECTION.y = world_dir.y;
						world.SUN_DIRECTION.z = world_dir.z;
						world.SUN_ENERGY.r = (node->light->tint * sun.strength).x;
						world.SUN_ENERGY.g = (node->light->tint * sun.strength).y;
						world.SUN_ENERGY.b = (node->light->tint * sun.strength).z;
						///std::cout<<"loaded sun light with strength "<< sun.strength<< "and tint"<< world.SUN_ENERGY.r<<world.SUN_ENERGY.g<<
						//world.SUN_ENERGY.b<<std::endl; 
					}
					
				}
				else if(std::holds_alternative<S72::Light::Sphere>(v)){

				}

				else if(std::holds_alternative<S72::Light::Spot>(v)){

				}
			}
			for(S72::Node *child : node->children){
				current_nodes.emplace_back(child, world_from_local);
			}
		}
		
	}
}