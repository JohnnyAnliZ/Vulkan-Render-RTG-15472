#include "SceneSystem.hpp"
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


unsigned long long seed = std::chrono::system_clock::now().time_since_epoch().count();
std::mt19937 engine((unsigned int)seed);
std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

SceneSystem::SceneSystem(RTG &rtg_, S72 &scene72_, MaterialSystem &material_system_)
	: rtg(rtg_), scene72(scene72_), material_system(material_system_) {}

SceneSystem::~SceneSystem() {
	if (object_vertices.handle != VK_NULL_HANDLE) {
		rtg.helpers.destroy_buffer(std::move(object_vertices));
	}
}

void SceneSystem::load_scene() {
	try {
		scene72 = S72::load(rtg.configuration.scene_file);
	} catch (std::exception &e) {
		std::cerr << "Failed to load s72-format scene from '" << rtg.configuration.scene_file << "':\n" << e.what() << std::endl;
		return;
	}


	{//preallocate some space on the lines buffer
		lines_vertices.clear();
		constexpr size_t lines_vert_count = 4096;
		lines_vertices.reserve(lines_vert_count);
		starts.reserve(512);
	}


	{//load cameras
		std::deque<Item> current_nodes;
		for (auto n : scene72.scene.roots) {
			current_nodes.emplace_back(Item{n, mat4::identity()});
		}
		while (!current_nodes.empty()) {
			auto [node, world_from_parent] = current_nodes.front();
			current_nodes.pop_front();
			mat4 world_from_local = world_from_parent * node->parent_from_local();

			if (node->camera != nullptr) {
				if (loaded_cameras.find(node->camera->name) != loaded_cameras.end()) {
					std::cerr << "duplicate camera " << node->camera->name << std::endl;
					throw;
				}
				assert(!node->camera->projection.valueless_by_exception());
				std::cout << "loading camera: " << node->camera->name << std::endl;
				S72::Camera::Perspective perspective = std::get<S72::Camera::Perspective>(node->camera->projection);
				loaded_cameras[node->camera->name] = BasicCamera{
					.eye = world_from_local.translation(),
					.dir = (world_from_local * vec4{0,0,-1,0}).xyz(),
					.up  = (world_from_local * vec4{0,1,0,0}).xyz(),
					.aspect = perspective.aspect,
					.vfov   = perspective.vfov,
					.near   = perspective.near,
					.far    = perspective.far,
				};
			}
			for (S72::Node *child : node->children) {
				current_nodes.emplace_back(child, world_from_local);
			}
		}
		if (rtg.configuration.required_camera != "") {
			if (loaded_cameras.find(rtg.configuration.required_camera) == loaded_cameras.end()) {
				throw std::runtime_error("Required camera named '" + rtg.configuration.required_camera + "' was not found");
			}
			current_camera = loaded_cameras.find(rtg.configuration.required_camera);
		} else {
			current_camera = loaded_cameras.begin();
		}
		assert(!loaded_cameras.empty());
	}

	{//load mesh vertex data into a GPU buffer
		std::vector<PosNorTanTexVertex> vertices;

		{
			meshes.clear();
			size_t base = 0;
			uint32_t count = 0;
			for (auto const &mesh : scene72.meshes) {
				std::cout << "loading mesh: " << mesh.first << std::endl;
				assert(meshes.find(mesh.first) == meshes.end());
				count = mesh.second.count;
				meshes[mesh.first].verts.first = uint32_t(vertices.size());
				meshes[mesh.first].verts.count = count;
				base = vertices.size();
				vertices.resize(base + count);
				S72::Mesh::Attribute const &att_pos = mesh.second.attributes.at("POSITION");
				S72::Mesh::Attribute const &att_nor = mesh.second.attributes.at("NORMAL");
				S72::Mesh::Attribute const &att_tan = mesh.second.attributes.at("TANGENT");
				S72::Mesh::Attribute const &att_tex = mesh.second.attributes.at("TEXCOORD");

				vec3 max_pos = vec3{-INFINITY,-INFINITY,-INFINITY};
				vec3 min_pos = vec3{ INFINITY, INFINITY, INFINITY};
				for (uint32_t i = 0; i < count; ++i) {
					std::memcpy(&vertices[base + i].Position, att_pos.src.data.data() + att_pos.offset + att_pos.stride * i, sizeof(vec3));
					std::memcpy(&vertices[base + i].Normal,   att_nor.src.data.data() + att_nor.offset + att_nor.stride * i, sizeof(vec3));
					std::memcpy(&vertices[base + i].Tangent,  att_tan.src.data.data() + att_tan.offset + att_tan.stride * i, sizeof(vec4));
					std::memcpy(&vertices[base + i].TexCoord, att_tex.src.data.data() + att_tex.offset + att_tex.stride * i, sizeof(float) * 2);
					if (vertices[base+i].Position.x > max_pos.x) max_pos.x = vertices[base+i].Position.x;
					if (vertices[base+i].Position.y > max_pos.y) max_pos.y = vertices[base+i].Position.y;
					if (vertices[base+i].Position.z > max_pos.z) max_pos.z = vertices[base+i].Position.z;
					if (vertices[base+i].Position.x < min_pos.x) min_pos.x = vertices[base+i].Position.x;
					if (vertices[base+i].Position.y < min_pos.y) min_pos.y = vertices[base+i].Position.y;
					if (vertices[base+i].Position.z < min_pos.z) min_pos.z = vertices[base+i].Position.z;
				}
				meshes[mesh.first].bbox.max = max_pos;
				meshes[mesh.first].bbox.min = min_pos;
			}
		}

		{//hardcoded spiky fruit
			fruit_vertices.first = uint32_t(vertices.size());
			IndexedMesh ball_mesh = make_spiky_icosphere(1);
			auto emplace_triangle = [&](Triangle tri) {
				for (int32_t i = 2; i > -1; i--) {
					vertices.emplace_back(PosNorTanTexVertex{
						.Position{ ball_mesh.first[tri.vertex[i]].x, ball_mesh.first[tri.vertex[i]].y, ball_mesh.first[tri.vertex[i]].z },
						.Normal{   ball_mesh.first[tri.vertex[i]].x, ball_mesh.first[tri.vertex[i]].y, ball_mesh.first[tri.vertex[i]].z },
						.Tangent{ -ball_mesh.first[tri.vertex[i]].y, ball_mesh.first[tri.vertex[i]].x, 0.0f, 1.0f },
						.TexCoord{  ball_mesh.first[tri.vertex[i]].x, ball_mesh.first[tri.vertex[i]].y },
					});
				}
			};
			for (Triangle const &tri : ball_mesh.second) {
				emplace_triangle(tri);
			}
			fruit_vertices.count = uint32_t(vertices.size()) - fruit_vertices.first;
		}

		size_t bytes = sizeof(vertices[0]) * vertices.size();
		object_vertices = rtg.helpers.create_buffer(bytes,
			VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, Helpers::Unmapped);
		rtg.helpers.transfer_to_buffer(vertices.data(), bytes, object_vertices);
	}
}

void SceneSystem::update_scene(float dt) {
	time_elapsed += dt;
	lines_vertices.clear();

	{//clear per-frame lists
		lambertian_object_instances.clear();
		env_mirror_object_instances.clear();
		pbr_object_instances.clear();
		lights.clear();
		light_shadow_map_sizes.clear();
	}

	//compute frustum for culling from the currently active camera
	std::array<vec3, 8> frustum_corners{};
	float const aspect = rtg.swapchain_extent.width / float(rtg.swapchain_extent.height);
	if (camera_mode == CameraMode::Free || prev_camera_mode == CameraMode::Free) {
		frustum_corners = free_camera.get_frustum_corners(aspect);
	} else if (camera_mode == CameraMode::Scene || prev_camera_mode == CameraMode::Scene) {
		if (current_camera != loaded_cameras.end()) {
			frustum_corners = current_camera->second.get_frustum_corners();
		}
	}
	bool const do_culling = rtg.configuration.cull;

	{//advance animation drivers
		for (S72::Driver const &d : scene72.drivers) {
			if (scene72.nodes.find(d.node.name) == scene72.nodes.end()) {
				std::cerr << "can't find " << d.node.name << " from the scene's nodes" << std::endl;
			}
			S72::Node &n = scene72.nodes.at(d.node.name);
			auto it = std::upper_bound(d.times.begin(), d.times.end(), time_elapsed) - 1;
			uint32_t offset = uint32_t(it - d.times.begin());
			float t = time_elapsed - *it;
			float t_percentage;
			if (it == d.times.end() - 1) {
				switch (d.channel) {
					case S72::Driver::Channel::translation: n.translation = vec3(&d.values[offset * 3]); break;
					case S72::Driver::Channel::rotation:    n.rotation    = quat(&d.values[offset * 4]); break;
					case S72::Driver::Channel::scale:       n.scale       = vec3(&d.values[offset * 3]); break;
				}
				continue;
			} else {
				t_percentage = t / (*(it+1) - *it);
			}
			switch (d.channel) {
				case S72::Driver::Channel::translation: {
					vec3 t0(&d.values[offset * 3]), t1(&d.values[(offset+1) * 3]);
					n.translation = (t1 - t0) * t_percentage + t0;
					break;
				}
				case S72::Driver::Channel::rotation: {
					quat r0(&d.values[offset * 4]), r1(&d.values[(offset+1) * 4]);
					n.rotation = quat::slerp(r0, r1, t_percentage);
					break;
				}
				case S72::Driver::Channel::scale: {
					vec3 s0(&d.values[offset * 3]), s1(&d.values[(offset+1) * 3]);
					n.scale = (s1 - s0) * t_percentage + s0;
					break;
				}
			}
		}
	}

	{//BFS over scene graph: instance meshes, update cameras and lights
		std::deque<Item> current_nodes;
		for (auto n : scene72.scene.roots) {
			current_nodes.emplace_back(Item{n, mat4::identity()});
		}
		while (!current_nodes.empty()) {
			auto [node, world_from_parent] = current_nodes.front();
			current_nodes.pop_front();
			mat4 world_from_local = world_from_parent * node->parent_from_local();

			if (node->mesh != nullptr) {
				assert(meshes.find(node->mesh->name) != meshes.end());
				AABB &b = meshes[node->mesh->name].bbox;
				std::array<vec3, 8> bbox_corners;
				b.get_box_corners(world_from_local, bbox_corners);

				if (!do_culling || !do_cull(frustum_corners, bbox_corners)) {
					auto v = node->mesh->material->brdf;
					mat3 linear(world_from_local);
					mat3 normal_matrix = linear.inverse().transpose();

					if (std::holds_alternative<S72::Material::Lambertian>(v)) {
						lambertian_object_instances.emplace_back(LambertianObjectInstance{
							.vertices = meshes[node->mesh->name].verts,
							.transform{ .WORLD_FROM_LOCAL = world_from_local, .WORLD_FROM_LOCAL_NORMAL = mat4(normal_matrix) },
							.texture = (node->mesh->material == nullptr) ? 0 : material_system.material_texture_descriptor_set_table.at(node->mesh->material->name),
						});
					} else if (std::holds_alternative<S72::Material::Mirror>(v) || std::holds_alternative<S72::Material::Environment>(v)) {
						env_mirror_object_instances.emplace_back(EnvMirrorObjectInstance{
							.vertices = meshes[node->mesh->name].verts,
							.transform{ .WORLD_FROM_LOCAL = world_from_local, .WORLD_FROM_LOCAL_NORMAL = mat4(normal_matrix) },
							.texture = (node->mesh->material == nullptr) ? 0 : material_system.material_texture_descriptor_set_table.at(node->mesh->material->name),
							.is_env = std::holds_alternative<S72::Material::Environment>(v) ? 1 : 0,
						});
					} else if (std::holds_alternative<S72::Material::PBR>(v)) {
						pbr_object_instances.emplace_back(PbrObjectInstance{
							.vertices = meshes[node->mesh->name].verts,
							.transform{ .WORLD_FROM_LOCAL = world_from_local, .WORLD_FROM_LOCAL_NORMAL = mat4(normal_matrix) },
							.texture = (node->mesh->material == nullptr) ? 0 : material_system.material_texture_descriptor_set_table.at(node->mesh->material->name),
						});
					}
				}


			}

			if (node->camera != nullptr) {
				assert(!node->camera->projection.valueless_by_exception());
				assert(loaded_cameras.find(node->camera->name) != loaded_cameras.end());
				S72::Camera::Perspective perspective = std::get<S72::Camera::Perspective>(node->camera->projection);
				loaded_cameras[node->camera->name] = BasicCamera{
					.eye = world_from_local.translation(),
					.dir = (world_from_local * vec4{0,0,-1,0}).xyz(),
					.up  = (world_from_local * vec4{0,1,0,0}).xyz(),
					.aspect = perspective.aspect,
					.vfov   = perspective.vfov,
					.near   = perspective.near,
					.far    = perspective.far,
				};
			}

			if (node->light != nullptr) {
				std::variant<S72::Light::Sun, S72::Light::Sphere, S72::Light::Spot> &v = node->light->source;
				vec4 color     = vec4(node->light->tint.x, node->light->tint.y, node->light->tint.z, 0);
				vec4 world_dir = world_from_local * vec4{0,0,-1,0};
				vec4 world_pos = world_from_local * vec4{0,0,0,1};

				if (std::holds_alternative<S72::Light::Sun>(v)) {
					default_world_lights = false;
					S72::Light::Sun &sun = std::get<S72::Light::Sun>(v);
					lights.emplace_back(Light{ .color=color, .direction=world_dir, .type=0, .angle=sun.angle, .strength=sun.strength });
					light_shadow_map_sizes.emplace_back(node->light->shadow);
				} else if (std::holds_alternative<S72::Light::Sphere>(v)) {
					default_world_lights = false;
					S72::Light::Sphere &sphere = std::get<S72::Light::Sphere>(v);
					lights.emplace_back(Light{ .color=color, .position=world_pos, .type=1, .limit=sphere.limit, .radius=sphere.radius, .power=sphere.power });
					lights.back().compute_clip_from_world_sphere();
					light_shadow_map_sizes.emplace_back(node->light->shadow);
				} else if (std::holds_alternative<S72::Light::Spot>(v)) {
					default_world_lights = false;
					S72::Light::Spot &spot = std::get<S72::Light::Spot>(v);
					lights.emplace_back(Light{ .color=color, .position=world_pos, .direction=world_dir, .type=2, .limit=spot.limit, .radius=spot.radius, .power=spot.power, .fov=spot.fov, .blend=spot.blend });
					lights.back().compute_clip_from_world_spot();
					light_shadow_map_sizes.emplace_back(node->light->shadow);
				}
			}

			for (S72::Node *child : node->children) {
				current_nodes.emplace_back(child, world_from_local);
			}
		}

		{//rebuild shadow atlas
			uint32_t old_atlas_size = atlas_size;
			atlas_size = Shadow2DPipeline::find_fitting_atlas_size(total_shadow_map_size);
			assert(atlas_size == old_atlas_size);
			allocate_texture_atlas(point{atlas_size, atlas_size}, light_shadow_map_sizes);
		}
	}

	{//update camera matrices from active camera
		float const asp = rtg.swapchain_extent.width / float(rtg.swapchain_extent.height);
		if (camera_mode == CameraMode::Scene) {
			if (current_camera == loaded_cameras.end()) {
				//fallback orbit when no s72 camera is loaded
				float ang = float(M_PI) * 2.0f * (std::fmod(time_elapsed, 5.0f) / 5.0f);
				CLIP_FROM_WORLD = perspective(60.0f * float(M_PI) / 180.0f, asp, 0.1f, 1000.0f)
				                * look_at(vec3(13.0f * std::cos(ang), 13.0f * std::sin(ang), 5.0f),
				                          vec3(0.0f, 0.0f, 5.0f), vec3(0.0f, 0.0f, 1.0f));
				EYE    = vec3(13.0f * std::cos(ang), 13.0f * std::sin(ang), 5.0f);
				CAM_DIR = normalized(vec3(0.0f, 0.0f, 5.0f) - EYE);
			} else {
				CLIP_FROM_WORLD = current_camera->second.clip_from_world();
				EYE    = current_camera->second.eye;
				CAM_DIR = normalized(current_camera->second.dir);
			}
		} else if (camera_mode == CameraMode::Free) {
			CLIP_FROM_WORLD = free_camera.clip_from_world(asp);
			EYE    = free_camera.get_eye();
			CAM_DIR = normalized(free_camera.target - EYE);
		} else if (camera_mode == CameraMode::Debug) {
			CLIP_FROM_WORLD = debug_camera.clip_from_world(asp);
			EYE    = debug_camera.get_eye();
			CAM_DIR = normalized(debug_camera.target - EYE);
			add_debug_lines_frustrum();
		}
	}
}

void SceneSystem::add_cuboid_from_corners(std::array<vec3, 8> const &box_corners, uint8_t r, uint8_t g, uint8_t b) {
	auto emplace_line = [&](uint32_t from, uint32_t to) {
		lines_vertices.emplace_back(LinesPipeline::Vertex{.Position = box_corners[from], .Color = {.r = r, .g = g, .b = b, .a = 255}});
		lines_vertices.emplace_back(LinesPipeline::Vertex{.Position = box_corners[to],   .Color = {.r = r, .g = g, .b = b, .a = 255}});
	};
	emplace_line(0,1); emplace_line(1,2); emplace_line(2,3); emplace_line(3,0); // near face
	emplace_line(4,5); emplace_line(5,6); emplace_line(6,7); emplace_line(7,4); // far face
	emplace_line(0,4); emplace_line(1,5); emplace_line(2,6); emplace_line(3,7); // connectors
}

void SceneSystem::add_debug_lines_bbox(AABB &bbox, mat4 WORLD_FROM_LOCAL) {
	std::array<vec3, 8> corners;
	bbox.get_box_corners(WORLD_FROM_LOCAL, corners);
	add_cuboid_from_corners(corners, 0, 255, 0);
}

void SceneSystem::add_debug_lines_frustrum() {
	std::array<vec3, 8> corners;
	vec3 color;
	if (prev_camera_mode == CameraMode::Scene && current_camera != loaded_cameras.end()) {
		corners = current_camera->second.get_frustum_corners();
		color = vec3{0.0f, 0.0f, 1.0f};
	} else if (prev_camera_mode == CameraMode::Free) {
		corners = free_camera.get_frustum_corners(rtg.swapchain_extent.width / float(rtg.swapchain_extent.height));
		color = vec3{0.0f, 1.0f, 0.0f};
	} else {
		return;
	}
	add_cuboid_from_corners(corners,
		uint8_t(std::round(256 * color.x)),
		uint8_t(std::round(256 * color.y)),
		uint8_t(std::round(256 * color.z)));
}

bool SceneSystem::do_cull(std::array<vec3, 8> const &frustrum_corners, std::array<vec3, 8> const &box_corners) {
	auto test_axis = [&](vec3 const &dir) -> bool {
		std::array<float, 8> box_points;
		std::array<float, 8> frustrum_points;
		for (uint32_t i = 0; i < 8; i++) {
			box_points[i]     = dot(box_corners[i],     dir);
			frustrum_points[i] = dot(frustrum_corners[i], dir);
		}
		std::sort(box_points.begin(),     box_points.end());
		std::sort(frustrum_points.begin(), frustrum_points.end());
		if (box_points[7] < frustrum_points[0] || box_points[0] > frustrum_points[7]) return true;
		return false;
	};

	vec3 box_axes[3] = {
		box_corners[1] - box_corners[0],
		box_corners[3] - box_corners[0],
		box_corners[4] - box_corners[0],
	};
	for (uint32_t i = 0; i < 3; i++) {
		if (test_axis(box_axes[i])) return true;
	}

	vec3 frustrum_edges[6] = {
		frustrum_corners[1] - frustrum_corners[0],
		frustrum_corners[2] - frustrum_corners[1],
		frustrum_corners[4] - frustrum_corners[0],
		frustrum_corners[6] - frustrum_corners[2],
		frustrum_corners[5] - frustrum_corners[1],
		frustrum_corners[7] - frustrum_corners[3],
	};
	vec3 frustrum_normals[5] = {
		cross(frustrum_edges[0], frustrum_edges[1]),
		cross(frustrum_edges[2], frustrum_edges[1]),
		cross(frustrum_edges[3], frustrum_edges[1]),
		cross(frustrum_edges[2], frustrum_edges[0]),
		cross(frustrum_edges[3], frustrum_edges[0]),
	};
	for (uint32_t i = 0; i < 5; i++) {
		if (test_axis(frustrum_normals[i])) return true;
	}

	for (vec3 const &bedge : box_axes) {
		for (vec3 const &fedge : frustrum_edges) {
			if (test_axis(cross(bedge, fedge))) return true;
		}
	}
	return false;
}

//texture atlas packing (https://lisyarus.github.io/blog/posts/texture-packing.html)
void SceneSystem::allocate_texture_atlas(point const &atlas_size_xy, std::vector<uint32_t> const &texture_sizes) {
	assert(lights.size() == texture_sizes.size());
	std::vector<uint32_t> sorted(texture_sizes.size());
	for (int i = 0; i < (int)sorted.size(); ++i) sorted[i] = i;
	std::sort(sorted.begin(), sorted.end(), [&](int i, int j){ return texture_sizes[i] > texture_sizes[j]; });

	std::vector<point> ladder;
	point pen{0, 0};

	for (int i : sorted) {
		int const size = texture_sizes[i];
		uint32_t faces = 0;
		if (lights[i].type == 0) faces = S72::Light::Sun::shadow_map_num;
		if (lights[i].type == 1) faces = S72::Light::Sphere::shadow_map_num;
		if (lights[i].type == 2) faces = S72::Light::Spot::shadow_map_num;

		for (uint32_t face = 0; face < faces; face++) {
			lights[i].shadow_atlases[face].x = float(pen.x) / float(atlas_size_xy.x);
			lights[i].shadow_atlases[face].y = float(pen.y) / float(atlas_size_xy.y);
			lights[i].shadow_atlases[face].z = float(size)  / float(atlas_size_xy.x);
			lights[i].shadow_atlases[face].w = float(size)  / float(atlas_size_xy.y);
			pen.x += size;

			if (!ladder.empty() && ladder.back().y == pen.y + size)
				ladder.back().x = pen.x;
			else
				ladder.push_back({pen.x, pen.y + (uint32_t)size});

			if (pen.x == atlas_size_xy.x) {
				ladder.pop_back();
				pen.y += size;
				pen.x = ladder.empty() ? 0 : ladder.back().x;
			}
		}
	}
}


vec3 SceneSystem::emplace_random_line(vec3 start, uint32_t iter)
{
	// do some approximation of tree growing based on current iteration number and height

	float length_modifier = powf(0.9f, (float)iter); // length gets smaller

	float up_modifier = powf(0.3f, (float)iter); // tree grows less and less "up"

	vec3 growth = vec3(dist(engine), dist(engine), up_modifier + std::abs(dist(engine)));

	vec3 new_location = start + length_modifier * normalized(growth);
	float color_key = (dist(engine) + 1.0f) / 2;
	uint8_t r = (static_cast<uint8_t>(std::floor(color_key * 256.0f)));
	uint8_t g = (static_cast<uint8_t>(std::floor(color_key * 486.0f * color_key)));
	uint8_t b = (static_cast<uint8_t>(std::floor(color_key * 238.0f)));
	uint8_t a = 1;
	lines_vertices.emplace_back(PosColVertex{
		.Position = start,
		.Color{.r = r, .g = g, .b = b, .a = a},
	});
	lines_vertices.emplace_back(PosColVertex{
		.Position = new_location,
	});
	return new_location;
}
