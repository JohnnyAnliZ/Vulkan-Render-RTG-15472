#include "S72.hpp"

#include <iostream>
#include <cstring>
#include <cassert>
#include <string>
#include <fstream>
//taken from https://github.com/15-472/s72-loader
void print_info(S72 &s72){
	std::cout << "--- Scene Objects ---"<< std::endl;
	std::cout << "Scene: " << s72.scene.name << std::endl;
	std::cout << "Roots: ";
	for (S72::Node* root : s72.scene.roots) {
		std::cout << root->name << ", ";
	}
	std::cout << std::endl;

	std::cout << "Nodes: ";
	for (auto const& pair : s72.nodes) {
		std::cout << pair.first << ", ";
	}
	std::cout << std::endl;

	std::cout << "Meshes: ";
	for (auto const& pair : s72.meshes) {
		std::cout << pair.first << "|" << pair.second.count<<", ";
	}
	std::cout << std::endl;

	std::cout << "Cameras: ";
	for (auto const& pair : s72.cameras) {
		std::cout << pair.first << ", ";
	}
	std::cout << std::endl;

	 std::cout << "Drivers: ";
	for (auto const& driver : s72.drivers) {
		std::cout << driver.name << ", ";
	}
	std::cout << std::endl;

	std::cout << "Materials: ";
	for (auto const& pair : s72.materials) {
		std::cout << pair.first << ", ";
	}
	std::cout << std::endl;

	std::cout << "Environment: ";
	for (auto const& pair : s72.environments) {
		std::cout << pair.first << ", ";
	}
	std::cout << std::endl;

	std::cout << "Lights: ";
	for (auto const& pair : s72.lights) {
		std::cout << pair.first << ", ";
	}
	std::cout << std::endl;

	std::cout << "DataFiles: ";
	for (auto const& pair : s72.data_files) {
		std::cout << pair.first << ", ";
	}
	std::cout << std::endl;
}

void traverse_children(S72 &s72, S72::Node* node, std::string prefix){
	//Print node information
	std::cout << prefix << node->name << ": {";
	if(node->camera != nullptr){
		std::cout << "Camera: " << node->camera->name;
	}
	if(node->mesh != nullptr){
		std::cout << "Mesh: " << node->mesh->name;
		if(node->mesh->material != nullptr){
			std::cout << " {Material: " <<node->mesh->material->name << "}";
		}
	}
	if(node->environment != nullptr){
		std::cout << "Environment: " << node->environment->name;
	}
	if(node->light != nullptr){
		std::cout << "Light: " << node->light->name;
	}

	std::cout << "}" <<std::endl;

	std::string new_prefix = prefix + "- ";
	for(S72::Node* child : node->children){
		traverse_children(s72, child, new_prefix);
	}
}
void print_scene_graph(S72 &s72){
	std::cout << std::endl << "--- Scene Graph ---"<< std::endl;
	for (S72::Node* root : s72.scene.roots) {
		std::cout << "Root: ";
		std::string prefix = "";
		traverse_children(s72, root, prefix);
	}
}

int main(int argc, char **argv) {
	std::string scene_file = "";
	bool print = false;
	for (int argi = 1; argi < argc; ++argi) {
		std::string arg = argv[argi];
		if(arg == "--scene"){
			if (argi + 1 >= argc) throw std::runtime_error("--scene requires a scene file.");
			argi += 1;
			scene_file = argv[argi];
		} else if(arg == "--print"){
			print = true;
		}else {
			throw std::runtime_error("Unrecognized argument '" + arg + "'.");
		}
	}

	//load scene:
	S72 s72;
	try {
		s72 = S72::load(scene_file);
	} catch (std::exception &e) {
		std::cerr << "Failed to load s72-format scene from '" << scene_file << "':\n" << e.what() << std::endl;
		return 1;
	}

	if(print){
		//print out some scene information:
		print_info(s72);
		print_scene_graph(s72);
	}

	return 0;
}
