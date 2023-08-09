#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/chrono.h>
#include <pybind11/functional.h>

#include "../include/exact.h"
#include "../include/ivf.h"



void FlatIndexL2::add_wrapper(pybind11::array_t<float> data) {
	pybind11::buffer_info info = data.request();
	if (info.ndim != 2)
		throw std::runtime_error("Number of dimensions must be two");
	if (info.format != pybind11::format_descriptor<float>::format())
		throw std::runtime_error("Data type must be float32");

	std::vector<float> data_vector = std::vector<float>((float *)info.ptr, (float *)info.ptr + info.size);

	add(data_vector);
}

std::vector<std::vector<std::pair<float, int>>> FlatIndexL2::search_wrapper(pybind11::array_t<float> query, int k) {
	pybind11::buffer_info info = query.request();
	if (info.ndim != 2)
		throw std::runtime_error("Number of dimensions must be two");
	if (info.format != pybind11::format_descriptor<float>::format())
		throw std::runtime_error("Data type must be float32");

	std::vector<float> query_vector = std::vector<float>((float *)info.ptr, (float *)info.ptr + info.size);

	return search(query_vector, k);
}


void IVFIndex::add_wrapper(pybind11::array_t<float> data) {
	pybind11::buffer_info info = data.request();
	if (info.ndim != 2)
		throw std::runtime_error("Number of dimensions must be two");
	if (info.format != pybind11::format_descriptor<float>::format())
		throw std::runtime_error("Data type must be float32");

	std::vector<float> data_vector = std::vector<float>((float *)info.ptr, (float *)info.ptr + info.size);

	add(data_vector);
}


void IVFIndex::train_wrapper(pybind11::array_t<float> train_data) {
	pybind11::buffer_info info = train_data.request();
	if (info.ndim != 2)
		throw std::runtime_error("Number of dimensions must be two");
	if (info.format != pybind11::format_descriptor<float>::format())
		throw std::runtime_error("Data type must be float32");

	std::vector<float> train_data_vector = std::vector<float>((float *)info.ptr, (float *)info.ptr + info.size);

	train(train_data_vector);
}


std::tuple<
	std::vector<std::vector<float>>, 
	std::vector<std::vector<int>>
> IVFIndex::search_wrapper(pybind11::array_t<float> query, int k) {
	pybind11::buffer_info info = query.request();
	if (info.ndim != 2)
		throw std::runtime_error("Number of dimensions must be two");
	if (info.format != pybind11::format_descriptor<float>::format())
		throw std::runtime_error("Data type must be float32");

	std::vector<float> query_vector = std::vector<float>((float *)info.ptr, (float *)info.ptr + info.size);

	std::vector<std::vector<std::pair<float, int>>> _result = search(query_vector, k);
	
	// Convert to two vectors
	std::vector<std::vector<float>> distances;
	std::vector<std::vector<int>> indices;
	for (auto& vec: _result) {
		std::vector<float> dist;
		std::vector<int> ind;
		for (auto& pair: vec) {
			dist.push_back(pair.first);
			ind.push_back(pair.second);
		}
		distances.push_back(dist);
		indices.push_back(ind);
	}
	return std::make_tuple(distances, indices);
}

PYBIND11_MODULE(knnlib, m) {
	m.doc() = "knnlib";

	pybind11::class_<FlatIndexL2>(m, "FlatIndexL2")
		.def(pybind11::init<int>())
		.def("add", &FlatIndexL2::add_wrapper)
		.def("search", &FlatIndexL2::search_wrapper)
		.def("train", &FlatIndexL2::train_wrapper);

	pybind11::class_<IVFIndex>(m, "IVFIndex")
		.def(pybind11::init<int, int, int>())
		.def("add", &IVFIndex::add_wrapper)
		.def("search", &IVFIndex::search_wrapper)
		.def("train", &IVFIndex::train_wrapper)
		.def_readwrite("size", &IVFIndex::size)
		.def_readwrite("nprobe", &IVFIndex::n_probe);

}
