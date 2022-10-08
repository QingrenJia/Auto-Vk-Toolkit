#include <auto_vk_toolkit.hpp>
#include "../../meshoptimizer/src/meshoptimizer.h"

namespace avk
{

	std::vector<meshlet> divide_into_meshlets(std::vector<std::tuple<avk::model, std::vector<mesh_index_t>>>& aModelsAndMeshletIndices,
		const bool aCombineSubmeshes, const uint32_t aMaxVertices, const uint32_t aMaxIndices)
	{
		return divide_into_meshlets(aModelsAndMeshletIndices, opt_meshlets_divider, aCombineSubmeshes, aMaxVertices, aMaxIndices);
	}

	std::vector<meshlet> basic_meshlets_divider(const std::vector<uint32_t>& aIndices,
		const model_t& aModel,
		std::optional<mesh_index_t> aMeshIndex,
		uint32_t aMaxVertices, uint32_t aMaxIndices)
	{
		std::vector<meshlet> result;

		int vertexIndex = 0;
		const int numIndices = static_cast<int>(aIndices.size());
		while (vertexIndex < numIndices) {
			auto& ml = result.emplace_back();
			ml.mVertexCount = 0u;
			ml.mVertices.resize(aMaxVertices);
			ml.mIndices.resize(aMaxIndices);
			int meshletVertexIndex = 0;
			// this simple algorithm duplicates vertices, therefore vertexCount == indexCount
			while (meshletVertexIndex < aMaxVertices - 3 && ml.mVertexCount < aMaxIndices - 3 && vertexIndex + meshletVertexIndex < numIndices) {
				ml.mIndices[meshletVertexIndex  + 0] = meshletVertexIndex + 0;
				ml.mIndices[meshletVertexIndex  + 1] = meshletVertexIndex + 1;
				ml.mIndices[meshletVertexIndex  + 2] = meshletVertexIndex + 2;
				ml.mVertices[meshletVertexIndex + 0] = aIndices[vertexIndex + meshletVertexIndex + 0];
				ml.mVertices[meshletVertexIndex + 1] = aIndices[vertexIndex + meshletVertexIndex + 1];
				ml.mVertices[meshletVertexIndex + 2] = aIndices[vertexIndex + meshletVertexIndex + 2];
				ml.mVertexCount += 3u;

				meshletVertexIndex += 3;
			}
			ml.mIndexCount = ml.mVertexCount;
			// resize to actual size
			ml.mVertices.resize(ml.mVertexCount);
			ml.mVertices.shrink_to_fit();
			ml.mIndices.resize(ml.mVertexCount);
			ml.mIndices.shrink_to_fit();
			ml.mMeshIndex = aMeshIndex;
			vertexIndex += meshletVertexIndex;
		}

		return result;
	}

	std::vector<meshlet> opt_meshlets_divider(const std::vector<glm::vec3>& tVertices, const std::vector<uint32_t>& aIndices,
		const avk::model_t& aModel, std::optional<avk::mesh_index_t> aMeshIndex,
		uint32_t aMaxVertices, uint32_t aMaxIndices) {

			// definitions
			size_t max_triangles = aMaxIndices / 3;
			const float cone_weight = 0.0f;

			// get the maximum number of meshlets that could be generated
			size_t max_meshlets = meshopt_buildMeshletsBound(aIndices.size(), aMaxVertices, max_triangles);
			std::vector<meshopt_Meshlet> meshlets(max_meshlets);
			std::vector<unsigned int> meshlet_vertices(max_meshlets * aMaxVertices);
			std::vector<unsigned char> meshlet_triangles(max_meshlets * max_triangles * 3);

			// let meshoptimizer build the meshlets for us
			size_t meshlet_count = meshopt_buildMeshlets(meshlets.data(), meshlet_vertices.data(), meshlet_triangles.data(),
				aIndices.data(), aIndices.size(), &tVertices[0].x, tVertices.size(), sizeof(glm::vec3),
				aMaxVertices, max_triangles, cone_weight);

			// copy the data over to Auto-Vk-Toolkit's meshlet structure
			std::vector<avk::meshlet> generatedMeshlets(meshlet_count);
			generatedMeshlets.resize(meshlet_count);
			generatedMeshlets.reserve(meshlet_count);
			for (int k = 0; k < meshlet_count; k++) {
				auto& m = meshlets[k];
				auto& gm = generatedMeshlets[k];
				gm.mIndexCount = m.triangle_count * 3;
				gm.mVertexCount = m.vertex_count;
				gm.mVertices.reserve(m.vertex_count);
				gm.mVertices.resize(m.vertex_count);
				gm.mIndices.reserve(gm.mIndexCount);
				gm.mIndices.resize(gm.mIndexCount);
				std::ranges::copy(meshlet_vertices.begin() + m.vertex_offset,
					meshlet_vertices.begin() + m.vertex_offset + m.vertex_count,
					gm.mVertices.begin());
				std::ranges::copy(meshlet_triangles.begin() + m.triangle_offset,
					meshlet_triangles.begin() + m.triangle_offset + gm.mIndexCount,
					gm.mIndices.begin());
			}

			return generatedMeshlets;
	}

	std::vector<meshlet> divide_into_groups(std::vector<std::tuple<avk::model, std::vector<mesh_index_t>>>& aModelsAndMeshletIndices,
		std::vector<meshlet>& recived_meshlets,
		const bool aCombineSubmeshes, const uint32_t aMaxVertices, const uint32_t aMaxIndices, const uint32_t aMaxMeshlets)
	{
		return divide_into_groups(aModelsAndMeshletIndices, recived_meshlets, opt_group_divider, aCombineSubmeshes, aMaxVertices, aMaxIndices, aMaxMeshlets);
	}
	std::vector<meshlet> opt_group_divider(const std::vector<glm::vec3>& tVertices, const std::vector<uint32_t>& aIndices,
		const avk::model_t& aModel, std::optional<avk::mesh_index_t> aMeshIndex, std::vector<meshlet>& recived_meshlets,
		uint32_t aMaxVertices, uint32_t aMaxIndices, const uint32_t aMaxMeshlets) {
		// divide all meshlets into groups contains 4 meshlets
		// definitions
		size_t max_triangles = aMaxIndices / 3;
		const float cone_weight = 0.0f;

		// get the maximum number of meshlets that could be generated
		size_t max_meshlets = meshopt_buildMeshletsBound(aIndices.size(), aMaxVertices, max_triangles);
		std::vector<meshopt_Meshlet> meshlets(max_meshlets);
		std::vector<unsigned int> meshlet_vertices(max_meshlets * aMaxVertices);
		std::vector<unsigned char> meshlet_triangles(max_meshlets * max_triangles * 3);

		// let meshoptimizer build the meshlets for us
		size_t meshlet_count = meshopt_buildMeshlets(meshlets.data(), meshlet_vertices.data(), meshlet_triangles.data(),
			aIndices.data(), aIndices.size(), &tVertices[0].x, tVertices.size(), sizeof(glm::vec3),
			aMaxVertices, max_triangles, cone_weight);

		// copy the data over to Auto-Vk-Toolkit's meshlet structure
		std::vector<avk::meshlet> generatedMeshlets(meshlet_count);
		generatedMeshlets.resize(meshlet_count);
		generatedMeshlets.reserve(meshlet_count);
		for (int k = 0; k < meshlet_count; k++) {
			auto& m = meshlets[k];
			auto& gm = generatedMeshlets[k];
			gm.mIndexCount = m.triangle_count * 3;
			gm.mVertexCount = m.vertex_count;
			gm.mVertices.reserve(m.vertex_count);
			gm.mVertices.resize(m.vertex_count);
			gm.mIndices.reserve(gm.mIndexCount);
			gm.mIndices.resize(gm.mIndexCount);
			std::ranges::copy(meshlet_vertices.begin() + m.vertex_offset,
				meshlet_vertices.begin() + m.vertex_offset + m.vertex_count,
				gm.mVertices.begin());
			std::ranges::copy(meshlet_triangles.begin() + m.triangle_offset,
				meshlet_triangles.begin() + m.triangle_offset + gm.mIndexCount,
				gm.mIndices.begin());
		}

		return generatedMeshlets;
	}
}

