# Meshlets

When using _graphics mesh pipelines_ with _task_ and _mesh shaders_, a typical use case is to segmented meshes into smaller segments called meshlets, and process these. These meshlets contain multiple triangles that should ideally be structured in a meaningful way in memory in order to enable efficient processing, like for example good memory and cache coherency. Meshlets of triangle meshes typically consist of relatively small packages of vertices and triangles of the original mesh geometry.
More information on how the meshlet pipeline works can be found on Nvidia's developer blog: [Christoph Kubisch - Introduction to Turing Mesh Shaders](https://developer.nvidia.com/blog/introduction-turing-mesh-shaders/).

## Dividing Meshes into Meshlets in Auto-Vk-Toolkit

_Auto-Vk-Toolkit_ provides utility functions that help to divide a mesh into meshlets and convert the resulting data structure into one that can be directly used on the GPU. The implementation can be found in [meshlet_helpers.hpp](../auto_vk_toolkit/include/meshlet_helpers.hpp) and [meshlet_helpers.cpp](../auto_vk_toolkit/src/meshlet_helpers.cpp).

As a first step, the models and mesh indices that are to be divided into meshlets need to be selected. The helper function `avk::make_model_references_and_mesh_indices_selection` can be used for this purpose.

The resulting collection can be used with one of the overloads of `avk::divide_into_meshlets`. If no custom division function is provided to this helper function, a simple algorithm (via `avk::basic_meshlets_divider`) is used by default, which just combines consecutive vertices into a meshlet until the limits defined by its parameters `aMaxVertices` and `aMaxIndices` have been reached. It should be noted that this will likely not result in good vertex reuse, but is a quick way to get up and running. 

The function `avk::divide_into_meshlets` also offers a custom division function to be passed as parameter. This custom division function allows the usage of custom division algorithms, as provided through external libraries like [meshoptimizer](https://github.com/zeux/meshoptimizer), for example.

### Using a Custom Division Function

The custom division function (parameter `aMeshletDivision` to `avk::divide_into_meshlets`) can either receive the vertices and indices or just the indices depending on the use case.
Additionally it receives the model and optionally the mesh index. If no mesh index is provided then the meshes were combined beforehand.

Please note that the model will be assigned to each [`struct meshlet`](../auto_vk_toolkit/include/meshlet_helpers.hpp#L7) after `aMeshletDivision` has executed (see implementation of [`avk::divide_indexed_geometry_into_meshlets`](../auto_vk_toolkit/include/meshlet_helpers.hpp#L130)).

The custom division function must follow a specific declaration schema. Optional parameters can be omitted, but all of them need to be provided in the following order:

| Parameter | Mandatory? | Description |
| :------ | :---: | :--- |
| `const std::vector<glm::vec3>& tVertices` | no | The vertices of the mesh, or of combined meshes of the model |
| `const std::vector<uint32_t>& tIndices` | yes | The indices of the mesh, or of combined meshes of the model | 
| `const model_t& tModel` | yes	| The model these meshlets are generated from | 
| `std::optional<mesh_index_t> tMeshIndex` | yes | The optional mesh index. If no value is passed, it means the meshes of the model are combined into a single vertex and index buffer. | 
| `uint32_t tMaxVertices` | yes | The maximum number of vertices that should be added to a single meshlet | 
| `uint32_t tMaxIndices` | yes | The maximum number of indices that should be added to a single meshlet | 

The custom division function **must** return a `std::vector<meshlet>` by value, containing the generated meshlets.

#### Example for Vertices and Indices:

Example of a custom division function that can be passed to `avk::divide_into_meshlets`, that takes both, vertices and indices:

```C++
[](const std::vector<glm::vec3>& tVertices, const std::vector<uint32_t>& aIndices,
    const avk::model_t& aModel, std::optional<avk::mesh_index_t> aMeshIndex,
    uint32_t aMaxVertices, uint32_t aMaxIndices) {

    std::vector<avk::meshlet> generatedMeshlets;
    // Perform meshlet division here, and store resulting meshlet in generatedMeshlets
    return generatedMeshlets;
}
```

#### Example for Indices Only:

Example of a custom division function that can be passed to `avk::divide_into_meshlets`, that takes indices only:

```C++
[](const std::vector<uint32_t>& aIndices,
    const avk::model_t& aModel, std::optional<avk::mesh_index_t> aMeshIndex,
    uint32_t aMaxVertices, uint32_t aMaxIndices) {

    std::vector<avk::meshlet> generatedMeshlets;
    // Perform meshlet division here, and store resulting meshlet in generatedMeshlets
    return generatedMeshlets;
}
```

#### Example That Uses a 3rd Party Library:

An example about how [meshoptimizer](https://github.com/zeux/meshoptimizer) **can** be used via the custom division function that can be passed to `avk::divide_into_meshlets`:

Please note: meshoptimizer expects `aMaxIndices` to be divisible by 4. Therefore the value 124 triangles (372 indices) has been recommended by [meshoptimizer](https://github.com/zeux/meshoptimizer#mesh-shading) for Nvidia cards!

```C++
[](const std::vector<glm::vec3>& tVertices, const std::vector<uint32_t>& aIndices,
    const avk::model_t& aModel, std::optional<avk::mesh_index_t> aMeshIndex,
    uint32_t aMaxVertices, uint32_t aMaxIndices) {

    // definitions
    size_t max_triangles = aMaxIndices/3;
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
    for(int k = 0; k < meshlet_count; k++) {
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
```

## Converting Into a Format for GPU Usage

Meshlets in the host-side format [`struct meshlet`](../auto_vk_toolkit/include/meshlet_helpers.hpp#L7) cannot be directly used in device code, because they store data in data types from the C++ Standard Library. Therefore, these records need to be converted into a suitable format for the GPU. The `avk::convert_for_gpu_usage<T>` utility functions **can** be used for this purpose. 

Two types for device-side meshlets are provided by _Auto-Vk-Toolkit_, which are both supported by `avk::convert_for_gpu_usage<T>`:

```c++
/** Meshlet for GPU usage
 *  @tparam NV    The number of vertices
 *  @tparam NI    The number of indices
 */
template <size_t NV = 64, size_t NI = 378>
struct meshlet_gpu_data
{
    static const size_t sNumVertices = NV;
    static const size_t sNumIndices = NI;

    /** Vertex indices into the vertex array */
    uint32_t mVertices[NV];
    /** Indices into the vertex indices */
    uint8_t mIndices[NI];  
    /** The vertex count */
    uint8_t mVertexCount;
    /** The primitive count */
    uint8_t mPrimitiveCount;
};

/** Meshlet for GPU usage in combination with the meshlet data generated by convert_for_gpu_usage */
struct meshlet_redirected_gpu_data
{
    /** Data offset into the meshlet data array */
    uint32_t mDataOffset;
    /** The vertex count */
    uint8_t mVertexCount;
    /** The primitive count */
    uint8_t mPrimitiveCount;
};
```

The main conceptual difference between the two types `avk::meshlet_gpu_data` and `avk::meshlet_redirected_gpu_data` is that `avk::meshlet_gpu_data` has the vertex indices of a meshlet stored directly in the meshlet struct instance, whereas `avk::meshlet_redirected_gpu_data` uses a separate vertex index array that is indexed by the data stored in the meshlet struct instance. Therefore, the latter type is called "redirected" and it can help to reduce the memory footprint of a meshlet. On the other hand, it requires an additional indirection into a separate index buffer.

If a custom GPU-suitable format is needed, our implementation can be used as a reference for converting [`struct meshlet`](../auto_vk_toolkit/include/meshlet_helpers.hpp#L7) into that custom GPU-suitable format. Transformation into a different GPU-suitable format must be implemented manually.
