#include <cuda_runtime.h>
#include <stdio.h>
#include <fstream>
#include <sstream>
#include <chrono>
#include <iostream>
#include <vector>
#include <filesystem>
#include <iomanip>
#include <string>

// OS-specific headers
#ifdef _WIN32
    #include <windows.h>
#else
    #include <unistd.h>
#endif

// Include CUDA CCD implementation headers
#include "direct_cubic_root_finder_ccd.h"
#include "interval_partition_ccd.h"
#include "cubic.h"
#include "types.h"
#include "typedef.h"
#include "math.h"
#include "autogen.h"


using namespace std::filesystem;

// Data structures for CPU
struct QueryResult {
    std::vector<cuccd::real> roots;
    std::chrono::nanoseconds duration;
};

struct QueryData {
    std::vector<cuccd::point> points;
};

// Data structures for GPU
struct GPUQueryResult {
    cuccd::real roots[4];  // Maximum of 3 roots for cubic equation + 1 for safeguard
    int num_roots;
    long long duration_ns;
};

// Helper function to convert string to double
double parse_rational(const std::string& num_str, const std::string& den_str) {
    double num = std::stod(num_str);
    double den = std::stod(den_str);
    return num / den;
}

// CUDA error checking macro
#define CUDA_CHECK(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}

// Function to convert GPU clock cycles to nanoseconds
// This is a rough approximation and depends on the specific GPU
double clock_cycles_to_ns(long long cycles, cudaDeviceProp& prop) {
    // Approximate conversion - this can be refined for specific GPUs
    double clock_rate_khz = (double)prop.clockRate;
    return (double)cycles * 1000000.0 / clock_rate_khz;
}

// CUDA kernels for CCD processing
__global__ void process_edge_edge_queries_kernel(const cuccd::point* points, GPUQueryResult* results, int num_queries, bool use_direct_method) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_queries) return;
    
    // Load the 8 points for this query (4 from each time step)
    const cuccd::point& ea0_t0 = points[idx * 8 + 0];
    const cuccd::point& ea1_t0 = points[idx * 8 + 1];
    const cuccd::point& eb0_t0 = points[idx * 8 + 2];
    const cuccd::point& eb1_t0 = points[idx * 8 + 3];
    const cuccd::point& ea0_t1 = points[idx * 8 + 4];
    const cuccd::point& ea1_t1 = points[idx * 8 + 5];
    const cuccd::point& eb0_t1 = points[idx * 8 + 6];
    const cuccd::point& eb1_t1 = points[idx * 8 + 7];
    
    // Generate cubic equation
    auto cubic = cuccd::autogen::edge_edge_ccd_equation(
        ea0_t0.x, ea0_t0.y, ea0_t0.z,
        ea1_t0.x, ea1_t0.y, ea1_t0.z,
        eb0_t0.x, eb0_t0.y, eb0_t0.z,
        eb1_t0.x, eb1_t0.y, eb1_t0.z,
        ea0_t1.x, ea0_t1.y, ea0_t1.z,
        ea1_t1.x, ea1_t1.y, ea1_t1.z,
        eb0_t1.x, eb0_t1.y, eb0_t1.z,
        eb1_t1.x, eb1_t1.y, eb1_t1.z);
    
    // Initialize roots array
    cuccd::root roots;
    roots[0] = -1; roots[1] = -1; roots[2] = -1; roots[3] = -1;
    
    // Find the roots using the specified method
    int num_roots;
    if (use_direct_method) {
        num_roots = cuccd::DirectCubicRootFinder::find_roots_none_conservative(cubic, roots);
    } else {
        num_roots = cuccd::IntervalPartitionCCD::interval_partition_root_finder(cubic, roots);
    }
    
    // Store the results - no timing in the kernel
    results[idx].num_roots = num_roots;
    for (int i = 0; i < num_roots; i++) {
        results[idx].roots[i] = roots[i];
    }
    // We'll set duration in the host code
    results[idx].duration_ns = 0;
}

__global__ void process_vertex_face_queries_kernel(const cuccd::point* points, GPUQueryResult* results, int num_queries, bool use_direct_method) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_queries) return;
    
    // Load the 8 points for this query
    const cuccd::point& p_t0 = points[idx * 8 + 0];
    const cuccd::point& t0_t0 = points[idx * 8 + 1];
    const cuccd::point& t1_t0 = points[idx * 8 + 2];
    const cuccd::point& t2_t0 = points[idx * 8 + 3];
    const cuccd::point& p_t1 = points[idx * 8 + 4];
    const cuccd::point& t0_t1 = points[idx * 8 + 5];
    const cuccd::point& t1_t1 = points[idx * 8 + 6];
    const cuccd::point& t2_t1 = points[idx * 8 + 7];
    
    // Generate cubic equation
    auto cubic = cuccd::autogen::point_triangle_ccd_equation(
        p_t0.x, p_t0.y, p_t0.z,
        t0_t0.x, t0_t0.y, t0_t0.z,
        t1_t0.x, t1_t0.y, t1_t0.z,
        t2_t0.x, t2_t0.y, t2_t0.z,
        p_t1.x, p_t1.y, p_t1.z,
        t0_t1.x, t0_t1.y, t0_t1.z,
        t1_t1.x, t1_t1.y, t1_t1.z,
        t2_t1.x, t2_t1.y, t2_t1.z);
    
    // Initialize roots array
    cuccd::root roots;
    roots[0] = -1; roots[1] = -1; roots[2] = -1; roots[3] = -1;
    
    // Find the roots using the specified method
    int num_roots;
    if (use_direct_method) {
        num_roots = cuccd::DirectCubicRootFinder::find_roots_none_conservative(cubic, roots);
    } else {
        num_roots = cuccd::IntervalPartitionCCD::interval_partition_root_finder(cubic, roots);
    }
    
    // Store the results - no timing in the kernel
    results[idx].num_roots = num_roots;
    for (int i = 0; i < num_roots; i++) {
        results[idx].roots[i] = roots[i];
    }
    // We'll set duration in the host code
    results[idx].duration_ns = 0;
}

// Function to read queries from CSV files
std::vector<QueryData> read_csv(const std::string& filename) {
    std::vector<QueryData> queries;
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return queries;
    }

    std::string line;
    QueryData current_query;
    int row_count = 0;
    int line_number = 0;

    while (std::getline(file, line)) {
        line_number++;
        std::stringstream ss(line);
        std::string value;
        std::vector<std::string> values;
        
        while (std::getline(ss, value, ',')) {
            values.push_back(value);
        }

        if (values.size() != 7) {
            std::cerr << "Error: Expected 7 values, got " << values.size() 
                      << " at line " << line_number << " in file " << filename << std::endl;
            continue;
        }

        try {
            cuccd::point p;
            p.x = static_cast<cuccd::real>(parse_rational(values[0], values[1]));
            p.y = static_cast<cuccd::real>(parse_rational(values[2], values[3]));
            p.z = static_cast<cuccd::real>(parse_rational(values[4], values[5]));
            current_query.points.push_back(p);
            // Ignore the ground truth value (values[6]) since it's not needed for root timing

            row_count++;
            if (row_count == 8) {
                queries.push_back(current_query);
                current_query.points.clear();
                row_count = 0;
            }
        } catch (const std::exception& e) {
            std::cerr << "Error processing line " << line_number << " in file " << filename 
                      << ": " << e.what() << std::endl;
            continue;
        }
    }

    if (row_count != 0) {
        std::cerr << "Warning: Incomplete query at the end of file " << filename 
                  << " (expected 8 rows per query)" << std::endl;
    }

    return queries;
}

// Process queries on GPU
void process_queries_on_gpu(const std::vector<QueryData>& queries, std::vector<QueryResult>& results, 
                           bool is_edge_edge, const std::string& method_name) {
    if (queries.empty()) {
        return;
    }
    
    int num_queries = queries.size();
    size_t points_size = num_queries * 8 * sizeof(cuccd::point);
    size_t results_size = num_queries * sizeof(GPUQueryResult);
    
    // Get device properties
    cudaDeviceProp deviceProp;
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, 0));
    
    // Allocate and prepare host memory for points
    cuccd::point* h_points = new cuccd::point[num_queries * 8];
    for (int i = 0; i < num_queries; i++) {
        for (int j = 0; j < 8; j++) {
            h_points[i * 8 + j] = queries[i].points[j];
        }
    }
    
    // Allocate host memory for results
    GPUQueryResult* h_results = new GPUQueryResult[num_queries];
    
    // Allocate device memory
    cuccd::point* d_points;
    GPUQueryResult* d_results;
    CUDA_CHECK(cudaMalloc(&d_points, points_size));
    CUDA_CHECK(cudaMalloc(&d_results, results_size));
    
    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_points, h_points, points_size, cudaMemcpyHostToDevice));
    
    // Launch the appropriate kernel
    bool use_direct_method = (method_name == "direct");
    int threadsPerBlock = 256;
    int blocksPerGrid = (num_queries + threadsPerBlock - 1) / threadsPerBlock;
    
    // Create a CUDA event to measure kernel execution time
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    
    if (is_edge_edge) {
        process_edge_edge_queries_kernel<<<blocksPerGrid, threadsPerBlock>>>(
            d_points, d_results, num_queries, use_direct_method);
    } else {
        process_vertex_face_queries_kernel<<<blocksPerGrid, threadsPerBlock>>>(
            d_points, d_results, num_queries, use_direct_method);
    }
    
    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());
    
    // Record end time and synchronize
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float kernel_milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&kernel_milliseconds, start, stop));
    
    // Copy results back to host
    CUDA_CHECK(cudaMemcpy(h_results, d_results, results_size, cudaMemcpyDeviceToHost));
    
    // Calculate average time per query in nanoseconds
    long long avg_ns_per_query = static_cast<long long>((kernel_milliseconds * 1000000) / num_queries);
    
    // Process results
    results.resize(num_queries);
    for (int i = 0; i < num_queries; i++) {
        for (int j = 0; j < h_results[i].num_roots; j++) {
            if (h_results[i].roots[j] >= 0 && h_results[i].roots[j] <= 1) {
                results[i].roots.push_back(h_results[i].roots[j]);
            }
        }
        // Use the average time for all queries
        results[i].duration = std::chrono::nanoseconds(avg_ns_per_query);
    }
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_points));
    CUDA_CHECK(cudaFree(d_results));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    delete[] h_points;
    delete[] h_results;
    
    // Print some statistics about the kernel execution
    std::cout << "GPU Kernel execution time: " << kernel_milliseconds << " ms" << std::endl;
    std::cout << "Processed " << num_queries << " queries" << std::endl;
    std::cout << "Average time per query: " << avg_ns_per_query << " ns" << std::endl;
}

void write_results(const std::vector<QueryResult>& results, std::ofstream& file) {
    std::chrono::nanoseconds total_time(0);
    
    for (const auto& result : results) {
        file << result.roots.size() << ",";
        for (const auto& root : result.roots) {
            file << std::setprecision(17) << root << ",";
        }
        file << result.duration.count() << "\n";
        total_time += result.duration;
    }
    
    file << "\nSummary:\n";
    file << "Total queries: " << results.size() << "\n";
    file << "Total processing time: " << total_time.count() << " nanoseconds\n";
    file << "Average time per query: " << (static_cast<double>(total_time.count()) / results.size()) << " nanoseconds\n";
    
    // Provide additional statistics in microseconds for readability
    double microseconds_total = static_cast<double>(total_time.count()) / 1000.0;
    double microseconds_avg = microseconds_total / results.size();
    file << "Total processing time: " << microseconds_total << " microseconds\n";
    file << "Average time per query: " << microseconds_avg << " microseconds\n";
}

void generate_summary_csv(const std::string& result_dir) {
    std::ofstream summary_file(result_dir + "/summary.csv");
    if (!summary_file.is_open()) {
        std::cerr << "Error: Could not create summary file" << std::endl;
        return;
    }
    
    // Write header
    summary_file << "Dataset,QueryType,Method,AverageTimeNs" << std::endl;
    
    // Iterate through result directories
    for (const auto& method : {"direct", "poly"}) {
        std::string method_dir = result_dir + "/" + method;
        if (!exists(method_dir)) continue;
        
        for (const auto& entry : directory_iterator(method_dir)) {
            if (!entry.is_directory()) continue;
            std::string dataset_name = entry.path().filename().string();
            
            // Process edge-edge results
            std::string edge_edge_file = entry.path().string() + "/edge_edge_results.csv";
            if (exists(edge_edge_file)) {
                double avg_time_ns = 0.0;
                bool found_avg = false;
                
                // Parse the file to get average time
                std::ifstream file(edge_edge_file);
                std::string line;
                
                while (std::getline(file, line)) {
                    if (line.find("Average time per query:") != std::string::npos && 
                        line.find("nanoseconds") != std::string::npos) {
                        std::istringstream iss(line.substr(line.find(":") + 1));
                        iss >> avg_time_ns;
                        found_avg = true;
                        break;
                    }
                }
                
                if (found_avg) {
                    summary_file << dataset_name << ",edge-edge," << method << "," 
                                << std::fixed << std::setprecision(2) << avg_time_ns << std::endl;
                }
            }
            
            // Process vertex-face results
            std::string vertex_face_file = entry.path().string() + "/vertex_face_results.csv";
            if (exists(vertex_face_file)) {
                double avg_time_ns = 0.0;
                bool found_avg = false;
                
                // Parse the file to get average time
                std::ifstream file(vertex_face_file);
                std::string line;
                
                while (std::getline(file, line)) {
                    if (line.find("Average time per query:") != std::string::npos && 
                        line.find("nanoseconds") != std::string::npos) {
                        std::istringstream iss(line.substr(line.find(":") + 1));
                        iss >> avg_time_ns;
                        found_avg = true;
                        break;
                    }
                }
                
                if (found_avg) {
                    summary_file << dataset_name << ",vertex-triangle," << method << "," 
                                << std::fixed << std::setprecision(2) << avg_time_ns << std::endl;
                }
            }
        }
    }
    
    std::cout << "Summary CSV generated at " << result_dir << "/summary.csv" << std::endl;
}

int main() {
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        std::cerr << "No CUDA devices found!" << std::endl;
        return 1;
    }
    
    cudaDeviceProp deviceProp;
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, 0));
    std::cout << "Using GPU: " << deviceProp.name << std::endl;
    std::cout << "Clock rate: " << deviceProp.clockRate << " kHz" << std::endl;
    std::cout << "Number of multiprocessors: " << deviceProp.multiProcessorCount << std::endl;
    
    std::string data_dir = cuccd::get_asset_path();
    #ifdef USE_FLOAT
        #ifdef USE_FAST_MATH
            std::string result_dir = cuccd::get_workspace_path() + "/gpu_root_timing/results_float_fast_math";
        #else
            std::string result_dir = cuccd::get_workspace_path() + "/gpu_root_timing/results_float";
        #endif
    #else
        #ifdef USE_FAST_MATH
            std::string result_dir = cuccd::get_workspace_path() + "/gpu_root_timing/results_double_fast_math";
        #else
            std::string result_dir = cuccd::get_workspace_path() + "/gpu_root_timing/results_double";
        #endif
    #endif
    std::vector<std::string> result_dirs = {result_dir + "/direct", result_dir + "/poly"};
    std::cout << "Data directory: " << data_dir << std::endl;
    
    // Create result directories
    for (const auto& dir : result_dirs) {
        create_directories(dir);
    }
   
    for (const auto& entry : directory_iterator(data_dir)) {
        if (!entry.is_directory()) continue;
        std::cout << "Processing dataset: " << entry.path().filename().string() << std::endl;
        std::string dataset_name = entry.path().filename().string();
        std::string edge_edge_dir = entry.path().string() + "/edge-edge";
        std::string vertex_face_dir = entry.path().string() + "/vertex-face";
        
        for (const auto& method : {"direct", "poly"}) {
            std::string method_result_dir = result_dir + "/" + std::string(method);
            create_directories(method_result_dir + "/" + dataset_name);
            
            // Process edge-edge queries
            std::vector<QueryResult> all_edge_edge_results;
            if (exists(edge_edge_dir)) {
                for (const auto& file : {"data_0_0.csv", "data_0_1.csv"}) {
                    std::string file_path = edge_edge_dir + "/" + file;
                    if (exists(file_path)) {
                        std::cout << "Processing edge-edge queries from " << file << std::endl;
                        auto queries = read_csv(file_path);
                        std::vector<QueryResult> results;
                        
                        if (!queries.empty()) {
                            std::cout << "\nProcessing " << dataset_name << " edge-edge queries from " << file 
                                    << " with " << method << " method on GPU...\n";
                            process_queries_on_gpu(queries, results, true, method);
                            all_edge_edge_results.insert(all_edge_edge_results.end(), results.begin(), results.end());
                        } else {
                            std::cout << "No queries found in " << file_path << std::endl;
                        }
                    }
                }
                
                if (!all_edge_edge_results.empty()) {
                    std::string result_file = method_result_dir + "/" + dataset_name + "/edge_edge_results.csv";
                    std::ofstream edge_edge_result(result_file);
                    write_results(all_edge_edge_results, edge_edge_result);
                }
            }
            
            // Process vertex-face queries
            std::vector<QueryResult> all_vertex_face_results;
            if (exists(vertex_face_dir)) {
                for (const auto& file : {"data_0_0.csv", "data_0_1.csv"}) {
                    std::string file_path = vertex_face_dir + "/" + file;
                    if (exists(file_path)) {
                        std::cout << "Processing vertex-face queries from " << file << std::endl;
                        auto queries = read_csv(file_path);
                        std::vector<QueryResult> results;
                        
                        if (!queries.empty()) {
                            std::cout << "\nProcessing " << dataset_name << " vertex-face queries from " << file 
                                    << " with " << method << " method on GPU...\n";
                            process_queries_on_gpu(queries, results, false, method);
                            all_vertex_face_results.insert(all_vertex_face_results.end(), results.begin(), results.end());
                        } else {
                            std::cout << "No queries found in " << file_path << std::endl;
                        }
                    }
                }
                
                if (!all_vertex_face_results.empty()) {
                    std::string result_file = method_result_dir + "/" + dataset_name + "/vertex_face_results.csv";
                    std::ofstream vertex_face_result(result_file);
                    write_results(all_vertex_face_results, vertex_face_result);
                }
            }
        }
    }
    
    generate_summary_csv(result_dir);
    CUDA_CHECK(cudaDeviceReset());
    
    return 0;
}
