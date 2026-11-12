#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest.h>
#include <xeg/XEG.hpp>

static std::string fixture(const char* name) {
  return std::string(XEG_TEST_DATA_DIR)+"/"+name+".json";
}

TEST_CASE("GPU tree retains legal witnesses at zero and rounded costs") {
  for (const char* name : {"tree_zero_cost_cycle", "tree_rounded_cost_cycle", "batch_then_direct_switch"}) {
    XEG graph; graph.read(fixture(name));
    const auto cpu = graph.run_cpu(1,0);
    for (int i = 0; i < 5; ++i) {
      auto result = graph.run_gpu(0,DagKernel::Block,64,512);
      DOCTEST_CHECK(result["seed_valid"] == true);
      DOCTEST_CHECK(result["validated"] == true);
      DOCTEST_CHECK(result["tree_cost"].get<double>() == doctest::Approx(cpu["tree_cost"].get<double>()));
      if (std::string(name) == "tree_zero_cost_cycle") {
        DOCTEST_CHECK(result["tree_dag_cost"] == 2);
        DOCTEST_CHECK(result["tree_cost"] == 4);
      }
    }
  }
}
TEST_CASE("GPU CP-SAT shares checked model and fallback with CPU") {
  XEG graph; graph.read(fixture("cpu_shared_choice"));
  for (const auto kernel : {DagKernel::Thread,DagKernel::Warp,DagKernel::Block}) {
    const uint32_t threads = kernel == DagKernel::Thread ? 1 : 64;
    const uint32_t bits = kernel == DagKernel::Thread ? 0 : 512;
    auto result = graph.run_gpu(30,kernel,threads,bits,1,0.05);
    DOCTEST_CHECK(result["solution"]["objective"] == 6);
    DOCTEST_CHECK(result["solution"]["validated"] == true);
    DOCTEST_CHECK(result["solver"]["level_constraints"] == false);
  }
  graph.read(fixture("batch_then_direct_switch"));
  auto result = graph.run_gpu(30,DagKernel::Block,64,512,1,1e-12);
  DOCTEST_CHECK(result["solver"]["status"] == "UNKNOWN");
  DOCTEST_CHECK(result["solution"]["source"] == "tree_fallback");
  DOCTEST_CHECK(result["solution"]["objective"] == result["tree_dag_cost"]);
  DOCTEST_CHECK_THROWS_AS(graph.run_gpu(30,DagKernel::Thread,1,0,8,1),std::invalid_argument);
}
TEST_CASE("Repeated GPU extraction releases device buffers") {
  XEG graph; graph.read(fixture("batch_then_direct_switch"));
  graph.run_gpu(0,DagKernel::Block,64,512);
  size_t before = 0, after = 0, total = 0;
  DOCTEST_REQUIRE(cudaMemGetInfo(&before,&total) == cudaSuccess);
  for (int i = 0; i < 8; ++i) graph.run_gpu(0,DagKernel::Block,64,512);
  DOCTEST_REQUIRE(cudaMemGetInfo(&after,&total) == cudaSuccess);
  DOCTEST_CHECK(after + 65536 >= before);
}
