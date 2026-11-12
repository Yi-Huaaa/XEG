#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest.h>
#include <g_ext/G_EXT.hpp>

TEST_CASE("GPU tree representatives are optimal acyclic and repeatable") {
  nlohmann::json first;
  for(int i=0;i<5;++i) {
    G_EXT graph;
    graph.read(std::string(G_EXT_TEST_DATA_DIR)+"/batch_then_direct_switch.json");
    nlohmann::json result;
    REQUIRE_NOTHROW(result=graph.unittest_gpu_tree());
    if(i==0)first=result;
    CHECK(result==first);
  }
}
TEST_CASE("GPU zero cost ties and repeated roots form a finite unique DAG") {
  G_EXT graph;
  graph.read(std::string(G_EXT_TEST_DATA_DIR)+"/tree_zero_cost_cycle.json");
  nlohmann::json result;
  REQUIRE_NOTHROW(result=graph.unittest_gpu_tree());
  CHECK(result["tree_cost"].get<double>()==4);
  CHECK(result["dag_cost"].get<double>()==2);
}
