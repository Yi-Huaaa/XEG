#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest.h>
#include <xeg/XEG.hpp>

TEST_CASE("CPU deterministic solving preserves final cost across construction threads") {
  XEG graph;
  graph.read(std::string(XEG_TEST_DATA_DIR)+"/tree_zero_cost_cycle.json");
  auto baseline=graph.run_cpu(1,30,1,0.05);
  auto parallel=graph.run_cpu(8,30,1,0.05);
  DOCTEST_CHECK(baseline["solution"]["objective"]==2);
  DOCTEST_CHECK(parallel["solution"]["objective"]==baseline["solution"]["objective"]);
  DOCTEST_CHECK(parallel["solution"]["selected_enodes"]==baseline["solution"]["selected_enodes"]);
  DOCTEST_CHECK(parallel["filtered_hash"]==baseline["filtered_hash"]);
  DOCTEST_CHECK(parallel["solver"]["level_constraints"]==false);
}
TEST_CASE("CPU solver timeout retains a validated feasible tree") {
  XEG graph;
  graph.read(std::string(XEG_TEST_DATA_DIR)+"/batch_then_direct_switch.json");
  auto report=graph.run_cpu(2,30,1,1e-12);
  DOCTEST_CHECK(report["solver"]["status"]=="UNKNOWN");
  DOCTEST_CHECK(report["solution"]["source"]=="tree_fallback");
  DOCTEST_CHECK(report["solution"]["objective"]==201);
  DOCTEST_CHECK(report["solution"]["validated"]==true);
}
TEST_CASE("CPU deterministic mode requires a single solver worker") {
  XEG graph;
  graph.read(std::string(XEG_TEST_DATA_DIR)+"/tree_zero_cost_cycle.json");
  DOCTEST_CHECK_THROWS_AS(graph.run_cpu(1,30,8,1),std::invalid_argument);
  DOCTEST_CHECK_THROWS_AS(graph.run_cpu(1,0,1,1),std::invalid_argument);
}
TEST_CASE("CPU retains a better seed when a short solve has a worse incumbent") {
  XEG graph;
  graph.read(std::string(XEG_TEST_DATA_DIR)+"/batch_then_direct_switch.json");
  auto report=graph.run_cpu(2,30,1,1);
  DOCTEST_CHECK(report["solution"]["objective"].get<double>()<=report["tree_dag_cost"].get<double>());
  if (report["solver"].contains("objective")) {
    DOCTEST_CHECK(report["solution"]["objective"].get<double>()<=report["solver"]["objective"].get<double>());
    if (report["solver"]["objective"].get<double>()>report["tree_dag_cost"].get<double>())
      DOCTEST_CHECK(report["solution"]["reason"]=="better_than_solver_incumbent");
  }
}
TEST_CASE("validated CPU DAG solves shared choices without level variables") {
  XEG graph;
  graph.read(std::string(XEG_TEST_DATA_DIR)+"/cpu_shared_choice.json");
  auto report=graph.run_cpu(4,30,1,0.05);
  DOCTEST_CHECK(report["tree_dag_cost"]==10);
  DOCTEST_CHECK(report["solution"]["objective"]==6);
  DOCTEST_CHECK(report["solver"]["status"]=="OPTIMAL");
  DOCTEST_CHECK(report["solver"]["level_constraints"]==false);
}
