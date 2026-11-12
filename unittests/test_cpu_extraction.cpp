#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest.h>
#include <xeg/cpu_extraction.hpp>
#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>

namespace {
struct Graph {
  std::vector<float> cost;
  std::vector<uint32_t> cls, children, ptr{0}, roots, adj, ap, inv, ip;
  std::vector<std::vector<uint32_t>> members;
  uint32_t add(uint32_t c, float value, std::vector<uint32_t> ch = {}) {
    uint32_t u = cost.size();
    cost.push_back(value); cls.push_back(c);
    if (members.size() <= c) members.resize(c + 1);
    members[c].push_back(u);
    children.insert(children.end(), ch.begin(), ch.end()); ptr.push_back(children.size());
    return u;
  }
  void finish() {
    std::vector<std::vector<uint32_t>> a(cost.size()), b(cost.size());
    for (uint32_t u = 0; u < cost.size(); ++u) {
      for (uint32_t k = ptr[u]; k < ptr[u + 1]; ++k)
        for (uint32_t v : members[children[k]]) if (u != v) a[u].push_back(v);
      std::sort(a[u].begin(), a[u].end());
      a[u].erase(std::unique(a[u].begin(), a[u].end()), a[u].end());
      for (uint32_t v : a[u]) b[v].push_back(u);
    }
    adj.clear(); inv.clear(); ap={0}; ip={0};
    for (size_t u=0;u<cost.size();++u) {
      adj.insert(adj.end(),a[u].begin(),a[u].end());ap.push_back(adj.size());
      inv.insert(inv.end(),b[u].begin(),b[u].end());ip.push_back(inv.size());
    }
  }
  xeg::CpuGraph view() const {return {cost,cls,children,ptr,members,roots,adj,ap,inv,ip};}
};
// Independent exhaustive relaxation for costs, without the production
// frontier, representative tie-breaking, or OpenMP machinery.
std::vector<double> reference_costs(const Graph& g) {
  std::vector<double> c(g.members.size(),std::numeric_limits<double>::infinity());
  bool changed=true;
  while(changed) {
    changed=false;
    for(size_t u=0;u<g.cost.size();++u) {
      double v=g.cost[u];
      for(uint32_t k=g.ptr[u];k<g.ptr[u+1];++k)v+=c[g.children[k]];
      if(v<c[g.cls[u]]){c[g.cls[u]]=v;changed=true;}
    }
  }
  return c;
}
std::vector<uint32_t> reference_best(const Graph& g) {
  std::vector<double> c(g.members.size(),std::numeric_limits<double>::infinity());
  std::vector<uint32_t> best(g.members.size(),UINT32_MAX);
  bool changed=true;
  while(changed) {
    changed=false;
    for(uint32_t u=0;u<g.cost.size();++u) {
      double v=g.cost[u];
      for(uint32_t k=g.ptr[u];k<g.ptr[u+1];++k)v+=c[g.children[k]];
      if(v<c[g.cls[u]]){c[g.cls[u]]=v;best[g.cls[u]]=u;changed=true;}
    }
  }
  return best;
}
void compare(const Graph& g) {
  auto a=xeg::cpu_tree(g.view(),1);
  CHECK(a.class_cost==reference_costs(g));
  CHECK(a.best==reference_best(g));
  CHECK(xeg::validate_cpu_solution(g.view(),a.selected)==doctest::Approx(a.dag_cost));
  auto d=xeg::cpu_dag(g.view(),a.selected,1);
  CHECK_NOTHROW(xeg::validate_cpu_dag(g.view(),d.selected));
  CHECK_NOTHROW(xeg::validate_cpu_candidates(g.view(),d.in_subgraph));
  for(size_t u=0;u<a.selected.size();++u)
    if(a.selected[u]) CHECK(d.in_subgraph[u]);
  for(int threads:{2,4,8}) {
    auto b=xeg::cpu_tree(g.view(),threads);
    CHECK(b.class_cost==a.class_cost);CHECK(b.node_cost==a.node_cost);
    CHECK(b.best==a.best);CHECK(b.selected==a.selected);
    auto e=xeg::cpu_dag(g.view(),b.selected,threads);
    CHECK(e.selected==d.selected);CHECK(e.in_subgraph==d.in_subgraph);
  }
}
}
TEST_CASE("zero-cost cycles use finite ordered-discovery representatives") {
  Graph g;
  g.add(0,0,{1});g.add(0,0);g.add(1,0,{0});g.add(1,0);
  g.roots={0,1};g.finish();compare(g);
  auto t=xeg::cpu_tree(g.view(),4);
  CHECK(t.best==std::vector<uint32_t>{1,2});
}
TEST_CASE("shared children and repeated roots count once in extracted DAG") {
  Graph g;g.add(0,3,{1,1});g.add(1,2);g.roots={0,0};g.finish();
  auto t=xeg::cpu_tree(g.view(),1);
  CHECK(t.tree_cost==14);CHECK(t.dag_cost==5);compare(g);
}
TEST_CASE("unproductive roots and invalid costs fail explicitly") {
  Graph g;g.add(0,0,{0});g.roots={0};g.finish();
  CHECK_THROWS_AS(xeg::cpu_tree(g.view(),1),std::runtime_error);
  g.cost[0]=-1;CHECK_THROWS_AS(xeg::cpu_tree(g.view(),1),std::invalid_argument);
  g.cost[0]=std::numeric_limits<float>::quiet_NaN();
  CHECK_THROWS_AS(xeg::cpu_tree(g.view(),1),std::invalid_argument);
  CHECK_THROWS_AS(xeg::cpu_tree(g.view(),0),std::invalid_argument);
}
TEST_CASE("validator checks roots closure choices and e-class self loops") {
  Graph g;g.add(0,1,{1});g.add(0,2);g.add(1,0,{1});g.add(1,1);g.roots={0};g.finish();
  CHECK_THROWS(xeg::validate_cpu_solution(g.view(),{0,0,0,1}));
  CHECK_THROWS(xeg::validate_cpu_solution(g.view(),{1,0,0,0}));
  CHECK_THROWS(xeg::validate_cpu_solution(g.view(),{1,1,0,1}));
  CHECK_THROWS(xeg::validate_cpu_solution(g.view(),{1,0,1,0}));
  CHECK(xeg::validate_cpu_solution(g.view(),{1,0,0,1})==2);
}
TEST_CASE("random cyclic egraphs agree with independent cost relaxation") {
  std::mt19937 random(42);
  for(int trial=0;trial<60;++trial) {
    Graph g; const uint32_t count=8+random()%25;
    for(uint32_t c=0;c<count;++c) {
      g.add(c,1+random()%10);
      for(int a=0;a<3;++a) {
        std::vector<uint32_t> children;
        for(uint32_t k=0,arity=random()%4;k<arity;++k)children.push_back(random()%count);
        g.add(c,random()%5,children);
      }
    }
    g.roots={0,count-1};g.finish();compare(g);
  }
}
TEST_CASE("parallel frontiers and word-adjacent masks are deterministic") {
  Graph g;
  for(uint32_t c=0;c<2200;++c) {
    g.add(c,3);
    if(c>=1100)g.add(c,0,{c-1100});
    g.roots.push_back(c);
  }
  g.finish();compare(g);
}
TEST_CASE("one graph can be constructed repeatedly with different seeds") {
  Graph g;g.add(0,1,{1});g.add(0,4);g.add(1,1);g.add(1,2);g.roots={0};g.finish();
  auto a=xeg::cpu_dag(g.view(),{1,0,1,0},2);
  auto b=xeg::cpu_dag(g.view(),{0,1,0,0},4);
  auto c=xeg::cpu_dag(g.view(),{1,0,1,0},1);
  CHECK(a.selected==c.selected);CHECK(a.in_subgraph==c.in_subgraph);
  CHECK_NOTHROW(xeg::validate_cpu_dag(g.view(),b.selected));
}
TEST_CASE("parallel ordered insertion propagates conflicts between new candidates") {
  Graph g;
  for(uint32_t i=0;i<1200;++i) {
    const uint32_t c=3*i;
    g.add(c,1,{c+1,c+2});g.add(c+1,1);g.add(c+2,1);
    g.add(c+1,1,{c+2});g.add(c+2,1,{c+1});g.roots.push_back(c);
  }
  g.finish();compare(g);
  auto tree=xeg::cpu_tree(g.view(),8);
  auto dag=xeg::cpu_dag(g.view(),tree.selected,8);
  CHECK(dag.neighbor_updates>0);
  for(uint32_t i=0;i<1200;++i) {
    CHECK(dag.selected[5*i+3]);
    CHECK_FALSE(dag.selected[5*i+4]);
  }
}
TEST_CASE("rounded cost ties still have an acyclic finite witness") {
  Graph g;
  const float large=std::ldexp(1.0f,80);
  g.add(0,1,{0});g.add(0,large);g.add(1,large,{0});
  g.roots={1};g.finish();compare(g);
}
TEST_CASE("candidate closure removes missing children and cascading parents") {
  Graph g;
  g.add(0,1,{1});g.add(0,7);g.add(1,1,{2});g.add(2,1);
  g.roots={0};g.finish();
  for (int threads : {1,8}) {
    std::vector<uint8_t> candidates{1,1,1,0};
    CHECK_THROWS(xeg::validate_cpu_candidates(g.view(),candidates));
    CHECK(xeg::prune_cpu_candidates(g.view(),candidates,threads)==2);
    CHECK(candidates==std::vector<uint8_t>{0,1,0,0});
    CHECK_NOTHROW(xeg::validate_cpu_candidates(g.view(),candidates));
  }
}
TEST_CASE("closure keeps a child class with another valid choice") {
  Graph g;
  g.add(0,1,{1});g.add(1,1,{2});g.add(1,2);g.add(2,1);
  g.roots={0};g.finish();
  std::vector<uint8_t> candidates{1,1,1,0};
  CHECK(xeg::prune_cpu_candidates(g.view(),candidates,1)==1);
  CHECK(candidates==std::vector<uint8_t>{1,0,1,0});
  CHECK_NOTHROW(xeg::validate_cpu_candidates(g.view(),candidates));
}
TEST_CASE("root-to-leaf reachability alone does not imply child closure") {
  Graph g;
  const auto invalid=g.add(0,1,{1,2});g.add(0,10);g.add(1,1);
  g.add(2,0,{2});g.add(3,1,{1});g.roots={0,3};g.finish();
  auto tree=xeg::cpu_tree(g.view(),1);
  auto dag=xeg::cpu_dag(g.view(),tree.selected,1);
  CHECK(dag.selected[invalid]); // It has a path to a leaf through class 1.
  CHECK_FALSE(dag.in_subgraph[invalid]); // Class 2 has no finite choice.
  CHECK(dag.invalid_candidates_removed>0);
  CHECK_NOTHROW(xeg::validate_cpu_candidates(g.view(),dag.in_subgraph));
  compare(g);
}
TEST_CASE("closure excludes e-nodes that depend on their own class") {
  Graph g;g.add(0,0,{0});g.add(0,1);g.roots={0};g.finish();
  std::vector<uint8_t> candidates{1,1};
  CHECK_THROWS(xeg::validate_cpu_candidates(g.view(),candidates));
  CHECK(xeg::prune_cpu_candidates(g.view(),candidates,1)==1);
  CHECK(candidates==std::vector<uint8_t>{0,1});
  CHECK_NOTHROW(xeg::validate_cpu_candidates(g.view(),candidates));
}
