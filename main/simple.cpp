#include <chrono>
#include <iostream>
#include <gegg/EGG.hpp>
// #define max_iterations 4

double round_to(double value, double precision = 1.0) {
  return std::round(value / precision) * precision;
}

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cerr << "Usage: cegg <path/to/file.eqn> max_iterations [sexpr_out.txt]\n";
    std::cerr << "./main/simple ../../E-Syn/converted_circuit_strash/Adders/01-adder.eqn 5 output.eqn\n";
    return 1;
  }
  
  // Timer
  std::chrono::duration<double> duration;

  // From user's perspective: test CPU function 
  egraph::EGraph egraph;
  std::string eqn_path = argv[1];

  // from .eqn to S-expr
  std::vector<std::string> all_sexprs = egraph.EqnToSexprAll(eqn_path);
  // printf("all_sexprs.size() = %lu\n", all_sexprs.size());
  for (size_t i = 0; i < all_sexprs.size(); i++) {
    // std::cout  << "S-Expr[" << i<< "]: " << all_sexprs[i] << "\n"; // debug log 
    egraph.parse_sexpr(all_sexprs[i], egraph);
  }
  // egraph.print_roots(); // debug log 

  // printf("\n=================== Before EQST starts =====================\n");
  // egraph.print_egraph();
  // egraph.print_hashcons();
  // egraph.dump_dot("./gv_files/dp_egraph.gv");
  // printf("\n=================== Before EQST ends =====================\n\n");

  const double timeout = 10000; // ms
  const std::vector<egraph::RewriteRule>& rules = get_boolean_rewriting_rules(); 
  std::vector<double> cost_time(3, 0.0);
  const int max_iterations = static_cast<uint32_t>(std::atoi(argv[2]));
    
  // Exploration
  auto start = std::chrono::steady_clock::now();
    egraph::EqStEngine EQST;
    uint iter = 0;
    EQST.equality_saturation(egraph, max_iterations, timeout, rules, iter, cost_time); 
  auto end = std::chrono::steady_clock::now();
  duration = end - start;
  std::cout << "CPU time: " << round_to((duration.count()) * 1000, 0.001) << " ms\n";
  std::cout << "iter = " << iter << " (max_iterations = " << max_iterations << ")\n"; 
  std::cout << "Collect matches time: " << cost_time[0] << " ms\n";
  std::cout << "Apply matches time: "  << cost_time[1] << " ms\n";
  std::cout << "Rebuild time: "         << cost_time[2] << " ms\n";
  std::cout << "Number of eclasses = " << egraph.num_ecls() 
            << ", Num of enodes = " << egraph.num_enodes() << "\n";


  // // Extraction
  // egraph.rebuild();
  // egraph.extraction();  // This must be run before find_best_cost/expr
  // // egraph.print_egraph_in_details();

  // uint32_t cost = egraph.find_best_cost();
  // std::string best_expr = egraph.find_best_expr();
  // std::cout << "\n==================== Greedy Extraction Result ====================\n";
  // std::cout << "Original input: " << input << "\n"; 
  // std::cout << "Best Cost: " << cost << ", Expression: " << best_expr << "\n";
  
  // // printf("\n\n");
  // // printf("==================== After EQST =====================\n");
  // // std::pair<float, float> res;
  // // bool correctness = egraph.check_answer(input, best_expr, res);    

  // // 如果有指定存檔
  // if (argc >= 3) {
  //   std::string output_name= argv[2];
  //   std::ofstream fout(argv[2]);
  //   for (auto& s : all_sexprs) fout << s << "\n";
  // }

  return 0;
}