/* trunk-ignore-all(clang-format) */
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <doctest.h>
#include <string>
#include <g_ext/G_EXT.hpp>


// TEST_CASE("Example with REQUIRE and MESSAGE") {
//   int x = 3;
//   int y = 4;
  
//   CHECK(x + y == 7);
//   REQUIRE(x != 0);  // 如果 x==0，整個測試直接中斷
//   MESSAGE("Current x value: " << x);  // 輔助 debug 用，失敗時才會印出
//   CHECK(y > x);
// }

// TEST_CASE("Doctest assertion examples" * doctest::timeout(10)) {
//   int x = 5, y = 10;
//   std::string msg = "testing";

//   MESSAGE("Starting the test case...");

//   CHECK(x + y == 15);  // 普通非致命 CHECK (錯了會繼續跑)
//   CHECK_EQ(x + y, 15);  // 確認等於 (等同 CHECK(x + y == 15))
//   CAPTURE(x); CAPTURE(y);  // 捕捉變數內容 (失敗時自動 print)
//   INFO("x and y captured for debugging...");  // INFO 也一樣，失敗時 print
//   // CHECK(x - y == 0);  // 故意錯的 (5 - 10 ≠ 0)  // // 故意一個錯的，讓我們能看到 CAPTURE 和 MESSAGE 效果
//   REQUIRE(x != 0);  // // error example: 強制 REQUIRES，錯了直接中止整個 testcase
//   REQUIRE_EQ(x, 5);  // 再來一個 REQUIRES 等於判斷 (這個會過)
//   if (x < 0) { FAIL("x should not be negative!"); }  // 有時你需要手動報錯
//   if (false) { FAIL("This won't trigger"); }  // 故意用 FAIL 範例
//   MESSAGE("Finished all assertions for this test!");
// }

// TEST_CASE("math_associate_adds" * doctest::timeout(300)) {
//   // Construct rules 
//   std::vector<RewriteRule> rules;
//     rules.emplace_back("comm-add",
//       Op(NodeKind::Add, { Var("a"), Var("b") }),
//       Op(NodeKind::Add, { Var("b"), Var("a") })
//     );  
//     rules.emplace_back("assoc-add",
//       Op(NodeKind::Add, { Var("a"), Op(NodeKind::Add, { Var("b"), Var("c") }) }),
//       Op(NodeKind::Add, { Op(NodeKind::Add, { Var("a"), Var("b") }), Var("c") })
//     );

//   // Construct egraph 
//   EGraph egraph;
//   const std::string input = "(+ 1 (+ 2 (+ 3 (+ 4 (+ 5 (+ 6 7))))))"; 
//   EClass* root = egraph.parse_sexpr(input, egraph);
  
//   // Step 3: Run the rewrite engine
//   const uint32_t max_iterations = 7; 
//   const double timeout = 3000; // ms
//   EqStEngine EQST;
//   EQST.equality_saturation(egraph, max_iterations, timeout, rules, true);
  
//   egraph.extraction();
//   uint32_t cost = egraph.find_best_cost(); // Aborted
//   std::string best_expr = egraph.find_best_expr();

//   // used to check the correctness of the answer 
//   std::pair<float, float> res;
//   bool correctness = egraph.check_answer(input, best_expr, res);

//   CHECK(correctness == true);
//   CHECK(res.second == 28);
// }

// TEST_CASE("math_simplify_add" * doctest::timeout(3000)) {
//   const std::vector<RewriteRule>& rules = get_basic_add_rules();
//   CAPTURE(rules.size());

//   // Construct egraph 
//   EGraph egraph;
//   const std::string input = "(+ x (+ x (+ x x)))";
//   EClass* root = egraph.parse_sexpr(input, egraph);   
  
//   // Step 3: Run the rewrite engine
//   const uint32_t max_iterations = 4; 
//   const double timeout = 3000; // ms
//   EqStEngine EQST;
//   EQST.equality_saturation(egraph, max_iterations, timeout, rules, true); 
  
//   egraph.extraction();
//   uint32_t cost = egraph.find_best_cost(); // Aborted
//   std::string best_expr = egraph.find_best_expr();

//   CAPTURE(best_expr);
//   CHECK(best_expr == "(* x 4)");
// }

// TEST_CASE("math_powers" * doctest::timeout(3000)) {
//   const std::vector<RewriteRule>& rules = get_basic_add_rules();

//   // Construct egraph 
//   EGraph egraph;
//   const std::string input = "(* (pow 2 x) (pow 2 y))";
//   EClass* root = egraph.parse_sexpr(input, egraph);
  
//   // Step 3: Run the rewrite engine
//   const uint32_t max_iterations = 4; 
//   const double timeout = 3000; // ms
//   EqStEngine EQST;
//   EQST.equality_saturation(egraph, max_iterations, timeout, rules, true); 
  
//   egraph.extraction();
//   uint32_t cost = egraph.find_best_cost(); // Aborted
//   std::string best_expr = egraph.find_best_expr();

//   CHECK(best_expr == "(pow 2 (+ x y))");
// }

// TEST_CASE("math_powers" * doctest::timeout(3000)) {
//   const std::vector<RewriteRule>& rules = get_basic_add_rules();

//   // Construct egraph 
//   EGraph egraph;
//   const std::string input = "(+ 1 (- a (* (- 2 1) a)))";
//   EClass* root = egraph.parse_sexpr(input, egraph);
  
//   // Step 3: Run the rewrite engine
//   const uint32_t max_iterations = 4; 
//   const double timeout = 3000; // ms
//   EqStEngine EQST;
//   EQST.equality_saturation(egraph, max_iterations, timeout, rules, true); 
  
//   egraph.extraction();
//   uint32_t cost = egraph.find_best_cost(); // Aborted
//   std::string best_expr = egraph.find_best_expr();

//   CHECK(best_expr == "1");
// }

// TEST_CASE("math_diff_same" * doctest::timeout(3000)) {
//   const std::vector<RewriteRule>& rules = get_basic_add_rules();

//   // Construct egraph 
//   EGraph egraph;
//   const std::string input = "(d x x)";
//   EClass* root = egraph.parse_sexpr(input, egraph);
  
//   // Step 3: Run the rewrite engine
//   const uint32_t max_iterations = 1; 
//   const double timeout = 3000; // ms
//   EqStEngine EQST;
//   EQST.equality_saturation(egraph, max_iterations, timeout, rules, true); 
  
//   egraph.extraction();
//   uint32_t cost = egraph.find_best_cost(); // Aborted
//   std::string best_expr = egraph.find_best_expr();

//   CHECK(best_expr == "1");
// }

// TEST_CASE("math_diff_different" * doctest::timeout(3000)) {
//   const std::vector<RewriteRule>& rules = get_basic_add_rules();

//   // Construct egraph 
//   EGraph egraph;
//   const std::string input = "(d x y)";
//   EClass* root = egraph.parse_sexpr(input, egraph);
  
//   // Step 3: Run the rewrite engine
//   const uint32_t max_iterations = 1; 
//   const double timeout = 3000; // ms
//   EqStEngine EQST;
//   EQST.equality_saturation(egraph, max_iterations, timeout, rules, true); 
  
//   egraph.extraction();
//   uint32_t cost = egraph.find_best_cost(); // Aborted
//   std::string best_expr = egraph.find_best_expr();

//   CHECK(best_expr == "0");
// }

// TEST_CASE("math_diff_simple1" * doctest::timeout(3000)) {
//   const std::vector<RewriteRule>& rules = get_basic_add_rules();

//   // Construct egraph 
//   EGraph egraph;
//   const std::string input = "(d x (+ 1 (* 2 x)))";
//   EClass* root = egraph.parse_sexpr(input, egraph);
  
//   // Step 3: Run the rewrite engine
//   const uint32_t max_iterations = 3; 
//   const double timeout = 3000; // ms
//   EqStEngine EQST;
//   EQST.equality_saturation(egraph, max_iterations, timeout, rules, true); 
  
//   egraph.extraction();
//   uint32_t cost = egraph.find_best_cost(); // Aborted
//   std::string best_expr = egraph.find_best_expr();

//   CHECK(best_expr == "2");
// }

// TEST_CASE("math_diff_simple2" * doctest::timeout(3000)) {
//   const std::vector<RewriteRule>& rules = get_basic_add_rules();

//   // Construct egraph 
//   EGraph egraph;
//   const std::string input = "(d x (+ 1 (* y x)))";
//   EClass* root = egraph.parse_sexpr(input, egraph);
  
//   // Step 3: Run the rewrite engine
//   const uint32_t max_iterations = 4; 
//   const double timeout = 3000; // ms
//   EqStEngine EQST;
//   EQST.equality_saturation(egraph, max_iterations, timeout, rules, true); 
  
//   egraph.extraction();
//   uint32_t cost = egraph.find_best_cost(); // Aborted
//   std::string best_expr = egraph.find_best_expr();

//   CHECK(best_expr == "y");
// }

// TEST_CASE("math_diff_ln" * doctest::timeout(3000)) {
//   const std::vector<RewriteRule>& rules = get_basic_add_rules();

//   // Construct egraph 
//   EGraph egraph;
//   const std::string input = "(d x (ln x))";
//   EClass* root = egraph.parse_sexpr(input, egraph);
  
//   // Step 3: Run the rewrite engine
//   const uint32_t max_iterations = 6;
//   const double timeout = 3000; // ms
//   EqStEngine EQST;
//   EQST.equality_saturation(egraph, max_iterations, timeout, rules, true); 
  
//   egraph.extraction();
//   uint32_t cost = egraph.find_best_cost(); // Aborted
//   std::string best_expr = egraph.find_best_expr();

//   bool correct = (best_expr == "(/ 1 x)" || best_expr == "(pow x -1)");
//   CHECK(correct);
// }

// TEST_CASE("integ_one" * doctest::timeout(3000)) {
//   const std::vector<RewriteRule>& rules = get_basic_add_rules();

//   // Construct egraph 
//   EGraph egraph;
//   const std::string input = "(i 1 x)";
//   EClass* root = egraph.parse_sexpr(input, egraph);
  
//   // Step 3: Run the rewrite engine
//   const uint32_t max_iterations = 1; 
//   const double timeout = 3000; // ms
//   EqStEngine EQST;
//   EQST.equality_saturation(egraph, max_iterations, timeout, rules, true); 
  
//   egraph.extraction();
//   uint32_t cost = egraph.find_best_cost(); // Aborted
//   std::string best_expr = egraph.find_best_expr();

//   CHECK(best_expr == "x");
// }

// TEST_CASE("integ_sin" * doctest::timeout(3000)) {
//   const std::vector<RewriteRule>& rules = get_basic_add_rules();

//   // Construct egraph 
//   EGraph egraph;
//   const std::string input = "(i (cos x) x)";
//   EClass* root = egraph.parse_sexpr(input, egraph);
  
//   // Step 3: Run the rewrite engine
//   const uint32_t max_iterations = 1; 
//   const double timeout = 3000; // ms
//   EqStEngine EQST;
//   EQST.equality_saturation(egraph, max_iterations, timeout, rules, true); 
  
//   egraph.extraction();
//   uint32_t cost = egraph.find_best_cost(); // Aborted
//   std::string best_expr = egraph.find_best_expr();

//   CHECK(best_expr == "(sin x)");
// }

// TEST_CASE("integ_x" * doctest::timeout(3000)) {
//   const std::vector<RewriteRule>& rules = get_basic_add_rules();

//   // Construct egraph 
//   EGraph egraph;
//   const std::string input = "(i (pow x 1) x)";
//   EClass* root = egraph.parse_sexpr(input, egraph);
  
//   // Step 3: Run the rewrite engine
//   const uint32_t max_iterations = 3; 
//   const double timeout = 3000; // ms
//   EqStEngine EQST;
//   EQST.equality_saturation(egraph, max_iterations, timeout, rules, true); 
  
//   egraph.extraction();
//   uint32_t cost = egraph.find_best_cost(); // Aborted
//   std::string best_expr = egraph.find_best_expr();

//   CHECK(best_expr == "(/ (pow x 2) 2)");
// }

// TEST_CASE("integ_part1" * doctest::timeout(3000)) {
//   const std::vector<RewriteRule>& rules = get_basic_add_rules();

//   // Construct egraph 
//   EGraph egraph;
//   const std::string input = "(i (* x (cos x)) x)";
//   EClass* root = egraph.parse_sexpr(input, egraph);
  
//   // Step 3: Run the rewrite engine
//   const uint32_t max_iterations = 5; 
//   const double timeout = 3000; // ms
//   EqStEngine EQST;
//   EQST.equality_saturation(egraph, max_iterations, timeout, rules, true); 
  
//   egraph.extraction();
//   uint32_t cost = egraph.find_best_cost(); // Aborted
//   std::string best_expr = egraph.find_best_expr();

//   bool correct = (best_expr == "(+ (* (sin x) x) (cos x))" || best_expr == "(+ (* x (sin x)) (cos x))");
//   CAPTURE(best_expr);
//   CHECK(correct);
// }

// TEST_CASE("integ_part3" * doctest::timeout(3000)) {
//   const std::vector<RewriteRule>& rules = get_basic_add_rules();

//   // Construct egraph 
//   EGraph egraph;
//   const std::string input = "(i (ln x) x)";
//   EClass* root = egraph.parse_sexpr(input, egraph);
  
//   // Step 3: Run the rewrite engine
//   const uint32_t max_iterations = 4; 
//   const double timeout = 3000; // ms
//   EqStEngine EQST;
//   EQST.equality_saturation(egraph, max_iterations, timeout, rules, true); 
  
//   egraph.extraction();
//   uint32_t cost = egraph.find_best_cost(); // Aborted
//   std::string best_expr = egraph.find_best_expr();

//   bool correct = (best_expr == "(- (* (ln x) x) x)" || best_expr == "(- (* x (ln x)) x)");
//   CHECK(correct);
// }



