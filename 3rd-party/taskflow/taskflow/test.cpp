#include "taskflow.hpp"

int main() {

  // Declare an executor and a taskflow
  tf::Executor executor;
  tf::Taskflow taskflow;

  // Add three tasks into the taskflow
  tf::Task A = taskflow.emplace([] () { std::cout << "This is TaskA\n"; });
  tf::Task B = taskflow.emplace([] () { std::cout << "This is TaskB\n"; });
  tf::Task C = taskflow.emplace([] () { std::cout << "This is TaskC\n"; });

  // Build precedence between tasks
  A.succeed(B, C);

  executor.run_n(taskflow, 4);
  executor.wait_for_all();  // block until all associated executions finish

  return 0;
}
