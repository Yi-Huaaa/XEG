/* trunk-ignore-all(clang-format) */
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <doctest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <g_ext/G_EXT.hpp>

namespace {

std::string test_data_path(
  const std::string& filename)
{
  return std::string(G_EXT_TEST_DATA_DIR)
    + "/"
    + filename;
}

template <typename T>
void check_vector_equal(
  const std::vector<T>& actual,
  const std::vector<T>& expected)
{
  DOCTEST_REQUIRE(actual.size() == expected.size());

  for (size_t i = 0; i < actual.size(); ++i) {
    DOCTEST_CAPTURE(i);
    DOCTEST_CHECK(actual[i] == expected[i]);
  }
}

void check_csr_sorted_and_unique(
  const std::vector<uint32_t>& adjacency,
  const std::vector<uint32_t>& pointer)
{
  DOCTEST_REQUIRE_FALSE(pointer.empty());

  DOCTEST_CHECK(pointer.front() == 0u);
  DOCTEST_CHECK(
    pointer.back()
    == static_cast<uint32_t>(adjacency.size()));

  for (size_t node = 0;
       node + 1 < pointer.size();
       ++node) {
    DOCTEST_CAPTURE(node);

    const uint32_t begin_index =
      pointer[node];

    const uint32_t end_index =
      pointer[node + 1];

    DOCTEST_REQUIRE(begin_index <= end_index);
    DOCTEST_REQUIRE(end_index <= adjacency.size());

    const auto begin =
      adjacency.begin() + begin_index;

    const auto end =
      adjacency.begin() + end_index;

    DOCTEST_CHECK(std::is_sorted(begin, end));

    DOCTEST_CHECK(
      std::adjacent_find(begin, end)
      == end);
  }
}

}  // namespace


TEST_CASE(
  "Forward and inverse CSR segments are sorted and unique")
{
  G_EXT g_ext;

  g_ext.read(
    test_data_path(
      "csr_unsorted_duplicate.json"));

  const auto csr =
    g_ext.unittest_get_csr_snapshot();

  check_csr_sorted_and_unique(
    csr.forward_adj,
    csr.forward_ptr);

  check_csr_sorted_and_unique(
    csr.inverse_adj,
    csr.inverse_ptr);

  const std::vector<uint32_t>
    expected_forward_adj = {
      1u, 2u, 4u, 5u,
      3u,
      3u,
      3u,
      3u
    };

  const std::vector<uint32_t>
    expected_forward_ptr = {
      0u,
      4u,
      5u,
      6u,
      6u,
      7u,
      8u
    };

  const std::vector<uint32_t>
    expected_inverse_adj = {
      0u,
      0u,
      1u, 2u, 4u, 5u,
      0u,
      0u
    };

  const std::vector<uint32_t>
    expected_inverse_ptr = {
      0u,
      0u,
      1u,
      2u,
      6u,
      7u,
      8u
    };

  check_vector_equal(
    csr.forward_adj,
    expected_forward_adj);

  check_vector_equal(
    csr.forward_ptr,
    expected_forward_ptr);

  check_vector_equal(
    csr.inverse_adj,
    expected_inverse_adj);

  check_vector_equal(
    csr.inverse_ptr,
    expected_inverse_ptr);
}


TEST_CASE(
  "Forward and inverse CSR describe the same edges")
{
  G_EXT g_ext;

  g_ext.read(
    test_data_path(
      "csr_unsorted_duplicate.json"));

  const auto csr =
    g_ext.unittest_get_csr_snapshot();

  DOCTEST_REQUIRE(
    csr.forward_ptr.size()
    == csr.inverse_ptr.size());

  const size_t num_nodes =
    csr.forward_ptr.size() - 1;

  // Every forward edge u -> v must appear as
  // parent u in inverse CSR segment v.
  for (size_t u = 0; u < num_nodes; ++u) {
    for (uint32_t index = csr.forward_ptr[u];
         index < csr.forward_ptr[u + 1];
         ++index) {
      const uint32_t v =
        csr.forward_adj[index];

      DOCTEST_CAPTURE(u);
      DOCTEST_CAPTURE(v);

      DOCTEST_REQUIRE(v < num_nodes);

      const auto inverse_begin =
        csr.inverse_adj.begin()
        + csr.inverse_ptr[v];

      const auto inverse_end =
        csr.inverse_adj.begin()
        + csr.inverse_ptr[v + 1];

      DOCTEST_CHECK(
        std::binary_search(
          inverse_begin,
          inverse_end,
          static_cast<uint32_t>(u)));
    }
  }

  // Every inverse entry u in segment v must
  // correspond to forward edge u -> v.
  for (size_t v = 0; v < num_nodes; ++v) {
    for (uint32_t index = csr.inverse_ptr[v];
         index < csr.inverse_ptr[v + 1];
         ++index) {
      const uint32_t u =
        csr.inverse_adj[index];

      DOCTEST_CAPTURE(u);
      DOCTEST_CAPTURE(v);

      DOCTEST_REQUIRE(u < num_nodes);

      const auto forward_begin =
        csr.forward_adj.begin()
        + csr.forward_ptr[u];

      const auto forward_end =
        csr.forward_adj.begin()
        + csr.forward_ptr[u + 1];

      DOCTEST_CHECK(
        std::binary_search(
          forward_begin,
          forward_end,
          static_cast<uint32_t>(v)));
    }
  }
}