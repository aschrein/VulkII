#define UTILS_TL_IMPL
#define UTILS_TL_IMPL_DEBUG
#include "script.hpp"
#include "utils.hpp"
//#include "rendering.hpp"
//#include "rendering_utils.hpp"

int main(int argc, char *argv[]) {
  (void)argc;
  (void)argv;
  struct TestNode {
    i32  key;
    i32  value;
    bool operator==(TestNode const &that) const { return key == that.key; }
    bool operator>(TestNode const &that) const { return key > that.key; }
    bool operator<(TestNode const &that) const { return key < that.key; }
  };
  {
    TMP_STORAGE_SCOPE;
    BinNode<TestNode> *root = NULL;
    ito(1000) {
      if (root == NULL)
        root = BinNode<TestNode>::create({(i32)i, (i32)i % 2});
      else
        root = root->put({(i32)i, (i32)i % 2});
    }
    // root->traverse([&](BinNode<TestNode> *node) {
    //  ito(node->depth) fprintf(stdout, " ");
    //  fprintf(stdout, "%i:%i\n", node->depth, node->key.key);
    //});
    ASSERT_DEBUG(NULL == root->lower_bound({-1, 0}));
    ASSERT_DEBUG(NULL == root->upper_bound({1000, 0}));
    ito(1000) {
      ASSERT_DEBUG(root->find({(i32)i, 0}));
      ASSERT_DEBUG(root->find({(i32)i, 0}) == root->lower_bound({(i32)i, 0}));
      ASSERT_DEBUG(root->find({(i32)i, 0}) == root->upper_bound({(i32)i, 0}));
    }
    ito(1000) {
      ASSERT_DEBUG(root);
      root = root->remove({(i32)i, 0});
    }
    ASSERT_DEBUG(root == NULL);
    ASSERT_DEBUG(get_tl()->allocated == 0);
  }
  {
    BinNode<i32> *root = NULL;
    root               = BinNode<i32>::create(0);
    root               = root->put(10);
    root               = root->put(-10);
    ito(1000) { root = root->put(-(i32)i); }
    ASSERT_DEBUG(root->find(-10) == root->lower_bound(-10));
    ASSERT_DEBUG(root->find(10) == root->lower_bound(10));
    ASSERT_DEBUG(root->find(10) == root->lower_bound(11));
    ASSERT_DEBUG(root->find(10) == root->lower_bound(12));
    ASSERT_DEBUG(root->find(10) == root->upper_bound(10));
    ASSERT_DEBUG(root->find(10) == root->upper_bound(9));
    ASSERT_DEBUG(root->find(10) == root->upper_bound(8));
    root->release();
    ASSERT_DEBUG(get_tl()->allocated == 0);
  }
  {
    BinNode<i32> *root = NULL;
    root               = BinNode<i32>::create(0);
    PCG pcg;
    // ito(1000) { root = root->put((pcg.next() % 100)); }
    ito(100) {
      root = root->put((pcg.next() % 100));
      TMP_STORAGE_SCOPE;
      static char buf[0x100];
      snprintf(buf, sizeof(buf), "dot_%05d.dot", i);
      BinNode<i32>::dump_dotgraph(root, stref_s(buf),
                                  [=](FILE *f, char const *fmt, BinNode<i32> *node) {
                                    static char buf[0x100];
                                    snprintf(buf, sizeof(buf), "%i", node->key);
                                    fprintf(f, fmt, (i32)strlen(buf), buf);
                                  });
    }
    // ito(1000) { root = root->put((i32)i % 30); }
    root->traverse_break([&](BinNode<i32> *node) {
      if (node->left) {
        ASSERT_DEBUG(node->left->key < node->key);
      }
      if (node->right) {
        ASSERT_DEBUG(node->key < node->right->key || node->key == node->right->key);
      }
      return BinNode<i32>::TRAVERSE_CONTINUE;
    });

    ASSERT_DEBUG(root->depth <= 12);
    root->release();
  }
  {
    static constexpr u32  N = 10000000;
    Util_Allocator        ual(N);
    Array<Pair<u32, u32>> allocs{};
    defer(allocs.release());
    PCG pcg;
    while (ual.get_free_space()) {
      u32 size   = pcg.next() % 1000 + 1;
      i32 offset = ual.alloc(0, size);
      if (offset >= 0) allocs.push({(u32)offset, size});
    }
    ASSERT_DEBUG(ual.get_free_space() == 0);
    u32 free_space = 0;
    ito(allocs.size) {
      ual.free(allocs[i].first, allocs[i].second);
      free_space += allocs[i].second;
      ASSERT_DEBUG(ual.get_free_space() == free_space);
    }
    ASSERT_DEBUG(ual.get_free_space() == N);
  }
  // Alignment tests
  {
    static constexpr u32  N = 10000000;
    Util_Allocator        ual(N);
    Array<Pair<u32, u32>> allocs{};
    defer(allocs.release());
    PCG pcg;
    int attempts = 10000;
    while (ual.get_free_space() && attempts--) {
      u32 size   = pcg.next() % 1000 + 1;
      i32 offset = ual.alloc(0x100, size);
      if (offset >= 0) allocs.push({(u32)offset, size});
    }
    u32 free_space = ual.get_free_space();
    ito(allocs.size) {
      ual.free(allocs[i].first, allocs[i].second);
      free_space += allocs[i].second;
      ASSERT_DEBUG(ual.get_free_space() == free_space);
    }
    ASSERT_DEBUG(ual.get_free_space() == N);
  }
  {
    static constexpr u32 N = 10000000;
    Util_Allocator       ual(N);
    ASSERT_DEBUG(ual.get_num_nodes() == 1);
    ual.alloc(0x100, N / 2);
    i32 offset = ual.alloc(0x100, 1);
    ual.free(0, N / 2);
    ual.free(offset, 1);
    ASSERT_DEBUG(ual.get_num_nodes() == 1);
    ASSERT_DEBUG(ual.alloc(0x100, N) == 0);
    ASSERT_DEBUG(ual.get_num_nodes() == 0);
    ASSERT_DEBUG(ual.get_free_space() == 0);
    ASSERT_DEBUG(get_tl()->allocated == 0);
  }
  return 0;
}
