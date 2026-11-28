#pragma once
#include <cassert>
#include <unordered_map>
namespace flash_gpgpu_sim {
// Register-level pending writes aggregate multiple dynamic producers. They
// cannot identify which instruction's final response releases a native token.
class load_response_counter {
public:
  void expect(unsigned uid, unsigned responses) {
    assert(responses && !pending_.count(uid));
    pending_.emplace(uid, responses);
  }
  bool retire(unsigned uid) {
    auto it = pending_.find(uid);
    assert(it != pending_.end() && it->second);
    if (--it->second)
      return false;
    pending_.erase(it);
    return true;
  }
  bool pending(unsigned uid) const { return pending_.count(uid) != 0; }

private:
  std::unordered_map<unsigned, unsigned> pending_;
};
} // namespace flash_gpgpu_sim
