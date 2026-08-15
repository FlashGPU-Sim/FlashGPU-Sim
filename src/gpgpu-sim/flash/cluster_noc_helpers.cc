// Standalone NoC helpers (no shader / gpu-sim link). Used by cluster_noc_t
// and by unit tests so ClusterNoc* exercises the shipped matrix/decode/drop.

#include "cluster_noc.h"

#include <cassert>
#include <cstdio>
#include <fstream>
#include <sstream>

namespace flash_gpgpu_sim {

void cluster_noc_latency_matrix::init(unsigned n_cores, unsigned local_lat,
                                      unsigned remote_lat) {
  m_n = n_cores;
  m_lat.assign(static_cast<size_t>(n_cores) * n_cores, remote_lat);
  for (unsigned i = 0; i < n_cores; i++) {
    m_lat[static_cast<size_t>(i) * n_cores + i] = local_lat;
  }
}

bool cluster_noc_latency_matrix::load_from_file(const std::string &path,
                                                unsigned n_cores) {
  std::ifstream in(path);
  if (!in) {
    printf("GPGPU-Sim WARNING: cluster NoC latency matrix file '%s' not found; "
           "using scalar defaults\n",
           path.c_str());
    return false;
  }
  std::vector<unsigned> vals;
  vals.reserve(static_cast<size_t>(n_cores) * n_cores);
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#' || line[0] == '/')
      continue;
    for (char &c : line) {
      if (c == ',')
        c = ' ';
    }
    std::istringstream iss(line);
    unsigned v;
    while (iss >> v) {
      vals.push_back(v);
    }
  }
  const size_t expect = static_cast<size_t>(n_cores) * n_cores;
  if (vals.size() != expect) {
    printf("GPGPU-Sim WARNING: cluster NoC matrix '%s' has %zu values, "
           "expected %zu (n_cores=%u); using scalar defaults\n",
           path.c_str(), vals.size(), expect, n_cores);
    return false;
  }
  m_n = n_cores;
  m_lat = std::move(vals);
  return true;
}

unsigned cluster_noc_latency_matrix::hop(unsigned src_cid,
                                         unsigned dst_cid) const {
  if (m_n == 0)
    return 0;
  assert(src_cid < m_n && dst_cid < m_n);
  return m_lat[static_cast<size_t>(src_cid) * m_n + dst_cid];
}

bool decode_shared_generic(addr_t addr, unsigned *out_smid,
                           addr_t *out_offset) {
  if (addr < SHARED_GENERIC_START)
    return false;
  const addr_t rel = addr - SHARED_GENERIC_START;
  if (rel >= TOTAL_SHARED_MEM)
    return false;
  const unsigned smid = static_cast<unsigned>(rel / SHARED_MEM_SIZE_MAX);
  const addr_t offset = static_cast<addr_t>(rel % SHARED_MEM_SIZE_MAX);
  if (out_smid)
    *out_smid = smid;
  if (out_offset)
    *out_offset = offset;
  return true;
}

bool is_remote_shared_generic(unsigned local_smid, addr_t addr,
                              unsigned *out_owner_smid, addr_t *out_offset) {
  unsigned owner = 0;
  addr_t off = 0;
  if (!decode_shared_generic(addr, &owner, &off))
    return false;
  if (owner == local_smid)
    return false;
  if (out_owner_smid)
    *out_owner_smid = owner;
  if (out_offset)
    *out_offset = off;
  return true;
}

size_t cluster_noc_drop_queue_to_cta(std::deque<cluster_noc_message> *q,
                                     unsigned cid, unsigned hw_cta) {
  if (!q)
    return 0;
  size_t dropped = 0;
  for (auto it = q->begin(); it != q->end();) {
    if (cluster_noc_msg_targets_cta(*it, cid, hw_cta)) {
      ++dropped;
      it = q->erase(it);
    } else {
      ++it;
    }
  }
  return dropped;
}

} // namespace flash_gpgpu_sim
