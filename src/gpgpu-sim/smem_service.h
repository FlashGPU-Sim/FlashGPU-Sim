#ifndef GPGPU_SIM_SMEM_SERVICE_H
#define GPGPU_SIM_SMEM_SERVICE_H

#include <algorithm>
#include <cstring>

// Per-SM shared-memory byte grant. Clients expose demand, then cycle()
// assigns up to the byte budget with rotating priority. take() consumes
// this cycle's grant. budget 0 = unlimited bytes (still one-cycle delayed).
class shared_memory_service_t {
 public:
  enum client { LSU = 0, TMA = 1, DSM = 2, NCLIENT = 3 };

  explicit shared_memory_service_t(unsigned bytes_per_cycle = 0)
      : m_budget(bytes_per_cycle) {}

  void set_budget(unsigned bytes_per_cycle) { m_budget = bytes_per_cycle; }
  unsigned budget() const { return m_budget; }

  void expose(unsigned client, unsigned bytes) {
    if (client < NCLIENT) m_exposed[client] += bytes;
  }

  unsigned take(unsigned client, unsigned bytes) {
    if (client >= NCLIENT || !bytes) return 0;
    unsigned g = std::min(bytes, m_granted[client]);
    m_granted[client] -= g;
    return g;
  }

  void cycle() {
    unsigned remaining = m_budget ? m_budget : ~0u;
    unsigned granted[NCLIENT] = {};
    for (unsigned k = 0; k < NCLIENT && remaining; k++) {
      unsigned c = (m_rr + k) % NCLIENT;
      unsigned g = std::min(m_exposed[c], remaining);
      granted[c] = g;
      remaining -= g;
    }
    m_rr = (m_rr + 1) % NCLIENT;
    std::memcpy(m_granted, granted, sizeof m_granted);
    std::memset(m_exposed, 0, sizeof m_exposed);
  }

  unsigned granted(unsigned client) const {
    return client < NCLIENT ? m_granted[client] : 0;
  }
  unsigned exposed(unsigned client) const {
    return client < NCLIENT ? m_exposed[client] : 0;
  }

 private:
  unsigned m_budget = 0;
  unsigned m_rr = 0;
  unsigned m_exposed[NCLIENT] = {};
  unsigned m_granted[NCLIENT] = {};
};

#endif
