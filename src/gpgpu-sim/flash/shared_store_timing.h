#pragma once

#include "issue_interval_timing.h"

namespace flash_gpgpu_sim {

// Track the minimum issue-to-issue recurrence of one class of ordinary shared
// memory instruction from one warp. Other instruction classes from that warp
// remain issuable while this interval is active; the shared-memory dispatch
// pipeline independently models aggregate cross-warp service bandwidth.
using shared_load_issue_timing_t = issue_interval_timing_t;
using shared_store_issue_timing_t = issue_interval_timing_t;

} // namespace flash_gpgpu_sim
