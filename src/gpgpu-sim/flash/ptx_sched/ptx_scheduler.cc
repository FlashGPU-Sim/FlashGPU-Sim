#include "ptx_scheduler.h"

#include "../../../../libcuda/gpgpu_context.h"
#include "../../../cuda-sim/opcodes.h"
#include "../../../cuda-sim/ptx_ir.h"

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <dirent.h>
#include <fstream>
#include <functional>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <vector>

namespace {

const char *k_sass_ptxline_guide_file = "sass_ptxline_guide.ptxline.sass";

enum class inst_class_t {
  tensor,
  ldmatrix,
  sfu,
  fp32,
  shfl,
  mem,
  intp,
  control,
  other,
  boundary
};

enum class pipe_t {
  tensor,
  sfu,
  ldst,
  fp32,
  xbar,
  intp,
  control,
  other,
  count
};

typedef std::set<const symbol *> reg_set_t;
typedef std::set<std::string> string_reg_set_t;

struct sched_inst_t {
  ptx_instruction *inst;
  unsigned original_index;
  inst_class_t cls;
  reg_set_t defs;
  reg_set_t uses;
};

struct dep_edge_t {
  unsigned src;
  unsigned dst;
  unsigned latency;
  bool raw;
};

struct dep_graph_t {
  std::vector<std::set<unsigned>> succ;
  std::vector<unsigned> indeg;
  std::vector<dep_edge_t> edges;
};

struct role_signature_t {
  char token;
  unsigned rank;
  int stage;
  int chain;
  unsigned pred_ld_count;
  std::vector<int> pred_ld_fanout;
  std::vector<int> pred_ld_stage;
  unsigned pred_tensor_count;
  unsigned succ_tensor_count;
  std::vector<int> succ_tensor_stage;
  std::vector<int> succ_tensor_chain;

  role_signature_t()
      : token('?'), rank(0), stage(-1), chain(-1), pred_ld_count(0),
        pred_tensor_count(0), succ_tensor_count(0) {}
};

struct guide_item_t {
  char token;
  role_signature_t sig;

  guide_item_t() : token('?') {}
};

struct sass_sched_inst_t {
  inst_class_t cls;
  char token;
  unsigned original_index;
  string_reg_set_t defs;
  string_reg_set_t uses;

  sass_sched_inst_t() : cls(inst_class_t::other), token(0), original_index(0) {}
};

struct reorder_stats_t {
  unsigned segments;
  unsigned skipped_segments;
  unsigned total_insts;
  unsigned moved_slots;
  unsigned max_segment;
  unsigned edges;
  unsigned sass_guided_segments;
  unsigned sass_guided_fallback_segments;
  unsigned sass_guide_cursor;
  std::string sass_guide_source;
  std::string sass_guide_head;

  reorder_stats_t()
      : segments(0), skipped_segments(0), total_insts(0), moved_slots(0),
        max_segment(0), edges(0), sass_guided_segments(0),
        sass_guided_fallback_segments(0), sass_guide_cursor(0) {}
};

struct sass_function_guide_t {
  std::string name;
  std::string lt_stream;
  std::vector<guide_item_t> guide_items;
  unsigned tensor_count;
  unsigned ldmatrix_count;

  sass_function_guide_t() : tensor_count(0), ldmatrix_count(0) {}
};

struct sass_file_guides_t {
  bool parsed;
  bool ok;
  std::vector<sass_function_guide_t> functions;

  sass_file_guides_t() : parsed(false), ok(false) {}
};

struct sass_primary_rules_t {
  bool parsed;
  bool ok;
  std::string arch;
  std::string gpu;
  std::string toolchain;
  std::string error;
  std::map<std::string, std::set<std::string>> primary_opcode;
  std::map<std::string, std::set<std::string>> attached_opcode_allowlist;
  std::map<std::string, std::string> policy;

  sass_primary_rules_t() : parsed(false), ok(false) {}
};

struct sass_ptxline_inst_t {
  unsigned sass_index;
  unsigned sass_offset;
  unsigned ptx_line;
  std::string opcode;
  inst_class_t cls;
  char token;

  sass_ptxline_inst_t()
      : sass_index(0), sass_offset(0), ptx_line(0), cls(inst_class_t::other),
        token(0) {}
};

struct sass_ptxline_function_t {
  std::string name;
  std::vector<sass_ptxline_inst_t> insts;
};

struct sass_ptxline_file_t {
  bool parsed;
  bool ok;
  std::string error;
  std::vector<sass_ptxline_function_t> functions;

  sass_ptxline_file_t() : parsed(false), ok(false) {}
};

struct ptxline_inst_ref_t {
  const ptx_instruction *inst;
  unsigned original_index;
  std::string ptx_opcode;
  inst_class_t cls;

  ptxline_inst_ref_t()
      : inst(NULL), original_index(0), cls(inst_class_t::other) {}
};

struct ptxline_guide_item_t {
  unsigned ptx_line;
  unsigned original_index;
  std::string ptx_opcode;
  std::string sass_opcode;
  unsigned sass_offset;
  char token;

  ptxline_guide_item_t()
      : ptx_line(0), original_index(0), sass_offset(0), token('?') {}
};

struct ptxline_guide_t {
  std::string source;
  std::vector<ptxline_guide_item_t> items;
  unsigned matched_ptx_lines;

  ptxline_guide_t() : matched_ptx_lines(0) {}
};

inst_class_t classify_inst(const ptx_instruction *inst);
char class_token(inst_class_t cls);

bool mkdir_if_needed(const std::string &path) {
  if (path.empty())
    return false;

  struct stat st;
  if (stat(path.c_str(), &st) == 0)
    return S_ISDIR(st.st_mode);

  if (mkdir(path.c_str(), 0755) == 0)
    return true;

  if (errno == EEXIST && stat(path.c_str(), &st) == 0)
    return S_ISDIR(st.st_mode);

  return false;
}

bool ensure_directory(const std::string &path) {
  if (path.empty())
    return false;

  std::string partial;
  std::size_t pos = 0;
  if (path[0] == '/') {
    partial = "/";
    pos = 1;
  }

  while (pos < path.size()) {
    std::size_t next = path.find('/', pos);
    std::string part = path.substr(pos, next - pos);
    if (!part.empty()) {
      if (!partial.empty() && partial[partial.size() - 1] != '/')
        partial += "/";
      partial += part;
      if (!mkdir_if_needed(partial))
        return false;
    }
    if (next == std::string::npos)
      break;
    pos = next + 1;
  }

  return true;
}

std::string sanitize_filename_prefix(const std::string &name) {
  std::string out;
  out.reserve(std::min<std::size_t>(name.size(), 96));
  for (std::size_t i = 0; i < name.size() && out.size() < 96; ++i) {
    const unsigned char c = static_cast<unsigned char>(name[i]);
    if (std::isalnum(c) || c == '_' || c == '-' || c == '.')
      out += static_cast<char>(c);
    else
      out += '_';
  }
  if (out.empty())
    out = "function";
  return out;
}

const char *class_name(inst_class_t cls) {
  switch (cls) {
  case inst_class_t::tensor:
    return "tensor";
  case inst_class_t::ldmatrix:
    return "ldmatrix";
  case inst_class_t::sfu:
    return "sfu";
  case inst_class_t::fp32:
    return "fp32";
  case inst_class_t::shfl:
    return "shfl";
  case inst_class_t::mem:
    return "mem";
  case inst_class_t::intp:
    return "int";
  case inst_class_t::control:
    return "control";
  case inst_class_t::other:
    return "other";
  case inst_class_t::boundary:
    return "boundary";
  }
  return "unknown";
}

std::string trim_copy(const std::string &in) {
  std::size_t begin = 0;
  while (begin < in.size() &&
         std::isspace(static_cast<unsigned char>(in[begin])))
    ++begin;
  std::size_t end = in.size();
  while (end > begin && std::isspace(static_cast<unsigned char>(in[end - 1])))
    --end;
  return in.substr(begin, end - begin);
}

bool starts_with(const std::string &text, const char *prefix) {
  const std::size_t n = strlen(prefix);
  return text.size() >= n && text.compare(0, n, prefix) == 0;
}

std::string uppercase_copy(const std::string &in) {
  std::string out = in;
  for (std::size_t i = 0; i < out.size(); ++i)
    out[i] =
        static_cast<char>(std::toupper(static_cast<unsigned char>(out[i])));
  return out;
}

std::string lowercase_copy(const std::string &in) {
  std::string out = in;
  for (std::size_t i = 0; i < out.size(); ++i)
    out[i] =
        static_cast<char>(std::tolower(static_cast<unsigned char>(out[i])));
  return out;
}

void ptx_reorder_fatal(const char *fmt, ...) {
  fprintf(stderr, "GPGPU-Sim PTX: SASS-guided reorder fatal: ");
  va_list args;
  va_start(args, fmt);
  vfprintf(stderr, fmt, args);
  va_end(args);
  fprintf(stderr, "\n");
  std::abort();
}

bool string_ends_with(const std::string &text, const std::string &suffix) {
  return text.size() >= suffix.size() &&
         text.compare(text.size() - suffix.size(), suffix.size(), suffix) == 0;
}

bool regular_file_exists(const char *path) {
  struct stat st;
  return path != NULL && stat(path, &st) == 0 && S_ISREG(st.st_mode);
}

std::string join_strings(const std::vector<std::string> &values,
                         const char *separator) {
  std::ostringstream out;
  for (unsigned i = 0; i < values.size(); ++i) {
    if (i != 0)
      out << separator;
    out << values[i];
  }
  return out.str();
}

std::string find_unique_sass_primary_rules_file() {
  DIR *dir = opendir(".");
  if (dir == NULL) {
    ptx_reorder_fatal("failed to open current config directory for .rules "
                      "discovery: %s",
                      strerror(errno));
  }

  std::vector<std::string> matches;
  while (struct dirent *entry = readdir(dir)) {
    const std::string name = entry->d_name;
    if (name == "." || name == ".." || !string_ends_with(name, ".rules"))
      continue;

    struct stat st;
    if (stat(name.c_str(), &st) != 0 || !S_ISREG(st.st_mode))
      continue;
    matches.push_back(name);
  }
  closedir(dir);

  std::sort(matches.begin(), matches.end());
  if (matches.empty()) {
    ptx_reorder_fatal(
        "SASS PTX-line guide is enabled but no .rules file exists in the "
        "current config/run directory");
  }
  if (matches.size() != 1) {
    ptx_reorder_fatal(
        "SASS PTX-line guide requires exactly one .rules file in the current "
        "config/run directory, found: %s",
        join_strings(matches, ", ").c_str());
  }
  return matches[0];
}

std::string strip_rule_comment(const std::string &line) {
  const std::size_t comment = line.find('#');
  if (comment == std::string::npos)
    return trim_copy(line);
  return trim_copy(line.substr(0, comment));
}

std::vector<std::string> split_csv_values(const std::string &text,
                                          bool uppercase) {
  std::vector<std::string> values;
  std::size_t begin = 0;
  while (begin <= text.size()) {
    std::size_t comma = text.find(',', begin);
    const std::string raw = comma == std::string::npos
                                ? text.substr(begin)
                                : text.substr(begin, comma - begin);
    std::string value = trim_copy(raw);
    if (!value.empty())
      values.push_back(uppercase ? uppercase_copy(value)
                                 : lowercase_copy(value));
    if (comma == std::string::npos)
      break;
    begin = comma + 1;
  }
  return values;
}

std::string normalize_ptx_opcode_key(const std::string &opcode) {
  return lowercase_copy(trim_copy(opcode));
}

std::string normalize_sass_opcode_key(const std::string &opcode) {
  return uppercase_copy(trim_copy(opcode));
}

bool opcode_set_contains(const std::set<std::string> &values,
                         const std::string &opcode) {
  return values.find(normalize_sass_opcode_key(opcode)) != values.end();
}

std::string ptx_mnemonic_from_source(const ptx_instruction *inst) {
  if (inst == NULL)
    return std::string();

  std::string source = trim_copy(inst->get_source());
  std::size_t pos = 0;
  while (pos < source.size() &&
         (std::isspace(static_cast<unsigned char>(source[pos])) ||
          source[pos] == '{'))
    ++pos;

  if (pos < source.size() && source[pos] == '@') {
    while (pos < source.size() &&
           !std::isspace(static_cast<unsigned char>(source[pos])))
      ++pos;
    while (pos < source.size() &&
           (std::isspace(static_cast<unsigned char>(source[pos])) ||
            source[pos] == '{'))
      ++pos;
  }

  const std::size_t begin = pos;
  while (pos < source.size()) {
    const unsigned char c = static_cast<unsigned char>(source[pos]);
    if (std::isspace(c) || source[pos] == ';' || source[pos] == '}')
      break;
    ++pos;
  }

  if (pos == begin)
    return normalize_ptx_opcode_key(inst->get_opcode_cstr());
  return normalize_ptx_opcode_key(source.substr(begin, pos - begin));
}

char sass_opcode_token(const std::string &opcode) {
  const std::string op = uppercase_copy(opcode);
  if (starts_with(op, "LDSM"))
    return 'L';
  if (starts_with(op, "HMMA") || starts_with(op, "IMMA") ||
      starts_with(op, "DMMA") || starts_with(op, "WGMMA") ||
      starts_with(op, "MMA"))
    return 'T';
  return 0;
}

inst_class_t classify_sass_opcode(const std::string &opcode) {
  const std::string op = uppercase_copy(opcode);
  if (starts_with(op, "LDSM"))
    return inst_class_t::ldmatrix;
  if (starts_with(op, "HMMA") || starts_with(op, "IMMA") ||
      starts_with(op, "DMMA") || starts_with(op, "WGMMA") ||
      starts_with(op, "MMA"))
    return inst_class_t::tensor;
  if (starts_with(op, "MUFU") || starts_with(op, "RRO"))
    return inst_class_t::sfu;
  if (starts_with(op, "FFMA") || starts_with(op, "FMUL") ||
      starts_with(op, "FADD") || starts_with(op, "FMNMX") ||
      starts_with(op, "FSET") || starts_with(op, "FSETP") ||
      starts_with(op, "F2F"))
    return inst_class_t::fp32;
  if (starts_with(op, "SHFL"))
    return inst_class_t::shfl;
  if (starts_with(op, "LDG") || starts_with(op, "LDC") ||
      starts_with(op, "ULDC") || starts_with(op, "LDS") ||
      starts_with(op, "STS") || starts_with(op, "STG") ||
      starts_with(op, "LDGSTS"))
    return inst_class_t::mem;
  if (starts_with(op, "BRA") || starts_with(op, "EXIT") ||
      starts_with(op, "RET") || starts_with(op, "BAR") ||
      starts_with(op, "DEPBAR") || starts_with(op, "LDGDEPBAR") ||
      starts_with(op, "NOP") || starts_with(op, "WARPSYNC") ||
      starts_with(op, "BSYNC"))
    return inst_class_t::control;
  if (starts_with(op, "I") || starts_with(op, "UI") ||
      starts_with(op, "UIMAD") || starts_with(op, "IMAD") ||
      starts_with(op, "LOP") || starts_with(op, "PLOP") ||
      starts_with(op, "MOV") || starts_with(op, "S2R") ||
      starts_with(op, "S2UR") || starts_with(op, "CS2R") ||
      starts_with(op, "LEA") || starts_with(op, "SHF") ||
      starts_with(op, "PRMT") || starts_with(op, "SEL") ||
      starts_with(op, "P2R") || starts_with(op, "R2P") ||
      starts_with(op, "VOTE"))
    return inst_class_t::intp;
  return inst_class_t::other;
}

std::vector<std::string> split_top_operands(const std::string &operands) {
  std::vector<std::string> out;
  std::size_t begin = 0;
  int depth = 0;
  for (std::size_t i = 0; i < operands.size(); ++i) {
    const char c = operands[i];
    if (c == '{' || c == '[' || c == '(') {
      ++depth;
    } else if ((c == '}' || c == ']' || c == ')') && depth > 0) {
      --depth;
    } else if (c == ',' && depth == 0) {
      out.push_back(trim_copy(operands.substr(begin, i - begin)));
      begin = i + 1;
    }
  }
  const std::string tail = trim_copy(operands.substr(begin));
  if (!tail.empty())
    out.push_back(tail);
  return out;
}

void add_sass_reg(string_reg_set_t &regs, const std::string &reg,
                  unsigned width) {
  const std::string upper = uppercase_copy(reg);
  if (upper == "RZ")
    return;
  if (upper.size() < 2 || upper[0] != 'R')
    return;
  char *end = NULL;
  const unsigned long base = strtoul(upper.c_str() + 1, &end, 10);
  if (end == NULL || *end != '\0') {
    regs.insert(upper);
    return;
  }
  const unsigned n = std::max(1u, width);
  for (unsigned i = 0; i < n; ++i) {
    char buf[32];
    snprintf(buf, sizeof(buf), "R%lu", base + i);
    regs.insert(buf);
  }
}

string_reg_set_t sass_regs_in(const std::string &text) {
  string_reg_set_t regs;
  for (std::size_t i = 0; i < text.size(); ++i) {
    if (std::toupper(static_cast<unsigned char>(text[i])) != 'R')
      continue;
    const bool left_ok =
        i == 0 || !std::isalnum(static_cast<unsigned char>(text[i - 1]));
    if (!left_ok)
      continue;
    std::size_t j = i + 1;
    if (j < text.size() &&
        std::toupper(static_cast<unsigned char>(text[j])) == 'Z') {
      ++j;
      const bool right_ok = j == text.size() ||
                            !std::isalnum(static_cast<unsigned char>(text[j]));
      if (right_ok)
        continue;
    }
    if (j >= text.size() || !std::isdigit(static_cast<unsigned char>(text[j])))
      continue;
    while (j < text.size() && std::isdigit(static_cast<unsigned char>(text[j])))
      ++j;
    const bool right_ok =
        j == text.size() || !std::isalnum(static_cast<unsigned char>(text[j]));
    if (!right_ok)
      continue;
    regs.insert(uppercase_copy(text.substr(i, j - i)));
    i = j;
  }
  return regs;
}

unsigned sass_ldmatrix_width(const std::string &opcode) {
  std::string op = uppercase_copy(opcode);
  std::size_t end = op.size();
  while (end > 0) {
    std::size_t begin = op.rfind('.', end - 1);
    begin = begin == std::string::npos ? 0 : begin + 1;
    const std::string piece = op.substr(begin, end - begin);
    bool all_digits = !piece.empty();
    for (std::size_t i = 0; i < piece.size(); ++i) {
      if (!std::isdigit(static_cast<unsigned char>(piece[i]))) {
        all_digits = false;
        break;
      }
    }
    if (all_digits)
      return static_cast<unsigned>(strtoul(piece.c_str(), NULL, 10));
    if (begin == 0)
      break;
    end = begin - 1;
  }
  return 1;
}

unsigned sass_tensor_dest_width(const std::string &opcode) {
  const std::string op = uppercase_copy(opcode);
  if (starts_with(op, "HMMA") || starts_with(op, "IMMA") ||
      starts_with(op, "DMMA") || starts_with(op, "MMA"))
    return 4;
  return 1;
}

void collect_sass_defs_uses(const std::string &opcode,
                            const std::string &operands_text,
                            string_reg_set_t &defs, string_reg_set_t &uses) {
  const std::vector<std::string> operands = split_top_operands(operands_text);
  if (operands.empty())
    return;

  const inst_class_t cls = classify_sass_opcode(opcode);
  if (cls != inst_class_t::ldmatrix && cls != inst_class_t::tensor)
    return;

  string_reg_set_t dst_regs = sass_regs_in(operands[0]);
  if (dst_regs.size() == 1) {
    add_sass_reg(defs, *dst_regs.begin(),
                 cls == inst_class_t::ldmatrix
                     ? sass_ldmatrix_width(opcode)
                     : sass_tensor_dest_width(opcode));
  } else {
    defs.insert(dst_regs.begin(), dst_regs.end());
  }

  for (unsigned i = 1; i < operands.size(); ++i) {
    const string_reg_set_t regs = sass_regs_in(operands[i]);
    uses.insert(regs.begin(), regs.end());
  }
}

bool parse_sass_instruction_line(const std::string &line, std::string *opcode,
                                 std::string *operands) {
  std::size_t pos = line.find("*/");
  if (pos == std::string::npos)
    return false;
  pos += 2;
  while (pos < line.size() &&
         std::isspace(static_cast<unsigned char>(line[pos])))
    ++pos;
  if (pos < line.size() && line[pos] == '@') {
    while (pos < line.size() &&
           !std::isspace(static_cast<unsigned char>(line[pos])))
      ++pos;
    while (pos < line.size() &&
           std::isspace(static_cast<unsigned char>(line[pos])))
      ++pos;
  }
  const std::size_t opcode_begin = pos;
  while (pos < line.size()) {
    const unsigned char c = static_cast<unsigned char>(line[pos]);
    if (std::isspace(c) || line[pos] == ';')
      break;
    ++pos;
  }
  if (pos == opcode_begin)
    return false;
  *opcode = line.substr(opcode_begin, pos - opcode_begin);
  while (pos < line.size() &&
         std::isspace(static_cast<unsigned char>(line[pos])))
    ++pos;
  const std::size_t operands_begin = pos;
  while (pos < line.size() && line[pos] != ';')
    ++pos;
  *operands = trim_copy(line.substr(operands_begin, pos - operands_begin));
  return true;
}

bool parse_sass_offset(const std::string &line, unsigned *offset) {
  const std::size_t begin = line.find("/*");
  if (begin == std::string::npos)
    return false;
  const std::size_t end = line.find("*/", begin + 2);
  if (end == std::string::npos || end <= begin + 2)
    return false;
  const std::string text = line.substr(begin + 2, end - begin - 2);
  char *parse_end = NULL;
  const unsigned long value = strtoul(text.c_str(), &parse_end, 16);
  if (parse_end == NULL || *parse_end != '\0')
    return false;
  *offset = static_cast<unsigned>(value);
  return true;
}

bool parse_sass_ptxline_marker(const std::string &line, unsigned *ptx_line) {
  const std::size_t marker = line.find("//##");
  if (marker == std::string::npos)
    return false;
  std::size_t line_pos = line.find("line ", marker);
  if (line_pos == std::string::npos)
    return false;
  line_pos += 5;
  while (line_pos < line.size() &&
         std::isspace(static_cast<unsigned char>(line[line_pos])))
    ++line_pos;
  if (line_pos == line.size() ||
      !std::isdigit(static_cast<unsigned char>(line[line_pos])))
    return false;
  char *parse_end = NULL;
  const unsigned long value = strtoul(line.c_str() + line_pos, &parse_end, 10);
  if (parse_end == NULL || parse_end == line.c_str() + line_pos)
    return false;
  *ptx_line = static_cast<unsigned>(value);
  return true;
}

bool parse_sass_function_name(const std::string &line, std::string *name) {
  const std::size_t text = line.find(".text.");
  if (text == std::string::npos)
    return false;
  std::size_t begin = text + 6;
  std::size_t end = begin;
  while (end < line.size()) {
    const unsigned char c = static_cast<unsigned char>(line[end]);
    if (std::isspace(c) || line[end] == ',' || line[end] == '"' ||
        line[end] == ':' || line[end] == '(')
      break;
    ++end;
  }
  if (end == begin)
    return false;
  *name = line.substr(begin, end - begin);
  return true;
}

sass_primary_rules_t parse_sass_primary_rules_from_file(const char *path) {
  sass_primary_rules_t rules;
  rules.parsed = true;
  if (path == NULL || path[0] == '\0') {
    rules.error = "empty primary hint file path";
    return rules;
  }

  std::ifstream in(path);
  if (!in.good()) {
    rules.error =
        std::string("failed to open primary hint file '") + path + "'";
    return rules;
  }

  std::string section;
  std::string line;
  unsigned lineno = 0;
  while (std::getline(in, line)) {
    ++lineno;
    line = strip_rule_comment(line);
    if (line.empty())
      continue;
    if (line[0] == '[' && line[line.size() - 1] == ']') {
      section = trim_copy(line.substr(1, line.size() - 2));
      continue;
    }

    const std::size_t equal = line.find('=');
    if (equal == std::string::npos) {
      std::ostringstream out;
      out << "invalid rule line " << lineno << ": " << line;
      rules.error = out.str();
      return rules;
    }

    const std::string key = trim_copy(line.substr(0, equal));
    const std::string value = trim_copy(line.substr(equal + 1));
    if (section == "metadata") {
      if (key == "arch")
        rules.arch = value;
      else if (key == "gpu")
        rules.gpu = value;
      else if (key == "toolchain")
        rules.toolchain = value;
    } else if (section == "primary_opcode") {
      const std::string ptx_opcode = normalize_ptx_opcode_key(key);
      const std::vector<std::string> values = split_csv_values(value, true);
      rules.primary_opcode[ptx_opcode].insert(values.begin(), values.end());
    } else if (section == "attached_opcode_allowlist") {
      const std::string ptx_opcode = normalize_ptx_opcode_key(key);
      const std::vector<std::string> values = split_csv_values(value, true);
      rules.attached_opcode_allowlist[ptx_opcode].insert(values.begin(),
                                                         values.end());
    } else if (section == "policy") {
      rules.policy[trim_copy(key)] = trim_copy(value);
    }
  }

  if (rules.primary_opcode.empty()) {
    rules.error = "primary hint file has no [primary_opcode] entries";
    return rules;
  }
  for (std::map<std::string, std::set<std::string>>::const_iterator it =
           rules.primary_opcode.begin();
       it != rules.primary_opcode.end(); ++it) {
    if (it->second.empty()) {
      rules.error =
          std::string("empty primary SASS opcode list for '") + it->first + "'";
      return rules;
    }
  }
  for (std::map<std::string, std::set<std::string>>::const_iterator it =
           rules.attached_opcode_allowlist.begin();
       it != rules.attached_opcode_allowlist.end(); ++it) {
    if (rules.primary_opcode.find(it->first) == rules.primary_opcode.end()) {
      rules.error =
          std::string("attached allowlist without primary rule for '") +
          it->first + "'";
      return rules;
    }
  }
  std::map<std::string, std::string>::const_iterator fallback =
      rules.policy.find("fallback");
  if (fallback != rules.policy.end() && fallback->second != "disabled") {
    rules.error = "SASS PTX-line guide requires policy fallback = disabled";
    return rules;
  }

  rules.ok = true;
  return rules;
}

const sass_primary_rules_t &get_sass_primary_rules(const char *path) {
  static std::map<std::string, sass_primary_rules_t> cache;
  const std::string key = path == NULL ? std::string() : std::string(path);
  std::map<std::string, sass_primary_rules_t>::iterator found = cache.find(key);
  if (found != cache.end())
    return found->second;

  sass_primary_rules_t parsed = parse_sass_primary_rules_from_file(path);
  std::pair<std::map<std::string, sass_primary_rules_t>::iterator, bool>
      inserted = cache.insert(std::make_pair(key, parsed));
  return inserted.first->second;
}

sass_ptxline_file_t parse_sass_ptxline_file(const char *path) {
  sass_ptxline_file_t result;
  result.parsed = true;
  if (path == NULL || path[0] == '\0') {
    result.error = "empty SASS PTX-line file path";
    return result;
  }

  std::ifstream in(path);
  if (!in.good()) {
    result.error =
        std::string("failed to open SASS PTX-line file '") + path + "'";
    return result;
  }

  sass_ptxline_function_t current;
  bool have_function = false;
  unsigned current_ptx_line = 0;
  std::string line;
  while (std::getline(in, line)) {
    std::string function_name;
    if (parse_sass_function_name(line, &function_name)) {
      if (!have_function || function_name != current.name) {
        if (have_function && !current.insts.empty())
          result.functions.push_back(current);
        current = sass_ptxline_function_t();
        current.name = function_name;
        have_function = true;
        current_ptx_line = 0;
      }
      continue;
    }

    if (!have_function)
      continue;

    unsigned marked_line = 0;
    if (parse_sass_ptxline_marker(line, &marked_line)) {
      current_ptx_line = marked_line;
      continue;
    }

    std::string opcode;
    std::string operands;
    if (!parse_sass_instruction_line(line, &opcode, &operands))
      continue;

    unsigned sass_offset = 0;
    if (!parse_sass_offset(line, &sass_offset))
      continue;
    if (current_ptx_line == 0)
      continue;

    sass_ptxline_inst_t inst;
    inst.sass_index = current.insts.size();
    inst.sass_offset = sass_offset;
    inst.ptx_line = current_ptx_line;
    inst.opcode = normalize_sass_opcode_key(opcode);
    inst.cls = classify_sass_opcode(inst.opcode);
    inst.token = class_token(inst.cls);
    current.insts.push_back(inst);
  }

  if (have_function && !current.insts.empty())
    result.functions.push_back(current);

  if (result.functions.empty()) {
    result.error = "SASS PTX-line file has no parsed function records";
    return result;
  }
  result.ok = true;
  return result;
}

const sass_ptxline_file_t &get_sass_ptxline_file(const char *path) {
  static std::map<std::string, sass_ptxline_file_t> cache;
  const std::string key = path == NULL ? std::string() : std::string(path);
  std::map<std::string, sass_ptxline_file_t>::iterator found = cache.find(key);
  if (found != cache.end())
    return found->second;

  sass_ptxline_file_t parsed = parse_sass_ptxline_file(path);
  std::pair<std::map<std::string, sass_ptxline_file_t>::iterator, bool>
      inserted = cache.insert(std::make_pair(key, parsed));
  return inserted.first->second;
}

void sort_int_vector(std::vector<int> &values) {
  std::sort(values.begin(), values.end());
}

std::vector<role_signature_t>
compute_sass_role_signatures(const std::vector<sass_sched_inst_t> &insts) {
  std::vector<role_signature_t> out(insts.size());
  std::map<char, unsigned> class_seen;
  for (unsigned i = 0; i < insts.size(); ++i) {
    const char token = class_token(insts[i].cls);
    out[i].token = token;
    out[i].rank = class_seen[token]++;
  }

  std::vector<std::set<unsigned>> ldmatrix_tensor_succ(insts.size());
  std::vector<std::set<unsigned>> tensor_ldmatrix_pred(insts.size());
  std::vector<std::set<unsigned>> tensor_tensor_pred(insts.size());
  std::map<std::string, unsigned> last_def;
  std::set<std::pair<unsigned, unsigned>> raw_edges;

  for (unsigned i = 0; i < insts.size(); ++i) {
    const sass_sched_inst_t &inst = insts[i];
    for (string_reg_set_t::const_iterator use = inst.uses.begin();
         use != inst.uses.end(); ++use) {
      std::map<std::string, unsigned>::const_iterator def = last_def.find(*use);
      if (def == last_def.end())
        continue;
      const unsigned src = def->second;
      const std::pair<unsigned, unsigned> edge(src, i);
      if (!raw_edges.insert(edge).second)
        continue;
      if (insts[src].cls == inst_class_t::ldmatrix &&
          inst.cls == inst_class_t::tensor) {
        ldmatrix_tensor_succ[src].insert(i);
        tensor_ldmatrix_pred[i].insert(src);
      } else if (insts[src].cls == inst_class_t::tensor &&
                 inst.cls == inst_class_t::tensor) {
        tensor_tensor_pred[i].insert(src);
      }
    }
    for (string_reg_set_t::const_iterator def = inst.defs.begin();
         def != inst.defs.end(); ++def)
      last_def[*def] = i;
  }

  std::map<unsigned, int> tensor_stage;
  std::map<unsigned, int> tensor_chain;
  int next_chain = 0;
  for (unsigned i = 0; i < insts.size(); ++i) {
    if (insts[i].cls != inst_class_t::tensor)
      continue;
    int stage = 0;
    bool have_stage = false;
    int chain = -1;
    for (std::set<unsigned>::const_iterator pred =
             tensor_tensor_pred[i].begin();
         pred != tensor_tensor_pred[i].end(); ++pred) {
      std::map<unsigned, int>::const_iterator pred_stage =
          tensor_stage.find(*pred);
      if (pred_stage != tensor_stage.end()) {
        stage = std::max(stage, pred_stage->second + 1);
        have_stage = true;
      }
      std::map<unsigned, int>::const_iterator pred_chain =
          tensor_chain.find(*pred);
      if (pred_chain != tensor_chain.end())
        chain = chain < 0 ? pred_chain->second
                          : std::min(chain, pred_chain->second);
    }
    tensor_stage[i] = have_stage ? stage : 0;
    if (chain < 0)
      chain = next_chain++;
    tensor_chain[i] = chain;
  }

  std::map<unsigned, int> ldmatrix_stage;
  for (unsigned i = 0; i < insts.size(); ++i) {
    if (ldmatrix_tensor_succ[i].empty())
      continue;
    int stage = -1;
    for (std::set<unsigned>::const_iterator succ =
             ldmatrix_tensor_succ[i].begin();
         succ != ldmatrix_tensor_succ[i].end(); ++succ) {
      std::map<unsigned, int>::const_iterator found = tensor_stage.find(*succ);
      if (found != tensor_stage.end())
        stage = stage < 0 ? found->second : std::min(stage, found->second);
    }
    if (stage >= 0)
      ldmatrix_stage[i] = stage;
  }

  for (unsigned i = 0; i < insts.size(); ++i) {
    role_signature_t &sig = out[i];
    std::map<unsigned, int>::const_iterator ts = tensor_stage.find(i);
    std::map<unsigned, int>::const_iterator ls = ldmatrix_stage.find(i);
    std::map<unsigned, int>::const_iterator tc = tensor_chain.find(i);
    sig.stage = ts != tensor_stage.end()
                    ? ts->second
                    : (ls != ldmatrix_stage.end() ? ls->second : -1);
    sig.chain = tc != tensor_chain.end() ? tc->second : -1;
    sig.pred_ld_count = tensor_ldmatrix_pred[i].size();
    sig.pred_tensor_count = tensor_tensor_pred[i].size();
    sig.succ_tensor_count = ldmatrix_tensor_succ[i].size();
    for (std::set<unsigned>::const_iterator pred =
             tensor_ldmatrix_pred[i].begin();
         pred != tensor_ldmatrix_pred[i].end(); ++pred) {
      sig.pred_ld_fanout.push_back(ldmatrix_tensor_succ[*pred].size());
      std::map<unsigned, int>::const_iterator pred_stage =
          ldmatrix_stage.find(*pred);
      sig.pred_ld_stage.push_back(
          pred_stage == ldmatrix_stage.end() ? -1 : pred_stage->second);
    }
    for (std::set<unsigned>::const_iterator succ =
             ldmatrix_tensor_succ[i].begin();
         succ != ldmatrix_tensor_succ[i].end(); ++succ) {
      std::map<unsigned, int>::const_iterator succ_stage =
          tensor_stage.find(*succ);
      std::map<unsigned, int>::const_iterator succ_chain =
          tensor_chain.find(*succ);
      sig.succ_tensor_stage.push_back(
          succ_stage == tensor_stage.end() ? -1 : succ_stage->second);
      sig.succ_tensor_chain.push_back(
          succ_chain == tensor_chain.end() ? -1 : succ_chain->second);
    }
    sort_int_vector(sig.pred_ld_fanout);
    sort_int_vector(sig.pred_ld_stage);
    sort_int_vector(sig.succ_tensor_stage);
    sort_int_vector(sig.succ_tensor_chain);
  }

  return out;
}

std::vector<guide_item_t>
build_sass_guide_items(const std::vector<sass_sched_inst_t> &insts) {
  const std::vector<role_signature_t> sigs =
      compute_sass_role_signatures(insts);
  std::vector<guide_item_t> items;
  for (unsigned i = 0; i < insts.size(); ++i) {
    const char token = class_token(insts[i].cls);
    if (token != 'T' && token != 'L')
      continue;
    guide_item_t item;
    item.token = token;
    item.sig = sigs[i];
    items.push_back(item);
  }
  return items;
}

sass_file_guides_t parse_sass_guides_from_file(const char *path) {
  sass_file_guides_t result;
  result.parsed = true;
  if (path == NULL || path[0] == '\0')
    return result;

  std::ifstream in(path);
  if (!in.good()) {
    fprintf(stderr,
            "GPGPU-Sim PTX: failed to open SASS guide file '%s'; "
            "SASS-guided PTX reorder will fall back\n",
            path);
    return result;
  }

  sass_function_guide_t current;
  std::vector<sass_sched_inst_t> current_insts;
  bool have_function = false;
  std::string line;
  while (std::getline(in, line)) {
    const std::size_t function_pos = line.find("Function");
    if (function_pos != std::string::npos) {
      if (have_function) {
        current.guide_items = build_sass_guide_items(current_insts);
        result.functions.push_back(current);
      }
      current = sass_function_guide_t();
      current_insts.clear();
      have_function = true;
      const std::size_t colon = line.find(':', function_pos);
      if (colon != std::string::npos)
        current.name = trim_copy(line.substr(colon + 1));
      else
        current.name = trim_copy(line.substr(function_pos + 8));
      continue;
    }

    if (!have_function)
      continue;

    std::string opcode;
    std::string operands;
    if (!parse_sass_instruction_line(line, &opcode, &operands))
      continue;
    sass_sched_inst_t inst;
    inst.cls = classify_sass_opcode(opcode);
    inst.token = sass_opcode_token(opcode);
    inst.original_index = current_insts.size();
    collect_sass_defs_uses(opcode, operands, inst.defs, inst.uses);
    current_insts.push_back(inst);

    const char token = sass_opcode_token(opcode);
    if (token == 0)
      continue;
    current.lt_stream += token;
    if (token == 'T')
      ++current.tensor_count;
    else if (token == 'L')
      ++current.ldmatrix_count;
  }

  if (have_function) {
    current.guide_items = build_sass_guide_items(current_insts);
    result.functions.push_back(current);
  }

  result.ok = !result.functions.empty();
  return result;
}

const sass_file_guides_t &get_sass_guides(const char *path) {
  static std::map<std::string, sass_file_guides_t> cache;
  const std::string key = path == NULL ? std::string() : std::string(path);
  std::map<std::string, sass_file_guides_t>::iterator found = cache.find(key);
  if (found != cache.end())
    return found->second;

  sass_file_guides_t parsed = parse_sass_guides_from_file(path);
  std::pair<std::map<std::string, sass_file_guides_t>::iterator, bool>
      inserted = cache.insert(std::make_pair(key, parsed));
  return inserted.first->second;
}

const sass_function_guide_t *select_sass_guide(const sass_file_guides_t &guides,
                                               const std::string &function_name,
                                               unsigned ptx_tensor_count,
                                               unsigned ptx_ldmatrix_count) {
  if (!guides.ok)
    return NULL;

  for (std::vector<sass_function_guide_t>::const_iterator it =
           guides.functions.begin();
       it != guides.functions.end(); ++it) {
    if (!it->lt_stream.empty() && it->name == function_name)
      return &(*it);
  }

  for (std::vector<sass_function_guide_t>::const_iterator it =
           guides.functions.begin();
       it != guides.functions.end(); ++it) {
    if (it->lt_stream.empty())
      continue;
    if (it->name.find(function_name) != std::string::npos ||
        function_name.find(it->name) != std::string::npos)
      return &(*it);
  }

  const sass_function_guide_t *best = NULL;
  unsigned best_score = std::numeric_limits<unsigned>::max();
  for (std::vector<sass_function_guide_t>::const_iterator it =
           guides.functions.begin();
       it != guides.functions.end(); ++it) {
    if (it->lt_stream.empty() || it->tensor_count == 0)
      continue;
    const unsigned tensor_delta = it->tensor_count > ptx_tensor_count
                                      ? it->tensor_count - ptx_tensor_count
                                      : ptx_tensor_count - it->tensor_count;
    const unsigned ldmatrix_delta =
        it->ldmatrix_count > ptx_ldmatrix_count
            ? it->ldmatrix_count - ptx_ldmatrix_count
            : ptx_ldmatrix_count - it->ldmatrix_count;
    const unsigned score = tensor_delta * 4 + ldmatrix_delta;
    if (best == NULL || score < best_score) {
      best = &(*it);
      best_score = score;
    }
  }

  return best;
}

const sass_ptxline_function_t *
select_sass_ptxline_function(const sass_ptxline_file_t &guides,
                             const std::string &function_name) {
  if (!guides.ok)
    return NULL;

  for (std::vector<sass_ptxline_function_t>::const_iterator it =
           guides.functions.begin();
       it != guides.functions.end(); ++it) {
    if (!it->insts.empty() && it->name == function_name)
      return &(*it);
  }

  for (std::vector<sass_ptxline_function_t>::const_iterator it =
           guides.functions.begin();
       it != guides.functions.end(); ++it) {
    if (it->insts.empty())
      continue;
    if (guides.functions.size() == 1 &&
        (it->name.find(function_name) != std::string::npos ||
         function_name.find(it->name) != std::string::npos))
      return &(*it);
  }

  return NULL;
}

std::map<unsigned, std::vector<ptxline_inst_ref_t>>
build_ptxline_inst_refs(const std::list<ptx_instruction *> &instructions) {
  std::map<unsigned, std::vector<ptxline_inst_ref_t>> refs;
  unsigned original_index = 0;
  for (std::list<ptx_instruction *>::const_iterator it = instructions.begin();
       it != instructions.end(); ++it, ++original_index) {
    const ptx_instruction *inst = *it;
    if (inst == NULL || inst->is_label())
      continue;

    ptxline_inst_ref_t ref;
    ref.inst = inst;
    ref.original_index = original_index;
    ref.ptx_opcode = ptx_mnemonic_from_source(inst);
    ref.cls = classify_inst(inst);
    refs[inst->source_line()].push_back(ref);
  }
  return refs;
}

struct ptxline_build_state_t {
  bool initialized;
  ptxline_inst_ref_t ref;
  std::vector<sass_ptxline_inst_t> primary_candidates;
  std::vector<sass_ptxline_inst_t> attached_insts;

  ptxline_build_state_t() : initialized(false) {}
};

std::string format_sass_offsets(const std::vector<sass_ptxline_inst_t> &insts) {
  std::ostringstream out;
  for (unsigned i = 0; i < insts.size(); ++i) {
    if (i != 0)
      out << ",";
    out << "0x" << std::hex << insts[i].sass_offset << std::dec << ":"
        << insts[i].opcode;
  }
  return out.str();
}

ptxline_guide_t
build_sass_ptxline_guide(const std::string &function_name,
                         const std::list<ptx_instruction *> &instructions,
                         const sass_ptxline_function_t &sass_func,
                         const sass_primary_rules_t &rules) {
  if (!rules.ok)
    ptx_reorder_fatal("invalid primary SASS hint rules for function '%s': %s",
                      function_name.c_str(), rules.error.c_str());

  const std::map<unsigned, std::vector<ptxline_inst_ref_t>> ptx_by_line =
      build_ptxline_inst_refs(instructions);
  std::map<unsigned, ptxline_build_state_t> states;

  for (std::vector<sass_ptxline_inst_t>::const_iterator sass =
           sass_func.insts.begin();
       sass != sass_func.insts.end(); ++sass) {
    std::map<unsigned, std::vector<ptxline_inst_ref_t>>::const_iterator
        ptx_line = ptx_by_line.find(sass->ptx_line);
    if (ptx_line == ptx_by_line.end())
      continue;

    std::vector<ptxline_inst_ref_t> ruled_ptx;
    for (std::vector<ptxline_inst_ref_t>::const_iterator ref =
             ptx_line->second.begin();
         ref != ptx_line->second.end(); ++ref) {
      if (rules.primary_opcode.find(ref->ptx_opcode) !=
          rules.primary_opcode.end())
        ruled_ptx.push_back(*ref);
    }
    if (ruled_ptx.empty())
      continue;
    if (ruled_ptx.size() != 1) {
      ptx_reorder_fatal(
          "function '%s' PTX line %u has %zu rule-covered PTX "
          "instructions; add a line-disambiguation rule before using it",
          function_name.c_str(), sass->ptx_line, ruled_ptx.size());
    }

    ptxline_build_state_t &state = states[sass->ptx_line];
    if (!state.initialized) {
      state.initialized = true;
      state.ref = ruled_ptx[0];
    } else if (state.ref.original_index != ruled_ptx[0].original_index) {
      ptx_reorder_fatal(
          "function '%s' PTX line %u maps to multiple PTX instructions "
          "(orig=%u and orig=%u)",
          function_name.c_str(), sass->ptx_line, state.ref.original_index,
          ruled_ptx[0].original_index);
    }

    const std::set<std::string> &primary =
        rules.primary_opcode.find(state.ref.ptx_opcode)->second;
    if (opcode_set_contains(primary, sass->opcode))
      state.primary_candidates.push_back(*sass);
    else
      state.attached_insts.push_back(*sass);
  }

  ptxline_guide_t guide;
  guide.source = sass_func.name;
  guide.matched_ptx_lines = states.size();

  for (std::map<unsigned, ptxline_build_state_t>::const_iterator state =
           states.begin();
       state != states.end(); ++state) {
    const unsigned ptx_line = state->first;
    const ptxline_build_state_t &entry = state->second;
    if (entry.primary_candidates.empty()) {
      ptx_reorder_fatal(
          "function '%s' PTX line %u opcode '%s' has no primary SASS "
          "candidate in guide '%s'",
          function_name.c_str(), ptx_line, entry.ref.ptx_opcode.c_str(),
          sass_func.name.c_str());
    }
    if (entry.primary_candidates.size() != 1) {
      ptx_reorder_fatal(
          "function '%s' PTX line %u opcode '%s' has multiple primary "
          "SASS candidates: %s",
          function_name.c_str(), ptx_line, entry.ref.ptx_opcode.c_str(),
          format_sass_offsets(entry.primary_candidates).c_str());
    }

    std::map<std::string, std::set<std::string>>::const_iterator allow =
        rules.attached_opcode_allowlist.find(entry.ref.ptx_opcode);
    for (std::vector<sass_ptxline_inst_t>::const_iterator attached =
             entry.attached_insts.begin();
         attached != entry.attached_insts.end(); ++attached) {
      if (allow == rules.attached_opcode_allowlist.end() ||
          !opcode_set_contains(allow->second, attached->opcode)) {
        ptx_reorder_fatal(
            "function '%s' PTX line %u opcode '%s' has unexpected "
            "attached SASS opcode '%s' at 0x%x",
            function_name.c_str(), ptx_line, entry.ref.ptx_opcode.c_str(),
            attached->opcode.c_str(), attached->sass_offset);
      }
    }

    const sass_ptxline_inst_t &primary = entry.primary_candidates[0];
    ptxline_guide_item_t item;
    item.ptx_line = ptx_line;
    item.original_index = entry.ref.original_index;
    item.ptx_opcode = entry.ref.ptx_opcode;
    item.sass_opcode = primary.opcode;
    item.sass_offset = primary.sass_offset;
    item.token = class_token(entry.ref.cls);
    guide.items.push_back(item);
  }

  std::sort(guide.items.begin(), guide.items.end(),
            [](const ptxline_guide_item_t &a, const ptxline_guide_item_t &b) {
              if (a.sass_offset != b.sass_offset)
                return a.sass_offset < b.sass_offset;
              return a.original_index < b.original_index;
            });

  if (guide.items.empty()) {
    ptx_reorder_fatal(
        "function '%s' matched SASS PTX-line guide '%s' but produced no "
        "primary guide items",
        function_name.c_str(), sass_func.name.c_str());
  }
  return guide;
}

std::string build_ptxline_guide_head(const ptxline_guide_t &guide,
                                     unsigned limit) {
  std::ostringstream out;
  for (unsigned i = 0; i < guide.items.size() && i < limit; ++i) {
    if (i != 0)
      out << " ";
    out << guide.items[i].token << "@" << guide.items[i].ptx_line << "/0x"
        << std::hex << guide.items[i].sass_offset << std::dec;
  }
  return out.str();
}

void dump_ptx_reorder_result(
    const std::string &function_name,
    const std::list<ptx_instruction *> &instructions,
    const std::map<const ptx_instruction *, unsigned> &original_indices,
    const reorder_stats_t &stats, int ready_slack, const char *dump_dir) {
  if (dump_dir == NULL || dump_dir[0] == '\0')
    return;

  const std::string dir(dump_dir);
  if (!ensure_directory(dir)) {
    fprintf(stderr,
            "GPGPU-Sim PTX: failed to create ptx reorder dump directory '%s': "
            "%s\n",
            dir.c_str(), strerror(errno));
    return;
  }

  const std::size_t hash = std::hash<std::string>()(function_name);
  char hash_buf[32];
  snprintf(hash_buf, sizeof(hash_buf), "%016zx", hash);

  std::string path = dir + "/ptx_sched_" + hash_buf + "_" +
                     sanitize_filename_prefix(function_name) + ".txt";
  FILE *fp = fopen(path.c_str(), "w");
  if (fp == NULL) {
    fprintf(stderr, "GPGPU-Sim PTX: failed to open ptx reorder dump '%s': %s\n",
            path.c_str(), strerror(errno));
    return;
  }

  fprintf(fp, "# ptx_sched dump\n");
  fprintf(fp, "# function: %s\n", function_name.c_str());
  fprintf(fp,
          "# stats: segments=%u skipped=%u insts=%u moved_slots=%u "
          "max_segment=%u edges=%u slack=%d sass_guided=%u "
          "sass_guided_fallback=%u sass_cursor=%u\n",
          stats.segments, stats.skipped_segments, stats.total_insts,
          stats.moved_slots, stats.max_segment, stats.edges, ready_slack,
          stats.sass_guided_segments, stats.sass_guided_fallback_segments,
          stats.sass_guide_cursor);
  if (!stats.sass_guide_source.empty() || !stats.sass_guide_head.empty()) {
    fprintf(fp, "# sass_guide: source=%s head=%s\n",
            stats.sass_guide_source.empty() ? "<none>"
                                            : stats.sass_guide_source.c_str(),
            stats.sass_guide_head.c_str());
  }
  fprintf(fp,
          "# columns: sched_index original_index source_file source_line uid "
          "class_token class_name opcode source\n");

  unsigned sched_index = 0;
  for (std::list<ptx_instruction *>::const_iterator it = instructions.begin();
       it != instructions.end(); ++it, ++sched_index) {
    const ptx_instruction *inst = *it;
    unsigned original_index = sched_index;
    std::map<const ptx_instruction *, unsigned>::const_iterator found =
        original_indices.find(inst);
    if (found != original_indices.end())
      original_index = found->second;

    const inst_class_t cls = classify_inst(inst);
    fprintf(fp, "%06u orig=%06u src=%s:%u uid=%u class=%c/%s opcode=%s | %s\n",
            sched_index, original_index, inst->source_file(),
            inst->source_line(), inst->uid(), class_token(cls), class_name(cls),
            inst->get_opcode_cstr(), inst->get_source());
  }

  fclose(fp);
}

bool ptx_opcode_has_dst(int opcode) {
  switch (opcode) {
#define OP_DEF(OP, FUNC, STR, DST, CLASSIFICATION)                             \
  case OP:                                                                     \
    return DST != 0;
#define OP_W_DEF(OP, FUNC, STR, DST, CLASSIFICATION)                           \
  case OP:                                                                     \
    return DST != 0;
#include "../../../cuda-sim/opcodes.def"
#undef OP_DEF
#undef OP_W_DEF
  default:
    return false;
  }
}

void add_reg(reg_set_t &regs, const symbol *sym) {
  if (sym == NULL)
    return;
  if (!sym->is_reg() || sym->is_non_arch_reg())
    return;
  if (sym->name() == "_")
    return;
  regs.insert(sym);
}

void collect_operand_regs(const operand_info &op, reg_set_t &regs) {
  if (op.is_vector()) {
    for (unsigned i = 0; i < op.get_vect_nelem(); ++i)
      add_reg(regs, op.vec_symbol(i));
    return;
  }

  if (op.is_memory_operand()) {
    add_reg(regs, op.get_symbol());
    return;
  }

  if (op.is_reg() || op.get_type() == address_t)
    add_reg(regs, op.get_symbol());
}

void collect_inst_regs(const ptx_instruction *inst, reg_set_t &uses,
                       reg_set_t &defs) {
  if (inst == NULL || inst->is_label())
    return;

  if (inst->has_pred())
    collect_operand_regs(inst->get_pred(), uses);

  const std::vector<operand_info> &operands = inst->get_operands();
  if (inst->get_opcode() == SETP_OP && operands.size() >= 2) {
    collect_operand_regs(operands[0], defs);
    collect_operand_regs(operands[1], defs);
    for (unsigned i = 2; i < operands.size(); ++i)
      collect_operand_regs(operands[i], uses);
    return;
  }

  const bool has_dst = ptx_opcode_has_dst(inst->get_opcode());
  const bool dst_is_accumulator = inst->get_opcode() == MMA_OP ||
                                  inst->get_opcode() == TENSOR_MMA_OP ||
                                  inst->get_opcode() == WGMMA_MMA_ASYNC_OP ||
                                  inst->get_opcode() == WGMMA_MMA_ASYNC_SP_OP;

  for (unsigned i = 0; i < operands.size(); ++i) {
    const operand_info &op = operands[i];
    if (has_dst && i == 0) {
      collect_operand_regs(op, defs);
      if (dst_is_accumulator || op.get_operand_lohi() != 0 ||
          op.get_double_operand_type() != 0)
        collect_operand_regs(op, uses);
      continue;
    }

    collect_operand_regs(op, uses);
    if (op.is_memory_operand2() && (op.get_double_operand_type() == 2 ||
                                    op.get_double_operand_type() == 3))
      collect_operand_regs(op, defs);
  }
}

inst_class_t classify_inst(const ptx_instruction *inst) {
  if (inst == NULL || inst->is_label())
    return inst_class_t::boundary;

  switch (inst->get_opcode()) {
  case MMA_OP:
  case TENSOR_MMA_OP:
    return inst_class_t::tensor;

  case LDMATRIX_OP:
  case STMATRIX_OP:
    return inst_class_t::ldmatrix;

  case EX2_OP:
  case LG2_OP:
  case RCP_OP:
  case RSQRT_OP:
  case SQRT_OP:
  case SIN_OP:
  case COS_OP:
  case TANH_OP:
  case DIV_OP:
    return inst_class_t::sfu;

  case FMA_OP:
  case MAD_OP:
  case MUL_OP:
  case ADD_OP:
  case SUB_OP:
  case MAX_OP:
  case MIN_OP:
  case CVT_OP:
    return inst_class_t::fp32;

  case SHFL_OP:
    return inst_class_t::shfl;

  case LD_OP:
  case LDU_OP:
  case ST_OP:
  case ATOM_OP:
  case RED_OP:
  case PREFETCH_OP:
  case PREFETCHU_OP:
  case CP_ASYNC_OP:
  case TMA_OP:
  case TMA_PREFETCH_OP:
  case TENSORMAP_OP:
    return inst_class_t::mem;

  case BRA_OP:
  case BRX_OP:
  case BRKPT_OP:
  case BREAK_OP:
  case BREAKADDR_OP:
  case CALL_OP:
  case CALLP_OP:
  case RET_OP:
  case RETP_OP:
  case EXIT_OP:
  case TRAP_OP:
  case BAR_OP:
  case MBAR_OP:
  case MEMBAR_OP:
  case FENCE_OP:
  case CP_ASYNC_COMMIT_OP:
  case CP_ASYNC_WAIT_OP:
  case WGMMA_MMA_ASYNC_OP:
  case WGMMA_MMA_ASYNC_SP_OP:
  case WGMMA_FENCE_OP:
  case WGMMA_COMMIT_GROUP_OP:
  case WGMMA_WAIT_GROUP_OP:
  case SETMAXNREG_OP:
  case GRIDDEPCONTROL_OP:
    return inst_class_t::control;

  case ADDP_OP:
  case ADDC_OP:
  case AND_OP:
  case ANDN_OP:
  case BFE_OP:
  case BFI_OP:
  case BFIND_OP:
  case BREV_OP:
  case CLZ_OP:
  case CNOT_OP:
  case CVTA_OP:
  case ISSPACEP_OP:
  case MAPA_OP:
  case MOV_OP:
  case MUL24_OP:
  case NANDN_OP:
  case NORN_OP:
  case NOT_OP:
  case OR_OP:
  case ORN_OP:
  case POPC_OP:
  case PRMT_OP:
  case REM_OP:
  case SAD_OP:
  case SELP_OP:
  case SETP_OP:
  case SET_OP:
  case SHF_OP:
  case SHL_OP:
  case SHR_OP:
  case SLCT_OP:
  case VOTE_OP:
  case ACTIVEMASK_OP:
  case XOR_OP:
  case ELECT_OP:
    return inst_class_t::intp;

  default:
    return inst_class_t::other;
  }
}

bool has_unsupported_operand_form(const ptx_instruction *inst,
                                  bool allow_ldmatrix_memory_operand) {
  if (inst == NULL || inst->is_label())
    return false;
  const bool allow_memory_operand =
      allow_ldmatrix_memory_operand && inst->get_opcode() == LDMATRIX_OP;
  const std::vector<operand_info> &operands = inst->get_operands();
  for (unsigned i = 0; i < operands.size(); ++i) {
    if (operands[i].is_memory_operand() && !allow_memory_operand)
      return true;
  }
  return false;
}

bool is_segment_boundary(const ptx_instruction *inst,
                         bool allow_ldmatrix_memory_operand = false) {
  inst_class_t cls = classify_inst(inst);
  return cls == inst_class_t::boundary || cls == inst_class_t::control ||
         has_unsupported_operand_form(inst, allow_ldmatrix_memory_operand);
}

bool is_memory_like(inst_class_t cls) {
  return cls == inst_class_t::mem || cls == inst_class_t::ldmatrix;
}

bool is_barrier_inst(const sched_inst_t &inst) {
  return inst.inst != NULL && inst.inst->get_opcode() == BAR_OP;
}

bool is_barrier_sensitive(const sched_inst_t &inst) {
  return inst.cls == inst_class_t::ldmatrix || inst.cls == inst_class_t::mem ||
         inst.cls == inst_class_t::control ||
         inst.cls == inst_class_t::boundary;
}

char class_token(inst_class_t cls) {
  switch (cls) {
  case inst_class_t::tensor:
    return 'T';
  case inst_class_t::ldmatrix:
    return 'L';
  case inst_class_t::sfu:
    return 'S';
  case inst_class_t::fp32:
    return 'F';
  case inst_class_t::shfl:
    return 'H';
  case inst_class_t::mem:
    return 'M';
  case inst_class_t::intp:
    return 'I';
  case inst_class_t::control:
    return 'C';
  case inst_class_t::other:
    return '.';
  case inst_class_t::boundary:
    return 'B';
  }
  return '?';
}

pipe_t inst_pipe(inst_class_t cls) {
  switch (cls) {
  case inst_class_t::tensor:
    return pipe_t::tensor;
  case inst_class_t::sfu:
    return pipe_t::sfu;
  case inst_class_t::ldmatrix:
  case inst_class_t::mem:
    return pipe_t::ldst;
  case inst_class_t::fp32:
    return pipe_t::fp32;
  case inst_class_t::shfl:
    return pipe_t::xbar;
  case inst_class_t::intp:
    return pipe_t::intp;
  case inst_class_t::control:
    return pipe_t::control;
  default:
    return pipe_t::other;
  }
}

unsigned inst_latency(const sched_inst_t &inst) {
  if (inst.inst->get_opcode() == CP_ASYNC_OP)
    return 1;

  switch (inst.cls) {
  case inst_class_t::tensor:
    return 32;
  case inst_class_t::sfu:
    return 28;
  case inst_class_t::ldmatrix:
  case inst_class_t::mem:
    return 8;
  case inst_class_t::fp32:
  case inst_class_t::shfl:
  case inst_class_t::intp:
    return 4;
  default:
    return 1;
  }
}

unsigned inst_initiation(const sched_inst_t &inst) {
  if (inst.inst->get_opcode() == CP_ASYNC_OP)
    return 1;

  switch (inst.cls) {
  case inst_class_t::tensor:
  case inst_class_t::sfu:
    return 8;
  default:
    return 1;
  }
}

double switch_bonus(char last, char next) {
  switch (last) {
  case 'T':
    switch (next) {
    case 'L':
      return 16.0;
    case 'F':
      return 12.0;
    case 'I':
      return 7.0;
    case 'S':
      return 5.0;
    case 'T':
      return 3.0;
    }
    break;
  case 'L':
    switch (next) {
    case 'T':
      return 18.0;
    case 'F':
      return 6.0;
    case 'I':
      return 5.0;
    case 'S':
      return 4.0;
    case 'L':
      return 2.0;
    }
    break;
  case 'F':
    switch (next) {
    case 'T':
      return 10.0;
    case 'I':
      return 10.0;
    case 'S':
      return 8.0;
    case 'M':
      return 4.0;
    case 'L':
      return 2.0;
    case 'F':
      return 2.0;
    }
    break;
  case 'S':
    switch (next) {
    case 'F':
      return 18.0;
    case 'I':
      return 8.0;
    case 'T':
      return 7.0;
    case 'L':
      return 2.0;
    }
    break;
  case 'M':
    switch (next) {
    case 'I':
      return 16.0;
    case 'F':
      return 8.0;
    case '.':
      return 4.0;
    case 'M':
      return 2.0;
    }
    break;
  case 'I':
    switch (next) {
    case 'M':
      return 12.0;
    case 'F':
      return 10.0;
    case 'T':
      return 6.0;
    case '.':
      return 5.0;
    case 'S':
      return 4.0;
    case 'L':
      return 2.0;
    case 'I':
      return 1.0;
    }
    break;
  case '.':
    switch (next) {
    case 'I':
      return 12.0;
    case 'F':
      return 7.0;
    case 'M':
      return 4.0;
    case 'T':
      return 2.0;
    }
    break;
  case 'C':
    switch (next) {
    case 'I':
      return 8.0;
    case '.':
      return 4.0;
    case 'T':
      return 2.0;
    }
    break;
  case 'H':
    switch (next) {
    case 'F':
      return 10.0;
    case 'I':
      return 2.0;
    case 'H':
      return 1.0;
    }
    break;
  }
  return 0.0;
}

void add_edge(dep_graph_t &graph,
              std::map<std::pair<unsigned, unsigned>, unsigned> &edge_index,
              unsigned src, unsigned dst, unsigned latency, bool raw = false) {
  if (src == dst)
    return;

  std::pair<unsigned, unsigned> key(src, dst);
  std::map<std::pair<unsigned, unsigned>, unsigned>::iterator found =
      edge_index.find(key);
  if (found != edge_index.end()) {
    dep_edge_t &edge = graph.edges[found->second];
    edge.latency = std::max(edge.latency, latency);
    edge.raw = edge.raw || raw;
    return;
  }

  edge_index[key] = graph.edges.size();
  graph.succ[src].insert(dst);
  ++graph.indeg[dst];
  dep_edge_t edge;
  edge.src = src;
  edge.dst = dst;
  edge.latency = latency;
  edge.raw = raw;
  graph.edges.push_back(edge);
}

dep_graph_t build_dependency_graph(const std::vector<sched_inst_t> &chunk,
                                   bool relax_ldmatrix_order,
                                   bool relax_barrier_reg) {
  dep_graph_t graph;
  graph.succ.resize(chunk.size());
  graph.indeg.assign(chunk.size(), 0);
  std::map<std::pair<unsigned, unsigned>, unsigned> edge_index;

  std::map<const symbol *, unsigned> last_def;
  std::map<const symbol *, std::set<unsigned>> last_uses;
  bool have_last_mem = false;
  unsigned last_mem = 0;
  bool have_last_barrier = false;
  unsigned last_barrier = 0;
  std::vector<unsigned> barrier_sensitive_since_last;

  for (unsigned i = 0; i < chunk.size(); ++i) {
    const sched_inst_t &inst = chunk[i];
    if (relax_barrier_reg && is_barrier_inst(inst)) {
      for (std::vector<unsigned>::const_iterator prior =
               barrier_sensitive_since_last.begin();
           prior != barrier_sensitive_since_last.end(); ++prior)
        add_edge(graph, edge_index, *prior, i, 0);
      have_last_barrier = true;
      last_barrier = i;
      barrier_sensitive_since_last.clear();
    }

    for (reg_set_t::const_iterator use = inst.uses.begin();
         use != inst.uses.end(); ++use) {
      std::map<const symbol *, unsigned>::const_iterator def =
          last_def.find(*use);
      if (def != last_def.end())
        add_edge(graph, edge_index, def->second, i,
                 inst_latency(chunk[def->second]), true);
      last_uses[*use].insert(i);
    }

    for (reg_set_t::const_iterator def = inst.defs.begin();
         def != inst.defs.end(); ++def) {
      std::map<const symbol *, unsigned>::const_iterator old_def =
          last_def.find(*def);
      if (old_def != last_def.end())
        add_edge(graph, edge_index, old_def->second, i, 0);

      std::map<const symbol *, std::set<unsigned>>::const_iterator uses =
          last_uses.find(*def);
      if (uses != last_uses.end()) {
        for (std::set<unsigned>::const_iterator use = uses->second.begin();
             use != uses->second.end(); ++use)
          add_edge(graph, edge_index, *use, i, 0);
      }

      last_def[*def] = i;
      last_uses[*def].clear();
    }

    if (is_memory_like(inst.cls)) {
      if (have_last_mem &&
          !(relax_ldmatrix_order && inst.cls == inst_class_t::ldmatrix &&
            chunk[last_mem].cls == inst_class_t::ldmatrix))
        add_edge(graph, edge_index, last_mem, i, 0);
      if (!(relax_ldmatrix_order && inst.cls == inst_class_t::ldmatrix)) {
        have_last_mem = true;
        last_mem = i;
      }
    }

    if (relax_barrier_reg && !is_barrier_inst(inst) &&
        is_barrier_sensitive(inst)) {
      if (have_last_barrier)
        add_edge(graph, edge_index, last_barrier, i, 0);
      barrier_sensitive_since_last.push_back(i);
    }
  }

  return graph;
}

std::vector<role_signature_t>
compute_ptx_role_signatures(const std::vector<sched_inst_t> &chunk,
                            const dep_graph_t &graph) {
  std::vector<role_signature_t> out(chunk.size());
  std::map<char, unsigned> class_seen;
  for (unsigned i = 0; i < chunk.size(); ++i) {
    const char token = class_token(chunk[i].cls);
    out[i].token = token;
    out[i].rank = class_seen[token]++;
  }

  std::vector<std::set<unsigned>> ldmatrix_tensor_succ(chunk.size());
  std::vector<std::set<unsigned>> tensor_ldmatrix_pred(chunk.size());
  std::vector<std::set<unsigned>> tensor_tensor_pred(chunk.size());
  for (unsigned i = 0; i < graph.edges.size(); ++i) {
    const dep_edge_t &edge = graph.edges[i];
    if (!edge.raw)
      continue;
    if (chunk[edge.src].cls == inst_class_t::ldmatrix &&
        chunk[edge.dst].cls == inst_class_t::tensor) {
      ldmatrix_tensor_succ[edge.src].insert(edge.dst);
      tensor_ldmatrix_pred[edge.dst].insert(edge.src);
    } else if (chunk[edge.src].cls == inst_class_t::tensor &&
               chunk[edge.dst].cls == inst_class_t::tensor) {
      tensor_tensor_pred[edge.dst].insert(edge.src);
    }
  }

  std::map<unsigned, int> tensor_stage;
  std::map<unsigned, int> tensor_chain;
  int next_chain = 0;
  for (unsigned i = 0; i < chunk.size(); ++i) {
    if (chunk[i].cls != inst_class_t::tensor)
      continue;
    int stage = 0;
    bool have_stage = false;
    int chain = -1;
    for (std::set<unsigned>::const_iterator pred =
             tensor_tensor_pred[i].begin();
         pred != tensor_tensor_pred[i].end(); ++pred) {
      std::map<unsigned, int>::const_iterator pred_stage =
          tensor_stage.find(*pred);
      if (pred_stage != tensor_stage.end()) {
        stage = std::max(stage, pred_stage->second + 1);
        have_stage = true;
      }
      std::map<unsigned, int>::const_iterator pred_chain =
          tensor_chain.find(*pred);
      if (pred_chain != tensor_chain.end())
        chain = chain < 0 ? pred_chain->second
                          : std::min(chain, pred_chain->second);
    }
    tensor_stage[i] = have_stage ? stage : 0;
    if (chain < 0)
      chain = next_chain++;
    tensor_chain[i] = chain;
  }

  std::map<unsigned, int> ldmatrix_stage;
  for (unsigned i = 0; i < chunk.size(); ++i) {
    if (ldmatrix_tensor_succ[i].empty())
      continue;
    int stage = -1;
    for (std::set<unsigned>::const_iterator succ =
             ldmatrix_tensor_succ[i].begin();
         succ != ldmatrix_tensor_succ[i].end(); ++succ) {
      std::map<unsigned, int>::const_iterator found = tensor_stage.find(*succ);
      if (found != tensor_stage.end())
        stage = stage < 0 ? found->second : std::min(stage, found->second);
    }
    if (stage >= 0)
      ldmatrix_stage[i] = stage;
  }

  for (unsigned i = 0; i < chunk.size(); ++i) {
    role_signature_t &sig = out[i];
    std::map<unsigned, int>::const_iterator ts = tensor_stage.find(i);
    std::map<unsigned, int>::const_iterator ls = ldmatrix_stage.find(i);
    std::map<unsigned, int>::const_iterator tc = tensor_chain.find(i);
    sig.stage = ts != tensor_stage.end()
                    ? ts->second
                    : (ls != ldmatrix_stage.end() ? ls->second : -1);
    sig.chain = tc != tensor_chain.end() ? tc->second : -1;
    sig.pred_ld_count = tensor_ldmatrix_pred[i].size();
    sig.pred_tensor_count = tensor_tensor_pred[i].size();
    sig.succ_tensor_count = ldmatrix_tensor_succ[i].size();
    for (std::set<unsigned>::const_iterator pred =
             tensor_ldmatrix_pred[i].begin();
         pred != tensor_ldmatrix_pred[i].end(); ++pred) {
      sig.pred_ld_fanout.push_back(ldmatrix_tensor_succ[*pred].size());
      std::map<unsigned, int>::const_iterator pred_stage =
          ldmatrix_stage.find(*pred);
      sig.pred_ld_stage.push_back(
          pred_stage == ldmatrix_stage.end() ? -1 : pred_stage->second);
    }
    for (std::set<unsigned>::const_iterator succ =
             ldmatrix_tensor_succ[i].begin();
         succ != ldmatrix_tensor_succ[i].end(); ++succ) {
      std::map<unsigned, int>::const_iterator succ_stage =
          tensor_stage.find(*succ);
      std::map<unsigned, int>::const_iterator succ_chain =
          tensor_chain.find(*succ);
      sig.succ_tensor_stage.push_back(
          succ_stage == tensor_stage.end() ? -1 : succ_stage->second);
      sig.succ_tensor_chain.push_back(
          succ_chain == tensor_chain.end() ? -1 : succ_chain->second);
    }
    sort_int_vector(sig.pred_ld_fanout);
    sort_int_vector(sig.pred_ld_stage);
    sort_int_vector(sig.succ_tensor_stage);
    sort_int_vector(sig.succ_tensor_chain);
  }

  return out;
}

double tuple_overlap_score(const std::vector<int> &a, const std::vector<int> &b,
                           double weight) {
  if (a.empty() && b.empty())
    return weight;
  if (a == b)
    return weight;

  std::map<int, unsigned> ca;
  std::map<int, unsigned> cb;
  for (unsigned i = 0; i < a.size(); ++i)
    ++ca[a[i]];
  for (unsigned i = 0; i < b.size(); ++i)
    ++cb[b[i]];

  unsigned intersection = 0;
  unsigned union_count = 0;
  std::set<int> keys;
  for (std::map<int, unsigned>::const_iterator it = ca.begin(); it != ca.end();
       ++it)
    keys.insert(it->first);
  for (std::map<int, unsigned>::const_iterator it = cb.begin(); it != cb.end();
       ++it)
    keys.insert(it->first);

  for (std::set<int>::const_iterator key = keys.begin(); key != keys.end();
       ++key) {
    const unsigned av = ca[*key];
    const unsigned bv = cb[*key];
    intersection += std::min(av, bv);
    union_count += std::max(av, bv);
  }
  return union_count == 0 ? 0.0
                          : weight * static_cast<double>(intersection) /
                                static_cast<double>(union_count);
}

double role_signature_similarity(const role_signature_t &candidate,
                                 const role_signature_t &guide) {
  if (candidate.token != guide.token)
    return -1000.0;

  double score = 4.0;
  if (candidate.stage >= 0 && guide.stage >= 0) {
    if (candidate.stage == guide.stage)
      score += 4.0;
    else
      score -= std::min(
          4.0, static_cast<double>(std::abs(candidate.stage - guide.stage)));
  }

  if (candidate.token == 'L') {
    score +=
        candidate.succ_tensor_count == guide.succ_tensor_count
            ? 4.0
            : -2.0 * std::abs(static_cast<int>(candidate.succ_tensor_count) -
                              static_cast<int>(guide.succ_tensor_count));
    score += tuple_overlap_score(candidate.succ_tensor_stage,
                                 guide.succ_tensor_stage, 4.0);
    score += tuple_overlap_score(candidate.succ_tensor_chain,
                                 guide.succ_tensor_chain, 10.0);
  } else if (candidate.token == 'T') {
    if (candidate.chain >= 0 && guide.chain >= 0) {
      if (candidate.chain == guide.chain)
        score += 12.0;
      else
        score -= std::min(8.0, 0.5 * std::abs(candidate.chain - guide.chain));
    }
    score += candidate.pred_ld_count == guide.pred_ld_count
                 ? 4.0
                 : -2.0 * std::abs(static_cast<int>(candidate.pred_ld_count) -
                                   static_cast<int>(guide.pred_ld_count));
    score +=
        candidate.pred_tensor_count == guide.pred_tensor_count
            ? 2.0
            : -1.0 * std::abs(static_cast<int>(candidate.pred_tensor_count) -
                              static_cast<int>(guide.pred_tensor_count));
    score += tuple_overlap_score(candidate.pred_ld_fanout, guide.pred_ld_fanout,
                                 3.0);
    score +=
        tuple_overlap_score(candidate.pred_ld_stage, guide.pred_ld_stage, 3.0);
  }

  score -= 0.02 * std::abs(static_cast<int>(candidate.rank) -
                           static_cast<int>(guide.rank));
  return score;
}

std::vector<sched_inst_t>
schedule_switch(const std::vector<sched_inst_t> &chunk, int ready_slack,
                unsigned *edge_count, bool *valid) {
  *valid = true;
  dep_graph_t graph = build_dependency_graph(chunk, false, false);
  *edge_count = graph.edges.size();

  std::vector<std::vector<dep_edge_t>> edge_by_src(chunk.size());
  for (unsigned i = 0; i < graph.edges.size(); ++i)
    edge_by_src[graph.edges[i].src].push_back(graph.edges[i]);

  std::vector<unsigned> height(chunk.size(), 1);
  for (int i = static_cast<int>(chunk.size()) - 1; i >= 0; --i) {
    height[i] = inst_latency(chunk[i]);
    for (std::vector<dep_edge_t>::const_iterator edge = edge_by_src[i].begin();
         edge != edge_by_src[i].end(); ++edge) {
      height[i] = std::max(height[i], edge->latency + height[edge->dst]);
    }
  }

  std::set<unsigned> ready;
  for (unsigned i = 0; i < graph.indeg.size(); ++i) {
    if (graph.indeg[i] == 0)
      ready.insert(i);
  }

  std::vector<unsigned> dep_ready(chunk.size(), 0);
  std::vector<unsigned> pipe_ready(static_cast<unsigned>(pipe_t::count), 0);
  unsigned warp_issue_ready = 0;
  std::vector<unsigned> emitted;
  std::deque<char> recent;
  char last_token = 0;

  while (!ready.empty()) {
    std::map<unsigned, unsigned> issues;
    unsigned min_issue = std::numeric_limits<unsigned>::max();
    for (std::set<unsigned>::const_iterator it = ready.begin();
         it != ready.end(); ++it) {
      const sched_inst_t &inst = chunk[*it];
      const unsigned pipe_index = static_cast<unsigned>(inst_pipe(inst.cls));
      unsigned issue = std::max(dep_ready[*it], pipe_ready[pipe_index]);
      issue = std::max(issue, warp_issue_ready);
      issues[*it] = issue;
      min_issue = std::min(min_issue, issue);
    }

    bool have_pick = false;
    unsigned pick = 0;
    double best_score = -std::numeric_limits<double>::infinity();
    int best_neg_issue = std::numeric_limits<int>::min();
    unsigned best_height = 0;
    int best_neg_index = std::numeric_limits<int>::min();
    const unsigned slack =
        ready_slack < 0 ? 0 : static_cast<unsigned>(ready_slack);

    for (std::set<unsigned>::const_iterator it = ready.begin();
         it != ready.end(); ++it) {
      const unsigned idx = *it;
      const unsigned issue = issues[idx];
      if (issue > min_issue + slack)
        continue;

      const sched_inst_t &inst = chunk[idx];
      const char token = class_token(inst.cls);
      unsigned same_recent = 0;
      for (std::deque<char>::const_iterator r = recent.begin();
           r != recent.end(); ++r) {
        if (*r == token)
          ++same_recent;
      }

      double score = -0.02 * static_cast<double>(issue) +
                     0.001 * static_cast<double>(height[idx]) -
                     0.0001 * static_cast<double>(inst.original_index);
      if (last_token != 0) {
        score += 0.6 * switch_bonus(last_token, token);
        if ((last_token == 'T' && token == 'L') ||
            (last_token == 'L' && token == 'T'))
          score += 4.0;
        score -= 2.0 * static_cast<double>(same_recent);
      }

      const int neg_issue = -static_cast<int>(issue);
      const int neg_index = -static_cast<int>(inst.original_index);
      const bool better =
          !have_pick || score > best_score + 1e-12 ||
          (std::fabs(score - best_score) <= 1e-12 &&
           (neg_issue > best_neg_issue ||
            (neg_issue == best_neg_issue &&
             (height[idx] > best_height ||
              (height[idx] == best_height && neg_index > best_neg_index)))));
      if (better) {
        have_pick = true;
        pick = idx;
        best_score = score;
        best_neg_issue = neg_issue;
        best_height = height[idx];
        best_neg_index = neg_index;
      }
    }

    if (!have_pick) {
      *valid = false;
      return chunk;
    }

    ready.erase(pick);
    emitted.push_back(pick);

    const sched_inst_t &inst = chunk[pick];
    const unsigned issue = issues[pick];
    const unsigned pipe_index = static_cast<unsigned>(inst_pipe(inst.cls));
    pipe_ready[pipe_index] = issue + inst_initiation(inst);
    warp_issue_ready = issue + 1;

    last_token = class_token(inst.cls);
    recent.push_back(last_token);
    if (recent.size() > 12)
      recent.pop_front();

    for (std::vector<dep_edge_t>::const_iterator edge =
             edge_by_src[pick].begin();
         edge != edge_by_src[pick].end(); ++edge) {
      dep_ready[edge->dst] =
          std::max(dep_ready[edge->dst], issue + edge->latency);
      if (graph.indeg[edge->dst] == 0) {
        *valid = false;
        return chunk;
      }
      --graph.indeg[edge->dst];
      if (graph.indeg[edge->dst] == 0)
        ready.insert(edge->dst);
    }
  }

  if (emitted.size() != chunk.size()) {
    *valid = false;
    return chunk;
  }

  std::vector<sched_inst_t> scheduled;
  scheduled.reserve(chunk.size());
  for (unsigned i = 0; i < emitted.size(); ++i)
    scheduled.push_back(chunk[emitted[i]]);
  return scheduled;
}

bool contains_class(const std::vector<sched_inst_t> &chunk, inst_class_t cls) {
  for (unsigned i = 0; i < chunk.size(); ++i) {
    if (chunk[i].cls == cls)
      return true;
  }
  return false;
}

bool contains_barrier_inst(const std::vector<sched_inst_t> &chunk) {
  for (unsigned i = 0; i < chunk.size(); ++i) {
    if (is_barrier_inst(chunk[i]))
      return true;
  }
  return false;
}

void count_lt_classes(const std::list<ptx_instruction *> &instructions,
                      unsigned *tensor_count, unsigned *ldmatrix_count) {
  *tensor_count = 0;
  *ldmatrix_count = 0;
  for (std::list<ptx_instruction *>::const_iterator it = instructions.begin();
       it != instructions.end(); ++it) {
    const inst_class_t cls = classify_inst(*it);
    if (cls == inst_class_t::tensor)
      ++(*tensor_count);
    else if (cls == inst_class_t::ldmatrix)
      ++(*ldmatrix_count);
  }
}

double sass_guide_target_score(const std::string &guide, unsigned cursor,
                               unsigned lookahead, char token,
                               unsigned *next_cursor) {
  if (next_cursor != NULL)
    *next_cursor = cursor;
  if (guide.empty() || (token != 'T' && token != 'L'))
    return 0.0;

  const unsigned end = std::min<unsigned>(
      guide.size(), cursor + std::max<unsigned>(1, lookahead));
  for (unsigned i = cursor; i < end; ++i) {
    if (guide[i] == token) {
      if (next_cursor != NULL)
        *next_cursor = i + 1;
      return -static_cast<double>(i - cursor);
    }
  }
  return -static_cast<double>(std::max<unsigned>(1, lookahead));
}

std::vector<sched_inst_t>
schedule_sass_guided(const std::vector<sched_inst_t> &chunk,
                     const sass_function_guide_t &guide, unsigned lookahead,
                     unsigned *guide_cursor, unsigned *edge_count,
                     bool *valid) {
  *valid = true;
  if (guide.lt_stream.empty() || guide.guide_items.empty() ||
      guide_cursor == NULL) {
    *valid = false;
    return chunk;
  }

  dep_graph_t graph = build_dependency_graph(chunk, true, false);
  *edge_count = graph.edges.size();

  std::vector<std::vector<dep_edge_t>> edge_by_src(chunk.size());
  std::vector<std::vector<unsigned>> pred_by_dst(chunk.size());
  for (unsigned i = 0; i < graph.edges.size(); ++i) {
    edge_by_src[graph.edges[i].src].push_back(graph.edges[i]);
    pred_by_dst[graph.edges[i].dst].push_back(graph.edges[i].src);
  }

  std::vector<unsigned> height(chunk.size(), 1);
  for (int i = static_cast<int>(chunk.size()) - 1; i >= 0; --i) {
    height[i] = inst_latency(chunk[i]);
    for (std::vector<dep_edge_t>::const_iterator edge = edge_by_src[i].begin();
         edge != edge_by_src[i].end(); ++edge) {
      height[i] = std::max(height[i], edge->latency + height[edge->dst]);
    }
  }

  std::set<unsigned> ready;
  for (unsigned i = 0; i < graph.indeg.size(); ++i) {
    if (graph.indeg[i] == 0)
      ready.insert(i);
  }

  std::vector<unsigned> emitted;
  std::vector<bool> emitted_flag(chunk.size(), false);
  std::deque<inst_class_t> recent;
  unsigned cursor = *guide_cursor;
  std::map<char, unsigned> remaining_guided;
  for (unsigned i = 0; i < chunk.size(); ++i) {
    const char token = class_token(chunk[i].cls);
    if (token == 'T' || token == 'L')
      ++remaining_guided[token];
  }
  const std::vector<role_signature_t> role_sigs =
      compute_ptx_role_signatures(chunk, graph);

  struct guide_skipper_t {
    const sass_function_guide_t &guide;
    const std::map<char, unsigned> &remaining;
    void operator()(unsigned &cursor) const {
      while (cursor < guide.guide_items.size()) {
        const char token = guide.guide_items[cursor].token;
        std::map<char, unsigned>::const_iterator found = remaining.find(token);
        if (found != remaining.end() && found->second != 0)
          break;
        ++cursor;
      }
    }
  } skip_unavailable = {guide, remaining_guided};
  skip_unavailable(cursor);

  while (!ready.empty()) {
    skip_unavailable(cursor);
    const guide_item_t *current_guide =
        cursor < guide.guide_items.size() ? &guide.guide_items[cursor] : NULL;
    bool have_desired_node = false;
    unsigned desired_node = 0;
    double best_identity = -std::numeric_limits<double>::infinity();
    if (current_guide != NULL) {
      for (unsigned i = 0; i < chunk.size(); ++i) {
        if (emitted_flag[i] ||
            class_token(chunk[i].cls) != current_guide->token)
          continue;
        const double identity =
            role_signature_similarity(role_sigs[i], current_guide->sig);
        if (!have_desired_node || identity > best_identity) {
          have_desired_node = true;
          desired_node = i;
          best_identity = identity;
        }
      }
    }

    std::set<unsigned> desired_ready_ancestors;
    if (have_desired_node && ready.find(desired_node) == ready.end()) {
      std::vector<unsigned> stack = pred_by_dst[desired_node];
      std::set<unsigned> seen;
      while (!stack.empty()) {
        const unsigned pred = stack.back();
        stack.pop_back();
        if (seen.find(pred) != seen.end() || emitted_flag[pred])
          continue;
        seen.insert(pred);
        if (ready.find(pred) != ready.end()) {
          desired_ready_ancestors.insert(pred);
        } else {
          const std::vector<unsigned> &preds = pred_by_dst[pred];
          stack.insert(stack.end(), preds.begin(), preds.end());
        }
      }
    }

    bool have_pick = false;
    unsigned pick = 0;
    double best_score = -std::numeric_limits<double>::infinity();
    double best_target_score = -std::numeric_limits<double>::infinity();
    unsigned best_height = 0;
    int best_neg_index = std::numeric_limits<int>::min();

    for (std::set<unsigned>::const_iterator it = ready.begin();
         it != ready.end(); ++it) {
      const unsigned idx = *it;
      const sched_inst_t &inst = chunk[idx];
      const char token = class_token(inst.cls);
      const bool guided_token = token == 'T' || token == 'L';
      double target_score = 0.0;
      if (current_guide != NULL && guided_token &&
          token == current_guide->token) {
        const double identity = std::max(
            0.0, role_signature_similarity(role_sigs[idx], current_guide->sig));
        if (have_desired_node && idx == desired_node)
          target_score = 100000.0 + 0.25 * identity;
        else
          target_score = 90000.0 + 0.25 * identity;
      } else if (current_guide != NULL && have_desired_node &&
                 desired_ready_ancestors.find(idx) !=
                     desired_ready_ancestors.end()) {
        target_score = 50000.0 + 0.001 * static_cast<double>(height[idx]);
      } else if (current_guide != NULL && guided_token) {
        target_score = -100000.0;
      } else if (guided_token) {
        target_score = sass_guide_target_score(guide.lt_stream, cursor,
                                               lookahead, token, NULL);
      }

      unsigned same_recent = 0;
      for (std::deque<inst_class_t>::const_iterator r = recent.begin();
           r != recent.end(); ++r) {
        if (*r == inst.cls)
          ++same_recent;
      }
      const double age = -static_cast<double>(inst.original_index);
      const double score =
          32.0 * target_score + 0.05 * static_cast<double>(height[idx]) +
          0.002 * age - 0.25 * static_cast<double>(same_recent);

      const int neg_index = -static_cast<int>(inst.original_index);
      const bool better =
          !have_pick || score > best_score + 1e-12 ||
          (std::fabs(score - best_score) <= 1e-12 &&
           (target_score > best_target_score + 1e-12 ||
            (std::fabs(target_score - best_target_score) <= 1e-12 &&
             (height[idx] > best_height ||
              (height[idx] == best_height && neg_index > best_neg_index)))));
      if (better) {
        have_pick = true;
        pick = idx;
        best_score = score;
        best_target_score = target_score;
        best_height = height[idx];
        best_neg_index = neg_index;
      }
    }

    if (!have_pick) {
      *valid = false;
      return chunk;
    }

    ready.erase(pick);
    emitted.push_back(pick);
    emitted_flag[pick] = true;

    const sched_inst_t &inst = chunk[pick];
    const char token = class_token(inst.cls);
    if (token == 'T' || token == 'L') {
      if (remaining_guided[token] != 0)
        --remaining_guided[token];
      if (cursor < guide.guide_items.size() &&
          token == guide.guide_items[cursor].token) {
        ++cursor;
      } else {
        unsigned next_cursor = cursor;
        sass_guide_target_score(guide.lt_stream, cursor, lookahead, token,
                                &next_cursor);
        cursor = next_cursor;
      }
    }

    recent.push_back(inst.cls);
    if (recent.size() > 16)
      recent.pop_front();

    for (std::vector<dep_edge_t>::const_iterator edge =
             edge_by_src[pick].begin();
         edge != edge_by_src[pick].end(); ++edge) {
      if (graph.indeg[edge->dst] == 0) {
        *valid = false;
        return chunk;
      }
      --graph.indeg[edge->dst];
      if (graph.indeg[edge->dst] == 0)
        ready.insert(edge->dst);
    }
  }

  if (emitted.size() != chunk.size()) {
    *valid = false;
    return chunk;
  }

  *guide_cursor = cursor;

  std::vector<sched_inst_t> scheduled;
  scheduled.reserve(chunk.size());
  for (unsigned i = 0; i < emitted.size(); ++i)
    scheduled.push_back(chunk[emitted[i]]);
  return scheduled;
}

std::vector<ptxline_guide_item_t>
filter_ptxline_guide_for_segment(const std::vector<sched_inst_t> &chunk,
                                 const ptxline_guide_t &guide) {
  std::set<unsigned> segment_indices;
  for (unsigned i = 0; i < chunk.size(); ++i)
    segment_indices.insert(chunk[i].original_index);

  std::vector<ptxline_guide_item_t> out;
  for (std::vector<ptxline_guide_item_t>::const_iterator item =
           guide.items.begin();
       item != guide.items.end(); ++item) {
    if (segment_indices.find(item->original_index) != segment_indices.end())
      out.push_back(*item);
  }
  return out;
}

std::vector<sched_inst_t> schedule_sass_ptxline_guided(
    const std::vector<sched_inst_t> &chunk,
    const std::vector<ptxline_guide_item_t> &segment_guide,
    unsigned *edge_count, bool *valid) {
  *valid = true;
  if (segment_guide.empty()) {
    *valid = false;
    return chunk;
  }

  dep_graph_t graph = build_dependency_graph(chunk, true, false);
  *edge_count = graph.edges.size();

  std::vector<std::vector<dep_edge_t>> edge_by_src(chunk.size());
  std::vector<std::vector<unsigned>> pred_by_dst(chunk.size());
  for (unsigned i = 0; i < graph.edges.size(); ++i) {
    edge_by_src[graph.edges[i].src].push_back(graph.edges[i]);
    pred_by_dst[graph.edges[i].dst].push_back(graph.edges[i].src);
  }

  std::vector<unsigned> height(chunk.size(), 1);
  for (int i = static_cast<int>(chunk.size()) - 1; i >= 0; --i) {
    height[i] = inst_latency(chunk[i]);
    for (std::vector<dep_edge_t>::const_iterator edge = edge_by_src[i].begin();
         edge != edge_by_src[i].end(); ++edge) {
      height[i] = std::max(height[i], edge->latency + height[edge->dst]);
    }
  }

  std::map<unsigned, unsigned> chunk_by_original;
  for (unsigned i = 0; i < chunk.size(); ++i)
    chunk_by_original[chunk[i].original_index] = i;

  std::vector<unsigned> guide_nodes;
  std::set<unsigned> guided_node_set;
  for (std::vector<ptxline_guide_item_t>::const_iterator item =
           segment_guide.begin();
       item != segment_guide.end(); ++item) {
    std::map<unsigned, unsigned>::const_iterator found =
        chunk_by_original.find(item->original_index);
    if (found == chunk_by_original.end()) {
      *valid = false;
      return chunk;
    }
    guide_nodes.push_back(found->second);
    guided_node_set.insert(found->second);
  }

  std::set<unsigned> ready;
  for (unsigned i = 0; i < graph.indeg.size(); ++i) {
    if (graph.indeg[i] == 0)
      ready.insert(i);
  }

  std::vector<unsigned> emitted;
  std::vector<bool> emitted_flag(chunk.size(), false);
  std::deque<inst_class_t> recent;
  unsigned guide_pos = 0;

  while (!ready.empty()) {
    while (guide_pos < guide_nodes.size() &&
           emitted_flag[guide_nodes[guide_pos]])
      ++guide_pos;

    const bool have_current_guide = guide_pos < guide_nodes.size();
    const unsigned desired_node =
        have_current_guide ? guide_nodes[guide_pos] : 0;
    std::set<unsigned> desired_ready_ancestors;
    if (have_current_guide && ready.find(desired_node) == ready.end()) {
      std::vector<unsigned> stack = pred_by_dst[desired_node];
      std::set<unsigned> seen;
      while (!stack.empty()) {
        const unsigned pred = stack.back();
        stack.pop_back();
        if (seen.find(pred) != seen.end() || emitted_flag[pred])
          continue;
        seen.insert(pred);
        if (ready.find(pred) != ready.end()) {
          desired_ready_ancestors.insert(pred);
        } else {
          const std::vector<unsigned> &preds = pred_by_dst[pred];
          stack.insert(stack.end(), preds.begin(), preds.end());
        }
      }
      if (desired_ready_ancestors.empty()) {
        *valid = false;
        return chunk;
      }
    }

    bool have_pick = false;
    unsigned pick = 0;
    double best_score = -std::numeric_limits<double>::infinity();
    double best_target_score = -std::numeric_limits<double>::infinity();
    unsigned best_height = 0;
    int best_neg_index = std::numeric_limits<int>::min();

    for (std::set<unsigned>::const_iterator it = ready.begin();
         it != ready.end(); ++it) {
      const unsigned idx = *it;
      const sched_inst_t &inst = chunk[idx];
      double target_score = 0.0;
      if (have_current_guide && idx == desired_node) {
        target_score = 100000.0;
      } else if (have_current_guide && desired_ready_ancestors.find(idx) !=
                                           desired_ready_ancestors.end()) {
        target_score = 50000.0 + 0.001 * static_cast<double>(height[idx]);
      } else if (have_current_guide &&
                 guided_node_set.find(idx) != guided_node_set.end()) {
        target_score = -100000.0;
      }

      unsigned same_recent = 0;
      for (std::deque<inst_class_t>::const_iterator r = recent.begin();
           r != recent.end(); ++r) {
        if (*r == inst.cls)
          ++same_recent;
      }

      const double age = -static_cast<double>(inst.original_index);
      const double score =
          32.0 * target_score + 0.05 * static_cast<double>(height[idx]) +
          0.002 * age - 0.25 * static_cast<double>(same_recent);

      const int neg_index = -static_cast<int>(inst.original_index);
      const bool better =
          !have_pick || score > best_score + 1e-12 ||
          (std::fabs(score - best_score) <= 1e-12 &&
           (target_score > best_target_score + 1e-12 ||
            (std::fabs(target_score - best_target_score) <= 1e-12 &&
             (height[idx] > best_height ||
              (height[idx] == best_height && neg_index > best_neg_index)))));
      if (better) {
        have_pick = true;
        pick = idx;
        best_score = score;
        best_target_score = target_score;
        best_height = height[idx];
        best_neg_index = neg_index;
      }
    }

    if (!have_pick) {
      *valid = false;
      return chunk;
    }

    ready.erase(pick);
    emitted.push_back(pick);
    emitted_flag[pick] = true;

    recent.push_back(chunk[pick].cls);
    if (recent.size() > 16)
      recent.pop_front();

    for (std::vector<dep_edge_t>::const_iterator edge =
             edge_by_src[pick].begin();
         edge != edge_by_src[pick].end(); ++edge) {
      if (graph.indeg[edge->dst] == 0) {
        *valid = false;
        return chunk;
      }
      --graph.indeg[edge->dst];
      if (graph.indeg[edge->dst] == 0)
        ready.insert(edge->dst);
    }
  }

  if (emitted.size() != chunk.size()) {
    *valid = false;
    return chunk;
  }

  std::vector<sched_inst_t> scheduled;
  scheduled.reserve(chunk.size());
  for (unsigned i = 0; i < emitted.size(); ++i)
    scheduled.push_back(chunk[emitted[i]]);
  return scheduled;
}

void flush_segment(std::vector<sched_inst_t> &segment,
                   std::list<ptx_instruction *> &out, int ready_slack,
                   reorder_stats_t &stats, bool sass_guided,
                   const sass_function_guide_t *sass_guide,
                   unsigned guide_lookahead, unsigned *sass_guide_cursor,
                   const ptxline_guide_t *sass_ptxline_guide) {
  if (segment.empty())
    return;

  stats.total_insts += segment.size();
  stats.max_segment =
      std::max(stats.max_segment, static_cast<unsigned>(segment.size()));

  if (segment.size() == 1) {
    out.push_back(segment[0].inst);
    segment.clear();
    return;
  }

  unsigned edge_count = 0;
  bool valid = true;
  std::vector<sched_inst_t> scheduled;
  std::vector<ptxline_guide_item_t> segment_ptxline_guide;
  if (sass_ptxline_guide != NULL)
    segment_ptxline_guide =
        filter_ptxline_guide_for_segment(segment, *sass_ptxline_guide);
  const bool use_sass_ptxline_guide = !segment_ptxline_guide.empty();
  const bool use_sass_guide =
      sass_guided && sass_guide != NULL && !sass_guide->lt_stream.empty() &&
      !sass_guide->guide_items.empty() && sass_guide_cursor != NULL &&
      !use_sass_ptxline_guide &&
      (contains_class(segment, inst_class_t::tensor) ||
       contains_class(segment, inst_class_t::ldmatrix));
  const bool has_relaxed_barrier = contains_barrier_inst(segment);
  if (use_sass_ptxline_guide) {
    scheduled = schedule_sass_ptxline_guided(segment, segment_ptxline_guide,
                                             &edge_count, &valid);
    if (!valid) {
      ptx_reorder_fatal(
          "SASS PTX-line guide failed to schedule a segment with %zu "
          "instructions and %zu guide anchors",
          segment.size(), segment_ptxline_guide.size());
    }
    ++stats.sass_guided_segments;
    stats.sass_guide_cursor += segment_ptxline_guide.size();
  } else if (use_sass_guide) {
    scheduled = schedule_sass_guided(segment, *sass_guide, guide_lookahead,
                                     sass_guide_cursor, &edge_count, &valid);
    if (valid)
      ++stats.sass_guided_segments;
    else
      ++stats.sass_guided_fallback_segments;
  }
  if ((!use_sass_ptxline_guide && (!use_sass_guide || !valid)) &&
      has_relaxed_barrier) {
    scheduled = segment;
    valid = false;
    edge_count = 0;
  } else if (!use_sass_ptxline_guide && (!use_sass_guide || !valid)) {
    bool switch_valid = true;
    scheduled =
        schedule_switch(segment, ready_slack, &edge_count, &switch_valid);
    valid = switch_valid;
  }
  stats.edges += edge_count;
  ++stats.segments;

  if (!valid) {
    ++stats.skipped_segments;
    scheduled = segment;
  }

  for (unsigned i = 0; i < scheduled.size(); ++i) {
    if (scheduled[i].inst != segment[i].inst)
      ++stats.moved_slots;
    out.push_back(scheduled[i].inst);
  }
  segment.clear();
}

} // namespace

namespace flash_gpgpu_sim {

void run_ptx_reorder(function_info *func) {
  if (func == NULL || func->gpgpu_ctx == NULL ||
      !func->gpgpu_ctx->ptx_reorder_enabled)
    return;

  std::list<ptx_instruction *> reordered;
  std::vector<sched_inst_t> segment;
  std::map<const ptx_instruction *, unsigned> original_indices;
  reorder_stats_t stats;
  unsigned original_index = 0;
  std::string sass_guide_stream;
  const sass_function_guide_t *sass_guide = NULL;
  ptxline_guide_t sass_ptxline_guide_storage;
  const ptxline_guide_t *sass_ptxline_guide = NULL;
  unsigned sass_guide_cursor = 0;
  const bool have_sass_ptxline_file =
      regular_file_exists(k_sass_ptxline_guide_file);
  const bool sass_ptxline_guided =
      func->gpgpu_ctx->ptx_reorder_sass_guided && have_sass_ptxline_file;
  const bool sass_guided = !sass_ptxline_guided &&
                           func->gpgpu_ctx->ptx_reorder_sass_guided &&
                           func->gpgpu_ctx->ptx_reorder_sass_file != NULL &&
                           func->gpgpu_ctx->ptx_reorder_sass_file[0] != '\0';

  if (func->gpgpu_ctx->ptx_reorder_sass_guided && !sass_ptxline_guided &&
      !sass_guided) {
    ptx_reorder_fatal(
        "SASS-guided reorder is enabled but neither cwd guide '%s' nor legacy "
        "-gpgpu_ptx_reorder_sass_file is available",
        k_sass_ptxline_guide_file);
  }

  if (sass_ptxline_guided) {
    const std::string primary_rules_file =
        find_unique_sass_primary_rules_file();
    const sass_primary_rules_t &rules =
        get_sass_primary_rules(primary_rules_file.c_str());
    if (!rules.ok) {
      ptx_reorder_fatal("failed to parse primary SASS hint rules '%s': %s",
                        primary_rules_file.c_str(), rules.error.c_str());
    }
    const sass_ptxline_file_t &guides =
        get_sass_ptxline_file(k_sass_ptxline_guide_file);
    if (!guides.ok) {
      ptx_reorder_fatal("failed to parse SASS PTX-line guide '%s': %s",
                        k_sass_ptxline_guide_file, guides.error.c_str());
    }
    const sass_ptxline_function_t *sass_func =
        select_sass_ptxline_function(guides, func->m_name);
    if (sass_func != NULL) {
      sass_ptxline_guide_storage = build_sass_ptxline_guide(
          func->m_name, func->m_instructions, *sass_func, rules);
      sass_ptxline_guide = &sass_ptxline_guide_storage;
      stats.sass_guide_source = sass_ptxline_guide->source;
      const unsigned guide_head_limit = std::min<unsigned>(
          32, static_cast<unsigned>(sass_ptxline_guide->items.size()));
      stats.sass_guide_head =
          build_ptxline_guide_head(*sass_ptxline_guide, guide_head_limit);
      if (func->gpgpu_ctx->ptx_reorder_stats) {
        printf("GPGPU-Sim PTX: SASS PTX-line guide function '%s': "
               "anchors=%zu matched_lines=%u source=%s\n",
               func->m_name.c_str(), sass_ptxline_guide->items.size(),
               sass_ptxline_guide->matched_ptx_lines,
               sass_ptxline_guide->source.c_str());
      }
    }
  } else if (sass_guided) {
    unsigned ptx_tensor_count = 0;
    unsigned ptx_ldmatrix_count = 0;
    count_lt_classes(func->m_instructions, &ptx_tensor_count,
                     &ptx_ldmatrix_count);
    const sass_file_guides_t &guides =
        get_sass_guides(func->gpgpu_ctx->ptx_reorder_sass_file);
    sass_guide = select_sass_guide(guides, func->m_name, ptx_tensor_count,
                                   ptx_ldmatrix_count);
    if (sass_guide != NULL)
      sass_guide_stream = sass_guide->lt_stream;
    stats.sass_guide_source =
        sass_guide != NULL ? sass_guide->name : std::string();
    stats.sass_guide_head = sass_guide_stream.substr(
        0, std::min<std::size_t>(128, sass_guide_stream.size()));
    if (func->gpgpu_ctx->ptx_reorder_stats) {
      printf("GPGPU-Sim PTX: SASS guide function '%s': ptx_T=%u ptx_L=%u "
             "guide_len=%zu source=%s\n",
             func->m_name.c_str(), ptx_tensor_count, ptx_ldmatrix_count,
             sass_guide_stream.size(),
             sass_guide != NULL ? sass_guide->name.c_str() : "<none>");
    }
  }

  for (std::list<ptx_instruction *>::iterator it = func->m_instructions.begin();
       it != func->m_instructions.end(); ++it, ++original_index) {
    ptx_instruction *inst = *it;
    original_indices[inst] = original_index;
    const bool guide_relaxed_ldmatrix =
        ((sass_guided && sass_guide != NULL && !sass_guide_stream.empty()) ||
         sass_ptxline_guide != NULL) &&
        inst != NULL && inst->get_opcode() == LDMATRIX_OP;
    if (is_segment_boundary(inst, guide_relaxed_ldmatrix)) {
      flush_segment(
          segment, reordered, func->gpgpu_ctx->ptx_reorder_ready_slack, stats,
          sass_guided, sass_guide,
          std::max(1, func->gpgpu_ctx->ptx_reorder_sass_guide_lookahead),
          &sass_guide_cursor, sass_ptxline_guide);
      reordered.push_back(inst);
      continue;
    }

    sched_inst_t sched_inst;
    sched_inst.inst = inst;
    sched_inst.original_index = original_index;
    sched_inst.cls = classify_inst(inst);
    collect_inst_regs(inst, sched_inst.uses, sched_inst.defs);
    segment.push_back(sched_inst);
  }
  flush_segment(segment, reordered, func->gpgpu_ctx->ptx_reorder_ready_slack,
                stats, sass_guided, sass_guide,
                std::max(1, func->gpgpu_ctx->ptx_reorder_sass_guide_lookahead),
                &sass_guide_cursor, sass_ptxline_guide);
  if (sass_ptxline_guide == NULL)
    stats.sass_guide_cursor = sass_guide_cursor;

  if (reordered.size() != func->m_instructions.size()) {
    printf("GPGPU-Sim PTX: reorder switch skipped function '%s' due to size "
           "mismatch (%zu vs %zu)\n",
           func->m_name.c_str(), reordered.size(), func->m_instructions.size());
    return;
  }

  func->m_instructions.swap(reordered);

  if (func->gpgpu_ctx->ptx_reorder_dump) {
    dump_ptx_reorder_result(func->m_name, func->m_instructions,
                            original_indices, stats,
                            func->gpgpu_ctx->ptx_reorder_ready_slack,
                            func->gpgpu_ctx->ptx_reorder_dump_dir);
  }

  if (func->gpgpu_ctx->ptx_reorder_stats || stats.moved_slots != 0) {
    const std::size_t guide_total = sass_ptxline_guide != NULL
                                        ? sass_ptxline_guide->items.size()
                                        : sass_guide_stream.size();
    printf("GPGPU-Sim PTX: reorder switch function '%s': segments=%u "
           "skipped=%u insts=%u moved_slots=%u max_segment=%u edges=%u "
           "slack=%d sass_guided=%u sass_guided_fallback=%u "
           "sass_cursor=%u/%zu\n",
           func->m_name.c_str(), stats.segments, stats.skipped_segments,
           stats.total_insts, stats.moved_slots, stats.max_segment, stats.edges,
           func->gpgpu_ctx->ptx_reorder_ready_slack, stats.sass_guided_segments,
           stats.sass_guided_fallback_segments, stats.sass_guide_cursor,
           guide_total);
  }
}

} // namespace flash_gpgpu_sim
