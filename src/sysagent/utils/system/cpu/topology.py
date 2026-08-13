# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
CPU topology utilities — hybrid core-type detection and CPU-list helpers.

Part of the ``sysagent.utils.system.cpu`` sub-package.

Detects P-core / E-core / LP E-core topology on hybrid Intel CPUs using a
three-level strategy:

1. ``core_type`` sysfs file (Linux ≥ 6.3) — authoritative kernel source.
2. ``lscpu -e=CPU,TYPE,CORE`` — available on most distributions.
3. L2 + L3 cache-sharing heuristic — reliable fallback on any Linux kernel:
   - Private L2  → P-core
   - Cluster L2 + L3 present → E-core
   - Cluster L2 + no L3  → LP E-core (Meteor Lake / Panther Lake SoC tile)
"""

import logging
import os

import psutil

from sysagent.utils.core.process import run_command

logger = logging.getLogger(__name__)


# ─── CPU list parsing helpers ──────────────────────────────────────────────────


def parse_cpu_list(s: str) -> list[int]:
    """Parse a Linux CPU-list string (e.g. ``"0-3,8,16-19"``) into a list of ints."""
    result: list[int] = []
    for part in s.strip().split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo_s, hi_s = part.split("-", 1)
            try:
                result.extend(range(int(lo_s), int(hi_s) + 1))
            except ValueError:
                pass
        else:
            try:
                result.append(int(part))
            except ValueError:
                pass
    return result


def compact_cpu_list(cpus: list[int]) -> str:
    """Convert a sorted list of logical CPU indices to a compact range string.

    e.g. ``[0, 1, 2, 3, 8, 16, 17]`` → ``"0-3,8,16-17"``
    """
    if not cpus:
        return ""
    cpus = sorted(set(cpus))
    ranges: list[str] = []
    start = end = cpus[0]
    for c in cpus[1:]:
        if c == end + 1:
            end = c
        else:
            ranges.append(f"{start}-{end}" if start != end else str(start))
            start = end = c
    ranges.append(f"{start}-{end}" if start != end else str(start))
    return ",".join(ranges)


# ─── CPU cache hierarchy reader ──────────────────────────────────────────────


def _read_cpu_cache_info(cpu_id: int, cpu_base: str) -> dict:
    """Read the cache hierarchy for one logical CPU from sysfs ``cache/index*`` dirs.

    Returns a dict keyed by ``"l1d"``, ``"l1i"``, ``"l2"``, ``"l3"`` (and
    ``"l4"`` when present) with the following per-level fields:

    * ``size_kb``          total size in KiB
    * ``ways``             set-associativity
    * ``sets``             number of sets
    * ``line_size_bytes``  coherency line size in bytes
    * ``shared_cpus``      number of logical CPUs sharing this cache instance
                           (1 = private; >1 = shared, e.g. 4 for E-core cluster L2)

    An empty dict is returned when sysfs is unavailable (containers, non-Linux).
    """
    cache_dir = os.path.join(cpu_base, f"cpu{cpu_id}", "cache")
    result: dict = {}

    if not os.path.isdir(cache_dir):
        return result

    def _rfile(path: str) -> str:
        try:
            with open(path) as fh:
                return fh.read().strip()
        except Exception:
            return ""

    def _parse_kb(s: str) -> int | None:
        """Parse a Linux kernel size string ('48K', '3072K', '36M') to KiB."""
        s = s.strip()
        if s.endswith("K"):
            try:
                return int(s[:-1])
            except ValueError:
                pass
        elif s.endswith("M"):
            try:
                return int(float(s[:-1]) * 1024)
            except ValueError:
                pass
        else:
            try:
                return max(1, int(s) // 1024)  # assume bytes, convert to KiB
            except ValueError:
                pass
        return None

    try:
        indexes = sorted(
            [d for d in os.listdir(cache_dir) if d.startswith("index") and d[5:].isdigit()],
            key=lambda d: int(d[5:]),
        )
    except Exception:
        return result

    for idx_name in indexes:
        idx_path = os.path.join(cache_dir, idx_name)
        level_s = _rfile(os.path.join(idx_path, "level"))
        cache_type = _rfile(os.path.join(idx_path, "type")).lower()
        size_s = _rfile(os.path.join(idx_path, "size"))
        ways_s = _rfile(os.path.join(idx_path, "ways_of_associativity"))
        sets_s = _rfile(os.path.join(idx_path, "number_of_sets"))
        line_s = _rfile(os.path.join(idx_path, "coherency_line_size"))
        shared_s = _rfile(os.path.join(idx_path, "shared_cpu_list"))

        try:
            level = int(level_s)
        except (ValueError, TypeError):
            continue

        # Map level + type to a human-readable key
        if level == 1 and cache_type == "data":
            key = "l1d"
        elif level == 1 and cache_type == "instruction":
            key = "l1i"
        elif level == 1:  # unified L1 (some Atom/LP-E cores)
            key = "l1"
        else:
            key = f"l{level}"  # "l2", "l3", "l4"

        entry: dict = {}
        size_kb = _parse_kb(size_s)
        if size_kb is not None:
            entry["size_kb"] = size_kb
        for field, raw in (("ways", ways_s), ("sets", sets_s), ("line_size_bytes", line_s)):
            try:
                entry[field] = int(raw)
            except (ValueError, TypeError):
                pass
        if shared_s:
            entry["shared_cpus"] = len(parse_cpu_list(shared_s))

        if entry:
            result[key] = entry

    return result


# ─── CPU cache hierarchy reader ─────────────────────────────────────────────────


def _read_cpu_cache_info(cpu_id: int, cpu_base: str) -> dict:
    """Read the cache hierarchy for one logical CPU from sysfs ``cache/index*`` dirs.

    Each entry in the returned dict represents one cache level, keyed by
    ``"l1d"``, ``"l1i"``, ``"l2"``, ``"l3"`` (and ``"l4"`` when present).
    Fields within each level:

    * ``size_kb``          total capacity of this cache instance, in KiB
    * ``ways``             set-associativity (ways per set)
    * ``sets``             number of cache sets
    * ``line_size_bytes``  coherency line size in bytes (typically 64)
    * ``shared_cpus``      number of logical CPUs sharing this instance
                           (1 = private to this logical CPU / core;
                            4 = 4-core E-core cluster L2;
                           24 = shared across all cores, as with L3)
    * ``shared_cpu_list``  compact CPU-list string of sharing CPUs
                           (e.g. ``"0"`` for private, ``"8-11"`` for cluster)

    An empty dict is returned when sysfs is unavailable (containers, non-Linux).
    """
    cache_dir = os.path.join(cpu_base, f"cpu{cpu_id}", "cache")
    result: dict = {}

    if not os.path.isdir(cache_dir):
        return result

    def _rfile(path: str) -> str:
        try:
            with open(path) as fh:
                return fh.read().strip()
        except Exception:
            return ""

    def _parse_kb(s: str) -> int | None:
        """Parse a Linux kernel size string (``'48K'``, ``'3072K'``, ``'36M'``) to KiB."""
        s = s.strip()
        if s.endswith("K"):
            try:
                return int(s[:-1])
            except ValueError:
                pass
        elif s.endswith("M"):
            try:
                return int(float(s[:-1]) * 1024)
            except ValueError:
                pass
        else:
            try:
                return max(1, int(s) // 1024)  # assume bytes, convert to KiB
            except ValueError:
                pass
        return None

    try:
        indexes = sorted(
            (d for d in os.listdir(cache_dir) if d.startswith("index") and d[5:].isdigit()),
            key=lambda d: int(d[5:]),
        )
    except Exception:
        return result

    for idx_name in indexes:
        idx_path = os.path.join(cache_dir, idx_name)
        level_s = _rfile(os.path.join(idx_path, "level"))
        cache_type = _rfile(os.path.join(idx_path, "type")).lower()
        size_s = _rfile(os.path.join(idx_path, "size"))
        ways_s = _rfile(os.path.join(idx_path, "ways_of_associativity"))
        sets_s = _rfile(os.path.join(idx_path, "number_of_sets"))
        line_s = _rfile(os.path.join(idx_path, "coherency_line_size"))
        shared_s = _rfile(os.path.join(idx_path, "shared_cpu_list"))

        try:
            level = int(level_s)
        except (ValueError, TypeError):
            continue

        # Map (level, type) to a canonical key
        if level == 1 and "data" in cache_type:
            key = "l1d"
        elif level == 1 and "instruction" in cache_type:
            key = "l1i"
        elif level == 1:  # unified L1 (some LP-E / Atom cores)
            key = "l1"
        else:
            key = f"l{level}"  # "l2", "l3", "l4"

        entry: dict = {}
        size_kb = _parse_kb(size_s)
        if size_kb is not None:
            entry["size_kb"] = size_kb
        for field, raw in (("ways", ways_s), ("sets", sets_s), ("line_size_bytes", line_s)):
            try:
                entry[field] = int(raw)
            except (ValueError, TypeError):
                pass
        if shared_s:
            entry["shared_cpus"] = len(parse_cpu_list(shared_s))
            entry["shared_cpu_list"] = shared_s

        if entry:
            result[key] = entry

    return result


# ─── CPU core-type topology ────────────────────────────────────────────────────


def collect_cpu_core_types() -> dict:
    """Detect CPU core types (P-core, E-core, LP E-core) for hybrid Intel CPUs.

    Reads from sysfs topology files (Linux 6.3+ for ``core_type``):

    * ``/sys/devices/system/cpu/cpuN/topology/core_type``
    * ``/sys/devices/system/cpu/cpuN/topology/core_id``
    * ``/sys/devices/system/cpu/cpuN/topology/core_cpus_list``  (HT siblings)

    Falls back to ``lscpu -e=CPU,TYPE,CORE`` when ``core_type`` sysfs files are
    absent (older kernels / non-hybrid platforms).  As a final fallback, uses an
    L2+L3 cache-sharing heuristic: CPUs with private L2 are P-cores; CPUs with
    cluster-shared L2 and L3 present are E-cores; CPUs with cluster-shared L2 but
    **no L3** (e.g. Meteor Lake / Panther Lake SoC tile) are LP E-cores.

    Returns
    -------
    dict
        ``hybrid``  — ``True`` when more than one core type is present.
        ``groups``  — list of per-type dicts, one entry per distinct core class.
                      Scales to uniform many-core CPUs (e.g. Xeon with 200+ cores)
                      as one group, using ``cpu_list`` compact notation.

        Each group contains:

        * ``name``                canonical label: ``"p-core"`` | ``"e-core"`` | ``"lpe-core"``
        * ``core_count``          number of physical cores of this type
        * ``thread_count``        number of logical CPUs of this type
        * ``threads_per_core``    1 = no SMT/HT, 2 = SMT enabled for this type
        * ``hyperthreading``      bool — SMT/HT enabled for this type
        * ``has_l3``              ``True`` when CPUs of this type have an L3 cache level
        * ``cpu_list``            compact range string of **logical** CPU numbers
                                  (use with ``taskset``, ``numactl``, etc.)
        * ``physical_core_ids``   sorted list of **APIC physical core IDs** as reported by
                                  ``/sys/devices/system/cpu/cpuN/topology/core_id``.
                                  These are hardware topology identifiers, **not** the same
                                  as logical CPU numbers.  Intel P-cores use non-sequential
                                  APIC IDs (e.g. 0, 4, 16, …) because each P-core
                                  reserves 4 APIC slots even when HT is disabled.
        * ``cache``               dict of per-level cache details (see ``_read_cpu_cache_info``):
                                  ``l1d``, ``l1i``, ``l2``, ``l3`` — each with
                                  ``size_kb``, ``ways``, ``sets``,
                                  ``line_size_bytes``, ``shared_cpus``
        * ``inferred``            ``True`` when type was determined by heuristic
    """
    CPU_BASE = "/sys/devices/system/cpu"
    result: dict = {"hybrid": False, "groups": []}

    # Canonical label map  (sysfs string → short label used in RT tooling)
    _LABEL_MAP = {
        "Intel Core": "p-core",
        "Intel Atom": "e-core",
        "Intel Atom LP": "lpe-core",
    }

    def _sysfs_read(path: str) -> str:
        try:
            with open(path) as fh:
                return fh.read().strip()
        except Exception:
            return ""

    # Discover all logical CPU indices from sysfs
    try:
        cpu_ids = sorted(int(d[3:]) for d in os.listdir(CPU_BASE) if d.startswith("cpu") and d[3:].isdigit())
    except Exception:
        cpu_ids = list(range(psutil.cpu_count(logical=True) or 0))

    # Per-CPU record: core_type, physical core_id, HT-sibling list
    records: dict[int, dict] = {}
    for cid in cpu_ids:
        topo = os.path.join(CPU_BASE, f"cpu{cid}", "topology")
        core_type_str = _sysfs_read(os.path.join(topo, "core_type"))  # kernel ≥ 6.3
        core_id_raw = _sysfs_read(os.path.join(topo, "core_id"))
        # core_cpus_list preferred; thread_siblings_list is the older alias
        siblings_raw = _sysfs_read(os.path.join(topo, "core_cpus_list")) or _sysfs_read(
            os.path.join(topo, "thread_siblings_list")
        )
        try:
            core_id = int(core_id_raw) if core_id_raw else cid
        except ValueError:
            core_id = cid
        siblings = parse_cpu_list(siblings_raw) if siblings_raw else [cid]
        records[cid] = {
            "core_type": core_type_str or None,
            "core_id": core_id,
            "siblings": siblings,
        }

    # ── Fallback: lscpu -e when sysfs core_type is universally missing ─────────
    if all(not r["core_type"] for r in records.values()):
        try:
            proc = run_command(["lscpu", "-e=CPU,TYPE,CORE"], timeout=5)
            out = proc.stdout if proc.returncode == 0 else ""
            lines = out.strip().splitlines()
            if len(lines) > 1:
                header = lines[0].split()
                try:
                    cpu_col = header.index("CPU")
                    type_col = header.index("TYPE")
                    core_col = header.index("CORE") if "CORE" in header else -1
                except ValueError:
                    cpu_col = type_col = core_col = -1

                if cpu_col >= 0 and type_col >= 0:
                    for line in lines[1:]:
                        parts = line.split()
                        if len(parts) <= max(cpu_col, type_col):
                            continue
                        try:
                            cid = int(parts[cpu_col])
                        except ValueError:
                            continue
                        if cid in records:
                            records[cid]["core_type"] = parts[type_col]
                            if core_col >= 0 and core_col < len(parts):
                                try:
                                    records[cid]["core_id"] = int(parts[core_col])
                                except ValueError:
                                    pass
        except Exception as exc:
            logger.debug(f"lscpu TYPE fallback failed: {exc}")

    # ── Fallback: L2 + L3 cache-sharing heuristic ─────────────────────────────
    # On hybrid Intel CPUs (12th Gen+) P-cores have a private L2 (shared only
    # with their HT sibling, if any), while E-cores share one L2 per cluster of
    # four.  Rule: if len(L2 sharers) > len(HT siblings) → cluster-shared L2.
    # LP E-cores (Meteor Lake / Panther Lake SoC tile) additionally have no L3
    # cache at all; regular E-cores share L3 with the P-cores.
    if all(not r["core_type"] for r in records.values()):
        l2_shared: dict[int, list[int]] = {}
        for cid in cpu_ids:
            shared_path = os.path.join(CPU_BASE, f"cpu{cid}", "cache", "index2", "shared_cpu_list")
            raw = _sysfs_read(shared_path)
            l2_shared[cid] = parse_cpu_list(raw) if raw else [cid]

        # Only meaningful when at least some CPUs share L2 (hybrid signature)
        has_cluster_l2 = any(len(l2_shared[cid]) > len(records[cid]["siblings"]) for cid in cpu_ids if cid in records)

        if has_cluster_l2:
            for cid in cpu_ids:
                if cid not in records:
                    continue
                if len(l2_shared[cid]) > len(records[cid]["siblings"]):
                    # Cluster-shared L2: E-core or LP E-core.
                    # LP E-cores (e.g. Meteor Lake / Panther Lake SoC tile) lack an
                    # L3 cache entirely; regular E-cores share L3 with the P-cores.
                    l3_index = os.path.join(CPU_BASE, f"cpu{cid}", "cache", "index3")
                    if os.path.isdir(l3_index):
                        records[cid]["core_type"] = "Intel Atom"  # E-core: has L3
                    else:
                        records[cid]["core_type"] = "Intel Atom LP"  # LP E-core: no L3
                else:
                    records[cid]["core_type"] = "Intel Core"  # P-core (private L2)
                records[cid]["inferred"] = True
            logger.debug("collect_cpu_core_types: used L2+L3-sharing heuristic for hybrid detection")

    # ── Normalise missing types ────────────────────────────────────────────────
    for r in records.values():
        if not r["core_type"]:
            r["core_type"] = "Unknown"

    types_seen = {r["core_type"] for r in records.values() if r["core_type"] != "Unknown"}
    result["hybrid"] = len(types_seen) > 1

    # ── L3 cache presence per logical CPU ─────────────────────────────────────
    # Checked independently of type detection so has_l3 is accurate regardless
    # of which detection path (sysfs / lscpu / heuristic) was used.  Defaults
    # True when sysfs is inaccessible (containers, non-Linux, etc.) to avoid
    # false LP-E classification.
    cpu_has_l3: dict[int, bool] = {
        cid: os.path.isdir(os.path.join(CPU_BASE, f"cpu{cid}", "cache", "index3")) for cid in cpu_ids
    }
    if not any(cpu_has_l3.values()):  # full sysfs unavailable → assume L3 present
        cpu_has_l3 = {cid: True for cid in cpu_ids}

    # ── Group logical CPUs by core type ───────────────────────────────────────
    groups: dict[str, dict] = {}
    for cid, rec in records.items():
        ct = rec["core_type"]
        if ct not in groups:
            groups[ct] = {"cpu_ids": [], "core_ids": set(), "siblings_map": {}, "any_has_l3": False}
        grp = groups[ct]
        grp["cpu_ids"].append(cid)
        grp["core_ids"].add(rec["core_id"])
        if cpu_has_l3.get(cid, True):
            grp["any_has_l3"] = True
        phys = rec["core_id"]
        if phys not in grp["siblings_map"]:
            grp["siblings_map"][phys] = set(rec["siblings"])

    _ORDER = {"p-core": 0, "e-core": 1, "lpe-core": 2}

    for type_name, grp in groups.items():
        threads_per_core = max((len(sibs) for sibs in grp["siblings_map"].values()), default=1)
        inferred = any(records[cid].get("inferred") for cid in grp["cpu_ids"])
        representative_cpu = sorted(grp["cpu_ids"])[0]
        result["groups"].append(
            {
                "name": _LABEL_MAP.get(type_name, type_name.lower().replace(" ", "-")),
                "core_count": len(grp["core_ids"]),
                "thread_count": len(grp["cpu_ids"]),
                "threads_per_core": threads_per_core,
                "hyperthreading": threads_per_core > 1,
                "has_l3": grp["any_has_l3"],
                "cpu_list": compact_cpu_list(sorted(grp["cpu_ids"])),
                "physical_core_ids": sorted(grp["core_ids"]),
                "cache": _read_cpu_cache_info(representative_cpu, CPU_BASE),
                "inferred": inferred,
            }
        )

    result["groups"].sort(key=lambda t: _ORDER.get(t["name"], 99))
    return result
