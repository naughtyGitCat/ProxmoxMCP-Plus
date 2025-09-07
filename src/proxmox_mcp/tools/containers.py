from typing import List, Dict, Optional, Tuple, Any, Union
import json
from mcp.types import TextContent as Content
from .base import ProxmoxTool


def _b2h(n: Union[int, float, str]) -> str:
    """bytes -> human (binary units)."""
    try:
        n = float(n)
    except Exception:
        return "0.00 B"
    units = ("B", "KiB", "MiB", "GiB", "TiB", "PiB")
    i = 0
    while n >= 1024.0 and i < len(units) - 1:
        n /= 1024.0
        i += 1
    return f"{n:.2f} {units[i]}"


def _as_dict(maybe: Any) -> Dict:
    """Return dict; unwrap {'data': dict}; else {}."""
    if isinstance(maybe, dict):
        data = maybe.get("data")
        if isinstance(data, dict):
            return data
        return maybe
    return {}


def _as_list(maybe: Any) -> List:
    """Return list; unwrap {'data': list}; else []."""
    if isinstance(maybe, list):
        return maybe
    if isinstance(maybe, dict):
        data = maybe.get("data")
        if isinstance(data, list):
            return data
    return []


def _get(d: Any, key: str, default: Any = None) -> Any:
    """Safe dict get."""
    if isinstance(d, dict):
        return d.get(key, default)
    return default


class ContainerTools(ProxmoxTool):
    """
    LXC container tools for Proxmox MCP.

    - Lists containers cluster-wide (or by node)
    - Live stats via /status/current
    - Limit fallback via /config (memory MiB, cores/cpulimit)
    - RRD fallback when live returns zeros
    - Pretty output rendered here; JSON path is raw & sanitized
    """

    # ---------- error / output ----------
    def _json_fmt(self, data: Any) -> List[Content]:
        """Return raw JSON string (never touch project formatters)."""
        return [Content(type="text", text=json.dumps(data, indent=2, sort_keys=True))]

    def _err(self, action: str, e: Exception) -> List[Content]:
        if hasattr(self, "handle_error"):
            return self.handle_error(e, action)  # type: ignore[attr-defined]
        if hasattr(self, "_handle_error"):
            return self._handle_error(action, e)  # type: ignore[attr-defined]
        return [Content(type="text", text=json.dumps({"error": str(e), "action": action}))]

    # ---------- helpers ----------
    def _list_ct_pairs(self, node: Optional[str]) -> List[Tuple[str, Dict]]:
        """Yield (node_name, ct_dict). Coerce odd shapes into dicts with vmid."""
        out: List[Tuple[str, Dict]] = []
        if node:
            raw = self.proxmox.nodes(node).lxc.get()
            for it in _as_list(raw):
                if isinstance(it, dict):
                    out.append((node, it))
                else:
                    try:
                        vmid = int(it)
                        out.append((node, {"vmid": vmid}))
                    except Exception:
                        continue
        else:
            nodes = _as_list(self.proxmox.nodes.get())
            for n in nodes:
                nname = _get(n, "node")
                if not nname:
                    continue
                raw = self.proxmox.nodes(nname).lxc.get()
                for it in _as_list(raw):
                    if isinstance(it, dict):
                        out.append((nname, it))
                    else:
                        try:
                            vmid = int(it)
                            out.append((nname, {"vmid": vmid}))
                        except Exception:
                            continue
        return out

    def _rrd_last(self, node: str, vmid: int) -> Tuple[Optional[float], Optional[int], Optional[int]]:
        """Return (cpu_pct, mem_bytes, maxmem_bytes) from the most recent RRD sample."""
        try:
            rrd = _as_list(self.proxmox.nodes(node).lxc(vmid).rrddata.get(timeframe="hour", ds="cpu,mem,maxmem"))
            if not rrd or not isinstance(rrd[-1], dict):
                return None, None, None
            last = rrd[-1]
            cpu_frac = float(_get(last, "cpu", 0.0) or 0.0)
            mem_b = int(_get(last, "mem", 0) or 0)
            maxmem_b = int(_get(last, "maxmem", 0) or 0)
            return round(cpu_frac * 100.0, 2), mem_b, maxmem_b
        except Exception:
            return None, None, None

    def _status_and_config(self, node: str, vmid: int) -> Tuple[Dict, Dict]:
        """Return (raw_status, raw_config) as dicts (unwrapped), or {}."""
        try:
            raw_status = _as_dict(self.proxmox.nodes(node).lxc(vmid).status.current.get())
        except Exception:
            raw_status = {}
        try:
            raw_config = _as_dict(self.proxmox.nodes(node).lxc(vmid).config.get())
        except Exception:
            raw_config = {}
        return raw_status, raw_config

    def _render_pretty(self, rows: List[Dict]) -> List[Content]:
        lines: List[str] = ["📦 Containers", ""]
        for r in rows:
            name = r.get("name") or f"ct-{r.get('vmid')}"
            vmid = r.get("vmid")
            status = (r.get("status") or "").upper()
            node = r.get("node") or "?"
            cores = r.get("cores")
            cpu_pct = r.get("cpu_pct", 0.0)
            mem_bytes = int(r.get("mem_bytes") or 0)
            maxmem_bytes = int(r.get("maxmem_bytes") or 0)
            mem_pct = r.get("mem_pct")
            unlimited = bool(r.get("unlimited_memory", False))

            lines.append(f"📦 {name} (ID: {vmid})")
            lines.append(f"  • Status: {status}")
            lines.append(f"  • Node: {node}")
            lines.append(f"  • CPU: {cpu_pct:.1f}%")
            lines.append(f"  • CPU Cores: {cores if cores is not None else 'N/A'}")

            if unlimited:
                lines.append(f"  • Memory: {_b2h(mem_bytes)} (unlimited)")
            else:
                if maxmem_bytes > 0:
                    pct_str = f" ({mem_pct:.1f}%)" if isinstance(mem_pct, (int, float)) else ""
                    lines.append(f"  • Memory: {_b2h(mem_bytes)} / {_b2h(maxmem_bytes)}{pct_str}")
                else:
                    lines.append(f"  • Memory: {_b2h(mem_bytes)} / 0.00 B")
            lines.append("")
        return [Content(type="text", text="\n".join(lines).rstrip())]

    # ---------- tool ----------
    def get_containers(
        self,
        node: Optional[str] = None,
        include_stats: bool = True,
        include_raw: bool = False,
        format_style: str = "pretty"  # "pretty" or "json"
    ) -> List[Content]:
        """
        List LXC containers. If `node` is provided, filter to that node.

        include_stats: add cpu/mem live stats and limits
        include_raw:   (ignored for JSON output) include raw status/config
        format_style:  "pretty" (render here) or "json" (raw JSON list)
        """
        try:
            rows: List[Dict] = []

            for nname, ct in self._list_ct_pairs(node):
                vmid_val = _get(ct, "vmid")
                try:
                    vmid_int = int(vmid_val) if vmid_val is not None else None
                except Exception:
                    vmid_int = None

                rec: Dict = {
                    "vmid": str(vmid_val) if vmid_val is not None else None,
                    "name": _get(ct, "name") or _get(ct, "hostname") or (f"ct-{vmid_val}" if vmid_val is not None else "ct-?"),
                    "node": nname,
                    "status": _get(ct, "status"),
                }

                if include_stats and vmid_int is not None:
                    raw_status, raw_config = self._status_and_config(nname, vmid_int)

                    cpu_frac = float(_get(raw_status, "cpu", 0.0) or 0.0)
                    cpu_pct = round(cpu_frac * 100.0, 2)
                    mem_bytes = int(_get(raw_status, "mem", 0) or 0)
                    maxmem_bytes = int(_get(raw_status, "maxmem", 0) or 0)

                    memory_mib = int(_get(raw_config, "memory", 0) or 0)  # MiB
                    if maxmem_bytes == 0 and memory_mib > 0:
                        maxmem_bytes = memory_mib * 1024 * 1024

                    # cores / cpulimit
                    cores: Optional[Union[int, float]] = None
                    cfg_cores = _get(raw_config, "cores")
                    cfg_cpulimit = _get(raw_config, "cpulimit")
                    try:
                        if cfg_cores is not None and int(cfg_cores) > 0:
                            cores = int(cfg_cores)
                        elif cfg_cpulimit is not None and float(cfg_cpulimit) > 0:
                            cores = float(cfg_cpulimit)
                    except Exception:
                        cores = None

                    # RRD fallback if zeros
                    if (mem_bytes == 0) or (maxmem_bytes == 0) or (cpu_pct == 0.0):
                        rrd_cpu, rrd_mem, rrd_maxmem = self._rrd_last(nname, vmid_int)
                        if cpu_pct == 0.0 and rrd_cpu is not None:
                            cpu_pct = rrd_cpu
                        if mem_bytes == 0 and rrd_mem is not None:
                            mem_bytes = rrd_mem
                        if maxmem_bytes == 0 and rrd_maxmem:
                            maxmem_bytes = rrd_maxmem
                            if memory_mib == 0:
                                try:
                                    memory_mib = int(round(maxmem_bytes / (1024 * 1024)))
                                except Exception:
                                    memory_mib = 0

                    rec.update({
                        "cores": cores,
                        "memory": memory_mib,
                        "cpu_pct": cpu_pct,
                        "mem_bytes": mem_bytes,
                        "maxmem_bytes": maxmem_bytes,
                        "mem_pct": (
                            round((mem_bytes / maxmem_bytes * 100.0), 2)
                            if (maxmem_bytes and maxmem_bytes > 0)
                            else None
                        ),
                        "unlimited_memory": True if (maxmem_bytes == 0) else False,
                    })

                    # For PRETTY only: allow raw blobs to be attached if requested.
                    if include_raw and format_style != "json":
                        rec["raw_status"] = raw_status
                        rec["raw_config"] = raw_config

                rows.append(rec)

            if format_style == "json":
                # JSON path must be immune to any formatter assumptions; no raw payloads.
                return self._json_fmt(rows)
            return self._render_pretty(rows)

        except Exception as e:
            return self._err("Failed to list containers", e)
