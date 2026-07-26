import logging
from dataclasses import dataclass, field
from app.observability.metrics import mcp_replica_health

logger = logging.getLogger("inferroute")


@dataclass
class MCPReplica:
    replica_id: str
    tool_name: str
    endpoint: str
    healthy: bool = True
    in_flight: int = 0


class MCPRegistry:

    def __init__(self):
        self._replicas: dict[str, dict[str, MCPReplica]] = {}
        self._rr_index: dict[str, int] = {}

    def register(self, tool_name: str, replica_id: str, endpoint: str):
        if tool_name not in self._replicas:
            self._replicas[tool_name] = {}
            self._rr_index[tool_name] = 0

        replica = MCPReplica(replica_id=replica_id, tool_name=tool_name, endpoint=endpoint)
        self._replicas[tool_name][replica_id] = replica
        mcp_replica_health.labels(tool_name=tool_name, replica_id=replica_id).set(1)
        logger.info("MCP replica registered: tool=%s replica=%s", tool_name, replica_id)

    def deregister(self, tool_name: str, replica_id: str):
        if tool_name in self._replicas and replica_id in self._replicas[tool_name]:
            del self._replicas[tool_name][replica_id]
            logger.info("MCP replica deregistered: tool=%s replica=%s", tool_name, replica_id)

    def get_healthy(self, tool_name: str) -> list[MCPReplica]:
        if tool_name not in self._replicas:
            return []
        return [r for r in self._replicas[tool_name].values() if r.healthy]

    def select(self, tool_name: str) -> MCPReplica | None:
        healthy = self.get_healthy(tool_name)
        if not healthy:
            return None

        idx = self._rr_index.get(tool_name, 0) % len(healthy)
        self._rr_index[tool_name] = idx + 1
        return healthy[idx]

    def mark_unhealthy(self, tool_name: str, replica_id: str):
        if tool_name in self._replicas and replica_id in self._replicas[tool_name]:
            self._replicas[tool_name][replica_id].healthy = False
            mcp_replica_health.labels(tool_name=tool_name, replica_id=replica_id).set(0)
            logger.warning("MCP replica marked unhealthy: tool=%s replica=%s", tool_name, replica_id)

    def mark_healthy(self, tool_name: str, replica_id: str):
        if tool_name in self._replicas and replica_id in self._replicas[tool_name]:
            self._replicas[tool_name][replica_id].healthy = True
            mcp_replica_health.labels(tool_name=tool_name, replica_id=replica_id).set(1)
            logger.info("MCP replica marked healthy: tool=%s replica=%s", tool_name, replica_id)

    def increment_in_flight(self, tool_name: str, replica_id: str):
        if tool_name in self._replicas and replica_id in self._replicas[tool_name]:
            self._replicas[tool_name][replica_id].in_flight += 1

    def decrement_in_flight(self, tool_name: str, replica_id: str):
        if tool_name in self._replicas and replica_id in self._replicas[tool_name]:
            self._replicas[tool_name][replica_id].in_flight = max(0, self._replicas[tool_name][replica_id].in_flight - 1)

    def get_status(self) -> list[dict]:
        status = []
        for tool_name, replicas in self._replicas.items():
            for replica_id, replica in replicas.items():
                status.append({
                    "tool_name": tool_name,
                    "replica_id": replica_id,
                    "endpoint": replica.endpoint,
                    "healthy": replica.healthy,
                    "in_flight": replica.in_flight,
                })
        return status

    def list_tools(self) -> list[str]:
        return list(self._replicas.keys())
