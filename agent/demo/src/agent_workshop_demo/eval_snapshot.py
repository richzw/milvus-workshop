"""Named Milvus snapshot pinning for reproducible online evaluation."""

from __future__ import annotations

import re
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable

_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,63}$")


@dataclass(frozen=True)
class EvalSnapshotProvenance:
    """Non-sensitive snapshot identity attached to an eval report."""

    snapshot_name: str
    source_collection: str
    target_collection: str
    restore_job_id: int | None
    reused_target: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class MilvusEvalSnapshot:
    """Create/reuse one snapshot and restore it into a fixed eval target."""

    def __init__(
        self,
        client: Any,
        *,
        poll_interval: float = 0.25,
        monotonic: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        if poll_interval <= 0:
            raise ValueError("poll_interval must be positive")
        self.client = client
        self.poll_interval = poll_interval
        self.monotonic = monotonic
        self.sleeper = sleeper

    def pin(
        self,
        *,
        source_collection: str,
        snapshot_name: str,
        target_collection: str,
        timeout_seconds: float = 60.0,
    ) -> EvalSnapshotProvenance:
        """Return only after the immutable target collection can be loaded."""

        for value, label in (
            (source_collection, "source collection"),
            (snapshot_name, "snapshot"),
            (target_collection, "target collection"),
        ):
            if not _NAME.fullmatch(value):
                raise ValueError(f"Invalid {label} name")
        if source_collection == target_collection:
            raise ValueError("Snapshot target must differ from source collection")
        if timeout_seconds <= 0 or timeout_seconds > 600:
            raise ValueError("timeout_seconds must be in (0, 600]")
        if not self.client.has_collection(collection_name=source_collection):
            raise RuntimeError("Milvus eval source collection does not exist")

        snapshot = self._describe(snapshot_name, source_collection)
        if snapshot is None:
            self.client.flush(collection_name=source_collection)
            self.client.create_snapshot(
                snapshot_name=snapshot_name,
                collection_name=source_collection,
                description="agent-workshop frozen evaluation dataset",
            )
        elif _attribute(snapshot, "collection_name") != source_collection:
            raise RuntimeError("Snapshot source collection does not match")

        if self.client.has_collection(collection_name=target_collection):
            jobs = self.client.list_restore_snapshot_jobs(
                collection_name=target_collection
            )
            matching = [
                job
                for job in jobs
                if str(_attribute(job, "snapshot_name")) == snapshot_name
                and str(_attribute(job, "collection_name")) == target_collection
                and str(_attribute(job, "state")) == "RestoreSnapshotCompleted"
            ]
            if not matching:
                raise RuntimeError(
                    "Existing eval target is not a completed restore of the snapshot"
                )
            self.client.load_collection(collection_name=target_collection)
            return EvalSnapshotProvenance(
                snapshot_name,
                source_collection,
                target_collection,
                int(_attribute(matching[-1], "job_id")),
                True,
            )

        job_id = int(
            self.client.restore_snapshot(
                snapshot_name=snapshot_name,
                source_collection_name=source_collection,
                target_collection_name=target_collection,
            )
        )
        deadline = self.monotonic() + timeout_seconds
        while True:
            state = self.client.get_restore_snapshot_state(job_id=job_id)
            status = str(_attribute(state, "state"))
            if status == "RestoreSnapshotCompleted":
                break
            if status in {"RestoreSnapshotFailed", "RestoreSnapshotNone"}:
                raise RuntimeError("Milvus snapshot restore failed")
            if status not in {
                "RestoreSnapshotPending",
                "RestoreSnapshotExecuting",
            }:
                raise RuntimeError("Milvus snapshot restore returned an unknown state")
            if self.monotonic() >= deadline:
                raise TimeoutError("Milvus snapshot restore timed out")
            self.sleeper(self.poll_interval)
        if not self.client.has_collection(collection_name=target_collection):
            raise RuntimeError("Restored eval collection is missing")
        self.client.load_collection(collection_name=target_collection)
        return EvalSnapshotProvenance(
            snapshot_name,
            source_collection,
            target_collection,
            job_id,
            False,
        )

    def _describe(self, snapshot_name: str, source_collection: str) -> Any | None:
        try:
            return self.client.describe_snapshot(
                snapshot_name=snapshot_name,
                collection_name=source_collection,
            )
        except Exception:
            snapshots = self.client.list_snapshots()
            matching = [
                item
                for item in snapshots
                if (
                    item == snapshot_name
                    if isinstance(item, str)
                    else str(_attribute(item, "name")) == snapshot_name
                )
            ]
            if matching:
                stored_source = (
                    None
                    if isinstance(matching[0], str)
                    else str(_attribute(matching[0], "collection_name"))
                )
                if stored_source is not None and stored_source != source_collection:
                    raise RuntimeError("Snapshot source collection does not match")
                raise RuntimeError("Unable to validate existing Milvus snapshot")
            return None


def _attribute(value: Any, name: str) -> Any:
    if isinstance(value, dict):
        return value.get(name)
    return getattr(value, name, None)
