"""Thread-safe quorum-replicated object storage with integrity repair.

StorageNode is an in-memory reference backend. Production deployments can
implement its read/write interface using durable storage and transport RPCs.
"""
from __future__ import annotations

import hashlib
import threading
from dataclasses import dataclass
from typing import Dict, Iterable, Optional


class VaultError(Exception):
	pass


class QuorumError(VaultError):
	pass


class ObjectNotFound(VaultError):
	pass


class CorruptionError(VaultError):
	"""Raised when quorum was reached but every replica failed its checksum."""
	pass


@dataclass(frozen=True)
class Record:
	key: str
	version: int
	data: bytes
	checksum: str
	deleted: bool = False

	@classmethod
	def make(cls, key: str, version: int, data: bytes,
			 deleted: bool = False) -> "Record":
		return cls(key, version, data, hashlib.sha256(data).hexdigest(), deleted)

	def is_valid(self) -> bool:
		return hashlib.sha256(self.data).hexdigest() == self.checksum


class StorageNode:
	"""Independently available storage endpoint."""

	def __init__(self, node_id: str):
		self.node_id = node_id
		self.online = True
		self._records: Dict[str, Record] = {}
		self._lock = threading.RLock()

	def read(self, key: str) -> Optional[Record]:
		if not self.online:
			raise ConnectionError(self.node_id)
		with self._lock:
			return self._records.get(key)

	def write(self, record: Record) -> bool:
		"""Accept the record unless a *valid* local copy is already newer.

		A corrupted local record must never block an incoming write: its
		version field cannot be trusted once its checksum fails, so it is
		always eligible for replacement regardless of version ordering.
		Returns True if the record was stored, False if rejected as stale.
		"""
		if not self.online:
			raise ConnectionError(self.node_id)
		with self._lock:
			old = self._records.get(record.key)
			if old is None or not old.is_valid() or record.version >= old.version:
				self._records[record.key] = record
				return True
			return False

	def corrupt(self, key: str, data: bytes) -> None:
		"""Diagnostic hook to simulate disk corruption."""
		with self._lock:
			old = self._records[key]
			self._records[key] = Record(old.key, old.version, data,
										old.checksum, old.deleted)


class Vault:
	"""Replicated object store with quorum writes, reads, and read repair."""

	def __init__(self, nodes: Iterable[StorageNode], replication: int = 3,
				 write_quorum: Optional[int] = None,
				 read_quorum: Optional[int] = None):
		self.nodes = list(nodes)
		if not self.nodes or not 1 <= replication <= len(self.nodes):
			raise ValueError("replication must be within the node count")
		self.replication = replication
		self.write_quorum = replication if write_quorum is None else write_quorum
		self.read_quorum = replication // 2 + 1 if read_quorum is None else read_quorum
		if (not 1 <= self.write_quorum <= replication or
				not 1 <= self.read_quorum <= replication or
				self.write_quorum + self.read_quorum <= replication):
			raise ValueError("quorums must be valid and intersect")
		self._key_locks: Dict[str, threading.RLock] = {}
		self._locks_guard = threading.Lock()

	def _lock(self, key: str) -> threading.RLock:
		with self._locks_guard:
			return self._key_locks.setdefault(key, threading.RLock())

	def _targets(self, key: str) -> list[StorageNode]:
		"""Deterministic placement; rendezvous-style rotation is stable per key."""
		start = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big")
		offset = start % len(self.nodes)
		return (self.nodes[offset:] + self.nodes[:offset])[:self.replication]

	@staticmethod
	def _next_version(nodes: Iterable[StorageNode], key: str) -> int:
		versions = []
		for node in nodes:
			try:
				record = node.read(key)
				if record is not None and record.is_valid():
					versions.append(record.version)
			except ConnectionError:
				pass
		return max(versions, default=0) + 1

	@staticmethod
	def _validate_key(key: str) -> None:
		if not isinstance(key, str) or not key:
			raise ValueError("key must be a non-empty string")

	def _commit(self, key: str, data: bytes, deleted: bool = False) -> int:
		targets = self._targets(key)
		record = Record.make(key, self._next_version(targets, key), data, deleted)
		successes = 0
		for node in targets:
			try:
				if node.write(record):
					successes += 1
			except ConnectionError:
				continue
		if successes < self.write_quorum:
			raise QuorumError(f"write reached {successes}/{self.write_quorum} nodes")
		return record.version

	def put(self, key: str, data: bytes) -> int:
		self._validate_key(key)
		if not isinstance(data, bytes):
			raise TypeError("data must be bytes")
		with self._lock(key):
			return self._commit(key, data)

	def get(self, key: str) -> bytes:
		self._validate_key(key)
		with self._lock(key):
			targets = self._targets(key)
			responses, valid_records, corrupt_seen = 0, [], False
			for node in targets:
				try:
					record = node.read(key)
					responses += 1
					if record is not None:
						if record.is_valid():
							valid_records.append(record)
						else:
							corrupt_seen = True
				except ConnectionError:
					pass
			if responses < self.read_quorum:
				raise QuorumError(f"read reached {responses}/{self.read_quorum} nodes")
			if not valid_records:
				if corrupt_seen:
					raise CorruptionError(
						f"all quorum-reachable replicas of {key!r} failed checksum")
				raise ObjectNotFound(key)
			newest = max(valid_records, key=lambda item: item.version)
			for node in targets:
				try:
					current = node.read(key)
					if (current is None or not current.is_valid() or
							current.version < newest.version):
						node.write(newest)
				except ConnectionError:
					pass
			if newest.deleted:
				raise ObjectNotFound(key)
			return newest.data

	def delete(self, key: str) -> int:
		"""Replicate a tombstone so stale replicas cannot resurrect deleted data."""
		self._validate_key(key)
		with self._lock(key):
			return self._commit(key, b"", deleted=True)

	def repair(self, key: Optional[str] = None) -> dict:
		"""Reconcile replicas for one object, or all objects known to local nodes.

		Only keys within a node's target set for that key are reconciled;
		a replica stored on a node outside its rendezvous set is orphaned
		and intentionally left untouched (it is not part of this object's
		quorum group).
		"""
		if key is not None:
			self._validate_key(key)
			keys = {key}
		else:
			keys = set()
			for node in self.nodes:
				if not node.online:
					continue
				with node._lock:
					keys.update(node._records)
		checked = repaired = 0
		for object_key in keys:
			with self._lock(object_key):
				targets = self._targets(object_key)
				valid = []
				for node in targets:
					try:
						record = node.read(object_key)
						if record is not None and record.is_valid():
							valid.append(record)
					except ConnectionError:
						continue
				if not valid:
					continue
				newest = max(valid, key=lambda item: item.version)
				for node in targets:
					checked += 1
					try:
						current = node.read(object_key)
						if (current is None or not current.is_valid() or
								current.version < newest.version):
							if node.write(newest):
								repaired += 1
					except ConnectionError:
						pass
		return {"checked": checked, "repaired": repaired}

	def verify(self) -> dict:
		"""Count healthy, absent, corrupt, and offline replica slots."""
		counts = {"healthy": 0, "missing": 0, "corrupt": 0, "offline": 0}
		keys = set()
		for node in self.nodes:
			if not node.online:
				continue
			with node._lock:
				keys.update(node._records)
		for key in keys:
			for node in self._targets(key):
				try:
					record = node.read(key)
					counts["missing" if record is None else
						   "healthy" if record.is_valid() else "corrupt"] += 1
				except ConnectionError:
					counts["offline"] += 1
		return counts
 