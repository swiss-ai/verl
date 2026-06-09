from dataclasses import dataclass
from typing import List, Any, Dict, Tuple
from verl.checkpoint_engine.utils import StatelessProcessGroup
from mooncake.engine import TransferEngine
import pickle
import ctypes
import msgpack

@dataclass
class RankInfo:
    local_rank: int
    rank: int
    addr: str
    port: int
    session_id: str
    is_trainer: bool

    def __hash__(self):
        return hash(self.addr + ":" + str(self.port))
    
    def __eq__(self, other):
        return (
            isinstance(other, RankInfo) and
            self.addr == self.addr and 
            self.port == self.port and
            self.is_trainer == other.is_trainer
        )
    
@dataclass
class NodeInfo:
    trainer_ranks: List[RankInfo]
    rollout_ranks: List[RankInfo]
    addr: str

    def __hash__(self):
        return hash(self.addr)

    def is_mixed(self):
        return len(self.trainer_ranks) > 0 and len(self.rollout_ranks) > 0
    
    def is_rollout(self):
        return len(self.trainer_ranks) == 0 and len(self.rollout_ranks) > 0

    def is_trainer(self):
        return len(self.trainer_ranks) > 0 and len(self.rollout_ranks) == 0

_has_cxi_store = False
_use_cxi_store = False

try:
    from cxi_store_py import CxiStore
    import pickle

    class SimpleStore():

        def __init__(self, device, buffer_size_kb):
            self.buffer_size_kb = buffer_size_kb
            self.store = CxiStore(device, buffer_size_kb)
            self.store.initialize()
            self.addr = self.store.get_addr()
        
        def add_peer(self, addr):
            self.store.add_peer(addr)

        def send_obj(self, dst: int, obj: object):
            data = msgpack.packb(obj, use_bin_type=True)
            # data = pickle.dumps(obj)
            sz = len(data)
            if (sz > self.buffer_size_kb * 1024):
                raise RuntimeError(f"object too big: {len(data) / 1024} KB")
            self.store.copy_to_buffer(data)
            print(f"sending object to {dst}")
            self.store.send(dst, sz)
            print("sent object")
        
        def recv_obj(self, src: int):
            print("receiving object")
            self.store.recv(src, self.buffer_size_kb * 1024)
            print("received object")
            data = self.store.copy_from_buffer(self.buffer_size_kb * 1024)
            unpacker = msgpack.Unpacker(raw=False)
            unpacker.feed(data)
            result = next(unpacker)
            return result
            
    _has_cxi_store = _use_cxi_store

except ImportError:
    pass 

class SimpleCommGroup:

    def __init__(self, nodes: List[NodeInfo]):
        if (len(nodes) > 1 and nodes[0].is_mixed()):
            raise RuntimeError("Mixed groups must be in isolated communication group")

        self.node_by_addr = {}
        for node in nodes:
            self.node_by_addr[node.addr] = node

        self.sends_to = {}
        self.recv_from = {}

        self.is_mixed = nodes[0].is_mixed()

        self.trainer_ranks: List[RankInfo] = []
        self.rollout_ranks: List[RankInfo] = []
        self.ranks: List[RankInfo] = []
        local_rank = 0
        for node in nodes:
            self.trainer_ranks += node.trainer_ranks
            self.rollout_ranks += node.rollout_ranks
            for info in node.trainer_ranks + node.rollout_ranks:
                info.local_rank = local_rank
                self.ranks.append(info)
                local_rank += 1

        for idx in range(len(self.trainer_ranks)):
            if (idx < len(self.rollout_ranks)):
                self.sends_to[self.trainer_ranks[idx].local_rank] = self.rollout_ranks[idx].local_rank
                self.recv_from[self.rollout_ranks[idx].local_rank] = self.trainer_ranks[idx].local_rank

        for idx in range(len(self.rollout_ranks)):
            rollout_sends_to_idx = idx + len(self.trainer_ranks)
            if (rollout_sends_to_idx < len(self.rollout_ranks)): # this rollout would have to send
                self.sends_to[self.rollout_ranks[idx].local_rank] = self.rollout_ranks[rollout_sends_to_idx].local_rank
            if (idx >= len(self.trainer_ranks)):
                rollout_recv_from_idx = idx  - len(self.trainer_ranks)
                self.recv_from[self.rollout_ranks[idx].local_rank] = self.rollout_ranks[rollout_recv_from_idx].local_rank
            
    def create_store(self, local_rank: int, device: str = None):
        world_size = len(self.trainer_ranks) + len(self.rollout_ranks)
        self.store = StatelessProcessGroup.create(
            rank = local_rank,
            world_size = world_size,
            host = self.trainer_ranks[0].addr,
            port = self.trainer_ranks[0].port 
        ) # store with master at first trainer rank of group
        if (_has_cxi_store):
            if (device is None):
                raise RuntimeError("SimpleStore requires specifying cxi device")
            self.simple_store = SimpleStore(
                device=device,
                buffer_size_kb=8
            )
            addrs = self.store.all_gather_obj({"addr": self.simple_store.addr})
            if (self.should_send(local_rank)):
                self.simple_store.add_peer(addrs[self.sends_to[local_rank]]["addr"])
            if (self.should_recv(local_rank)):
                self.simple_store.add_peer(addrs[self.recv_from[local_rank]]["addr"])
            

        print(f"created store of world_size={world_size} @ {self.trainer_ranks[0].addr}:{self.trainer_ranks[0].port}")

    def warmup_store(self):
        pass

    def should_recv(self, local_rank):
        return local_rank in self.recv_from
    
    def should_send(self, local_rank):
        return local_rank in self.sends_to

    def recv(self, local_rank: int):
        if (local_rank not in self.recv_from):
            raise RuntimeError(f"Illegal recv operation, local rank {local_rank} should not receive from any rank")
        else:
            if _has_cxi_store:
                peer = 0 if not self.should_send(local_rank) else 1 
                return self.simple_store.recv_obj(peer)
            return self.store.recv_obj(self.recv_from[local_rank])

    def recv_te_addr(self, local_rank: int):
        if (local_rank not in self.recv_from):
            raise RuntimeError(f"Local rank {local_rank} should not receive from any rank")
        return self.ranks[self.recv_from[local_rank]].session_id

    def send_te_addr(self, local_rank: int):
        if (local_rank not in self.sends_to):
            raise RuntimeError(f"Local rank {local_rank} should not send to any rank")
        return self.ranks[self.sends_to[local_rank]].session_id

    def send(self, obj: Any, local_rank: int):
        if (local_rank not in self.sends_to):
            raise RuntimeError(f"Illegal send operation, local rank {local_rank} should not receive send to rank")
        else:
            if _has_cxi_store:
                self.simple_store.send_obj(0, obj)
            self.store.send_obj(obj, self.sends_to[local_rank])

    def __eq__(self, other):
        if (not isinstance(other, SimpleCommGroup)):
            return False
        return other.ranks == self.ranks

class DataParallelSimpleTopology:
    """
    Assumes that all ranks participating in the transfer have a full replica of the weights
    in GPU HBM
    """

    def __init__(self):
        self._force_te_mixed_nodes = True # force RDMA on mixed train/rollout nodes, for debug purposes
        pass

    def build(self, trainer_ws: int, rollout_ws: int, metadatas: list[dict]):
        trainer_meta = metadatas[:trainer_ws]
        rollout_meta = metadatas[trainer_ws:]
        self.trainer_ws = trainer_ws
        self.rollout_ws = rollout_ws
        self._node_info(trainer_meta, rollout_meta)
        return self._build_groups()

    def _node_info(self, trainer_metas: list[dict], rollout_metas: list[dict]):
        self.node_info: Dict[str, NodeInfo] = {}
        self.rank_info = []
        rank = 0
        for meta in trainer_metas:
            addr = meta["addr"]
            port = meta["port"]
            session_id = meta["session_id"]
            info = RankInfo(-1, rank, addr, port, session_id, True)
            self.rank_info.append(info)
            if (addr in self.node_info.keys()):
                self.node_info[addr].trainer_ranks.append(info)
            else:
                self.node_info[addr] = NodeInfo([info], [], addr)
            rank += 1
        
        for meta in rollout_metas:
            addr = meta["addr"]
            port = meta["port"]
            session_id = meta["session_id"]
            info = RankInfo(-1, rank, addr, port, session_id, False)
            self.rank_info.append(info)
            if (addr in self.node_info.keys()):
                self.node_info[addr].rollout_ranks.append(info)
            else:
                self.node_info[addr] = NodeInfo([], [info], addr)
            rank += 1
            
        self.n_train_nodes = 0
        self.n_roll_nodes = 0
        self.n_mixed_nodes = 0
        for node in self.node_info.values():
            if (node.is_trainer()):
                self.n_train_nodes += 1
            elif (node.is_rollout()):
                self.n_roll_nodes += 1
            elif (node.is_mixed()):
                self.n_mixed_nodes += 1
            else:
                raise RuntimeError("Illegal node found")

        valid_topo = (
                self.n_train_nodes > 0 and self.n_roll_nodes > 0 and (
                    (self.n_train_nodes % self.n_roll_nodes) == 0 or (self.n_roll_nodes % self.n_train_nodes) == 0
                )
            ) or (self.n_train_nodes == 0 and self.n_roll_nodes == 0)

        if not valid_topo:
            raise RuntimeError("Invalid topology, there are unpairable non-mixed nodes!")

    def _build_groups(self):
        n_simple = min(self.n_train_nodes, self.n_roll_nodes)
        n_groups = n_simple + self.n_mixed_nodes
        groups_list = [[] for _ in range(n_groups)]
        t = 0
        r = 0
        m = 0
        trainer_kwargs = {
            "group": [None for _ in range(self.trainer_ws)],
            "info": [None for _ in range(self.trainer_ws)]
        }
        rollout_kwargs = {
            "group": [None for _ in range(self.rollout_ws)],
            "info": [None for _ in range(self.rollout_ws)]
        }
        for node in self.node_info.values():
            if (node.is_mixed()): 
                groups_list[n_simple + m] += [node]
                m += 1
            elif (node.is_rollout()):
                groups_list[r % n_simple] += [node]
                r += 1
            elif (node.is_trainer()):
                groups_list[t % n_simple] += [node]
                t += 1
        groups = [SimpleCommGroup(nodes) for nodes in groups_list]
        for group in groups:
            for t_rank in group.trainer_ranks:
                trainer_kwargs["group"][t_rank.rank] = group
            for r_rank in group.rollout_ranks:
                rollout_kwargs["group"][r_rank.rank - self.trainer_ws] = group
        trainer_kwargs["info"] = self.rank_info[:self.trainer_ws]
        rollout_kwargs["info"] = self.rank_info[self.trainer_ws:]
        return trainer_kwargs, rollout_kwargs

        