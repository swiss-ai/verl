from verl.checkpoint_engine.topology import *
import random
import ipaddress

def random_ipv4():
    return str(ipaddress.IPv4Address(random.getrandbits(32)))

def random_port():
    return random.randint(1, 65535)

def create_virtual_nodes(n_trainer_ranks, n_rollout_ranks):
    ranks_per_node = 4
    trainer_metas = [{"addr": None, "port": None, "session_id": None} for _ in range(n_trainer_ranks)]
    rollout_metas = [{"addr": None, "port": None, "session_id": None} for _ in range(n_rollout_ranks)]
    for idx, meta in enumerate(trainer_metas + rollout_metas):
        node_id = idx // ranks_per_node
        ip = f"203.0.113.{node_id % 254 + 1}"
        port = 1024 + idx
        session_port = 25565 + idx
        meta["addr"] = ip
        meta["port"] = port
        meta["session_id"] = f"{ip}:{session_port}"
    return trainer_metas + rollout_metas
    
def test_topology(n_trainer_ranks = 8, n_rollout_ranks = 32):
    metadatas = create_virtual_nodes(n_trainer_ranks, n_rollout_ranks)
    topology = DataParallelSimpleTopology()
    t_kwargs, r_kwargs = topology.build(
        trainer_ws=n_trainer_ranks,
        rollout_ws=n_rollout_ranks,
        metadatas=metadatas
    )
    for key_t in t_kwargs.keys():
        assert isinstance(t_kwargs[key_t], list) and len(t_kwargs[key_t]) == n_trainer_ranks
    for key_r in r_kwargs.keys():
        assert isinstance(r_kwargs[key_r], list) and len(r_kwargs[key_r]) == n_rollout_ranks
    
    ranks = set()
    for info in t_kwargs["info"]:
        assert info.is_trainer == True 
        assert info.local_rank != -1
        assert info.rank not in ranks
        ranks.add(info.rank)
    for info in r_kwargs["info"]:
        assert info.is_trainer == False
        assert info.local_rank != -1
        assert info.rank not in ranks
        ranks.add(info.rank)
    
    groups = set()
    for group in t_kwargs["group"] + r_kwargs["group"]:
        groups.add(group)
    print(f"I have: {len(groups)} groups composed")

    ranks.clear()
    ranks_in_group = 0
    for group in groups:
        assert not group.is_mixed
        if (ranks_in_group == 0):
            ranks_in_group = len(group.ranks)
        else:
            assert ranks_in_group == len(group.ranks)

        for local_rank, rank_info in enumerate(group.ranks): # no need to check for repeated, enumeration implies no possible rep
            assert rank_info.local_rank == local_rank
            assert rank_info.rank not in ranks
            ranks.add(rank_info.rank)
        
        # check send/recv logic

        # check trainer
        send_train_ranks = []
        for rank_info in group.trainer_ranks:
            local_rank = rank_info.local_rank
            assert rank_info.is_trainer
            assert local_rank not in group.recv_from.keys() # trainer ranks never receive
            if (local_rank in group.sends_to):
                send_train_ranks.append(rank_info)
                target_rank = group.sends_to[local_rank]
                target: RankInfo = group.ranks[target_rank]
                assert not target.is_trainer # trainer ranks send to rollout ranks
                assert target.addr != rank_info.addr # topology property, cxi should be used for inter-node
                assert target_rank == target.local_rank
                assert target_rank in group.recv_from.keys()
                assert group.recv_from[target_rank] == local_rank

        # by design, topology should be so that only one node of the trainer nodes in a rank
        addr = None
        for info in send_train_ranks:
            if (addr == None):
                addr = info.addr
            else:
                assert info.addr == addr

        # check rollouts
        send_rollout_ranks = []
        for rank_info in group.rollout_ranks:
            local_rank = rank_info.local_rank
            assert not rank_info.is_trainer
            # every rollout rank receives, might not send, but has to receive
            assert local_rank in group.recv_from
            if (local_rank in group.sends_to):
                send_rollout_ranks.append(rank_info)
                target_rank = group.sends_to[local_rank]
                target: RankInfo = group.ranks[target_rank]
                assert not target.is_trainer # never send to trainer
                assert target_rank == target.local_rank
                assert target_rank in group.recv_from.keys()
                assert group.recv_from[target_rank] == local_rank
                assert target.addr != rank_info.addr # must be inter node, we are just wasting bandwidth otherwise
        ranks_by_addr = {}
        for info in send_rollout_ranks:
            if info.addr not in ranks_by_addr:
                ranks_by_addr[info.addr] = 1
            else:
                ranks_by_addr[info.addr] += 1
        for addr, cnt in ranks_by_addr.items():
            node: NodeInfo = group.node_by_addr[addr]
            assert node.is_rollout()
            assert cnt == len(node.rollout_ranks)


if __name__ == "__main__":
    test_topology(4, 64)
    test_topology(8, 64)
    test_topology(16, 64)
    test_topology(32, 64)
    test_topology(64, 32)
    test_topology(64, 16)
    test_topology(64, 8)
    test_topology(64, 4)
