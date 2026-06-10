# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import asyncio
import gc
import logging
import os
import time
from typing import Any, AsyncGenerator, Generator

import ray
import torch
from mooncake.engine import TransferEngine
from verl.checkpoint_engine.utils import StatelessProcessGroup

from verl.checkpoint_engine.base import CheckpointEngine, CheckpointEngineRegistry, TensorMeta
from verl.utils.device import get_torch_device
from verl.utils.net_utils import get_free_port
import pickle
from verl.checkpoint_engine.buffer_utils import BufferManager
from math import prod
import queue
import threading


logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))

class CounterSpinner:
    def __init__(self):
        # using torch for convenience, but can be anything as long as the GC doesn't mess with the memory behind
        self.counter_buf = torch.zeros((1), dtype=torch.int32, device="cpu") 
        self.last_counter = torch.empty((1), dtype=torch.int32, device="cpu")
        self.last_counter[0] = 0
    def spin(self):
        while (True):
            if (self.counter_buf == self.last_counter + 1):
                self.last_counter += 1 
                break

class Buffer():
    def __init__(self, buf_size, device):
        if (device == "cpu"):
            self.buffer = torch.zeros(buf_size, device="cpu", dtype=torch.uint8).pin_memory()
        else:
            self.buffer = torch.zeros(buf_size, device=device, dtype=torch.uint8)
        print(f"initialized buffer on {self.buffer.device}")
        self.read_doorbell = CounterSpinner()
    def wait_spin(self):
        self.read_doorbell.spin()

class MetadataBuffer():

    def __init__(self, buf_size_kb: int, engine: TransferEngine):
        self.engine = engine
        self.buffer = torch.zeros(1024 * buf_size_kb, device="cpu", dtype=torch.uint8) # no pin, CPU only
        self.write_doorbell = CounterSpinner()

    def pickle_on_buffer(self, obj: object):
        # not really zero copy but until I find an alternative it's the best we have
        pickled = pickle.dumps(obj)
        self.buffer[0:len(pickled)] = torch.frombuffer(pickled, dtype=torch.uint8)

    def unpickle_from_buffer(self) -> object:
        return pickle.loads(bytes(self.buffer))

    def wait_spin(self):
        self.write_doorbell.spin()

    # async def wait_spin_async(self):
    #     self.write_doorbell.spin()

@CheckpointEngineRegistry.register("cxi")
class CxiCheckpointEngine(CheckpointEngine):
    def __init__(
        self,
        bucket_size: int,
        device: str = "cuda",
        rollout_dtype: torch.dtype = torch.bfloat16,
        device_name: str = "",
        is_master: bool = False,
        rebuild_group: bool = False,
        is_trainer: bool = True
    ):
        self.bucket_size = bucket_size
        self.device = device
        self.rollout_dtype = rollout_dtype
        self.is_master = is_master
        self.is_trainer = is_trainer

        print(f"initialized cxi, is_trainer={is_trainer}")

        rank = int(os.environ["RANK"])
        device_count = get_torch_device().device_count()
        local_rank = rank % device_count
        get_torch_device().set_device(local_rank)

        self.engine = TransferEngine()
        hostname = ray.util.get_node_ip_address().strip("[]")
        local_rank_ray = int(ray.get_runtime_context().get_accelerator_ids()["GPU"][0])

        ret = self.engine.initialize(
            hostname,
            "P2PHANDSHAKE",
            "cxi",
            f"cxi{local_rank_ray % 4}",
        )

        # print(f"topology: {self.engine.get_local_topology()}")
        assert ret == 0, f"TransferEngine initialize failed ret={ret}"

        # weights = generate_random_weights()
        # self.weights_to_test = {}
        # for name, weight in weights:
        #     self.weights_to_test[name] = weight

        rpc_port = self.engine.get_rpc_port()
        self.session_id = f"{hostname}:{rpc_port}"
        self.hostname = hostname

        self.pipeline_stages = 1

        buf_device = (self.device if not is_trainer else "cpu")
        self.buffer_manager = BufferManager(
            pipeline_stages=self.pipeline_stages,
            buffer_size=self.bucket_size,
            engine=self.engine,
            buffer_device=buf_device,
            metadata_buffer_size=1024 * 8
        )
        self.buffer_manager.initialize_buffers()

        logger.info(f"__init__ session_id={self.session_id}")

    def prepare(self) -> dict[str, Any]:
        """Prepare send and recv buckets"""
        port, _ = get_free_port(self.hostname)
        return {"addr": self.hostname, "port": port}

    @classmethod
    def build_topology(cls, trainer_world_size: int, rollout_world_size: int, metadatas: list[dict]):
        trainer_kwargs = {
            "rank": [0] + [-1] * (trainer_world_size - 1),
            "world_size": [rollout_world_size + 1] * trainer_world_size,
            "metadata": [metadatas[0]] * trainer_world_size,
        }
        rollout_kwargs = {
            "rank": list(range(1, rollout_world_size + 1)),
            "world_size": [rollout_world_size + 1] * rollout_world_size,
            "metadata": [metadatas[0]] * rollout_world_size,
        }
        print(metadatas)
        print(f"topology_trainer={trainer_kwargs}\ntopology_rollout={rollout_kwargs}")
        return trainer_kwargs, rollout_kwargs

    def init_process_group(self, rank: int, world_size: int, metadata: dict[str, Any]):
        # start the tcp store, the pattern is a ring, the actor is rank 0 and the rollouts are rank 1,...,n
        self.rank = rank
        self.world_size = world_size
        if rank < 0:
            logger.info(f"init_process_group rank={rank}")
            return
        self.store = StatelessProcessGroup.create(
            host=metadata["addr"],
            port=metadata["port"],
            rank=rank,
            world_size=world_size,
        )

        info = {
            "session_id": self.session_id,
            "buffer_ptrs": [],
            "metadata_buffer_ptrs": [],
            "doorbell_wb_ptrs": [],
            "doorbell_md_ptrs": []
        }
        for desc in self.buffer_manager.descs:
            info["buffer_ptrs"].append(desc.data_buffer.data_ptr())
            info["metadata_buffer_ptrs"].append(desc.metadata_buffer.data_ptr())
            info["doorbell_wb_ptrs"].append(desc.doorbell_wb.data_ptr())
            info["doorbell_md_ptrs"].append(desc.doorbell_md.data_ptr())

        info_list = self.store.all_gather_obj(info)
        self.target_addr = None if rank == 0 else info_list[rank - 1]["session_id"]
        self.target_addr_meta = None if rank == self.world_size - 1 else info_list[rank + 1]["session_id"]

        self.target_data_ptrs = None
        self.target_metadata_ptrs = None
        self.target_doorbell_wb_ptrs = None
        self.target_doorbell_md_ptrs = None
        if (rank > 0):
            self.target_data_ptrs = info_list[rank - 1]["buffer_ptrs"]
            self.target_metadata_ptrs = info_list[rank - 1]["metadata_buffer_ptrs"]
            self.target_doorbell_wb_ptrs = info_list[rank - 1]["doorbell_wb_ptrs"]
        if (rank < world_size - 1):
            self.target_doorbell_md_ptrs = info_list[rank + 1]["doorbell_md_ptrs"]

        self.wb_values = [0] * self.pipeline_stages
        self.md_values = [0] * self.pipeline_stages

        logger.info(f"init_process_group rank={rank} world_size={world_size}")

    def finalize(self):
        """Cleanup communication and deregister memory"""
        self.store = None
        self.buffer_manager.reset_counters()
        # for i in range(self.pipeline_stages):
        #     self.buffers[i].read_doorbell.counter_buf[0] = 0
        #     self.buffers[i].read_doorbell.last_counter[0] = 0

        get_torch_device().empty_cache()
        # gc.collect()
        logger.info(f"finalize rank={self.rank}")

    @torch.no_grad()
    async def send_weights(self, weights: Generator[tuple[str, torch.Tensor], None, None]):
        """Send weights using Mooncake TransferEngine"""
        if self.rank < 0:
            for name, weight in weights:
                pass
            logger.info(f"send_weights rank={self.rank}")
            return

        total_bytes = 0
        start_time = time.time()
        bucket_meta: dict[str, TensorMeta] = {}
        offset = 0
        idx = 0
        current = self.buffer_manager.descs[idx]
        
        start = time.perf_counter()

        for name, weight in weights:
            weight = weight.to(self.rollout_dtype)

            if offset + weight.nbytes > self.bucket_size:
                total_bytes += offset
                get_torch_device().synchronize() # gpu ops must be finished before telling rank 1 our buffer is ready
                end = time.perf_counter()
                fill_bucket_dur = end-start
                print(f"filled bucket: {fill_bucket_dur} [s], bw: {(total_bytes / 1e9) / (fill_bucket_dur)} ")
                # metadata
                info = {
                    "bucket_meta": bucket_meta,
                    "len": offset,
                    "is_last": False,
                    # data that must be modified when sent to the next
                    "ptr": current.data_buffer.data_ptr(),
                    "door_ptr": current.doorbell_wb.data_ptr(),
                    "door_val": current.comp_wb.item(),
                }
                # send to rank 1
                # set doorbell and proceed
                self.buffer_manager.copy_obj_to_buffer(idx % self.pipeline_stages, info)
                prev_door_md_val = self.md_values[idx % self.pipeline_stages]
                current.imm_scratch_md[0] = prev_door_md_val + 1
                self.md_values[idx % self.pipeline_stages] += 1 # IMPORTANT, increment or it doesn't work
                ret = self.engine.transfer_sync_write(
                    self.target_addr_meta,
                    current.imm_scratch_md.data_ptr(),
                    self.target_doorbell_md_ptrs[idx % self.pipeline_stages],
                    4
                )
                if (ret != 0):
                    print(f"write failed: target_addr={self.target_addr_meta}, scratch_md_ptr={hex(current.imm_scratch_md.data_ptr())}, remote_ptr={hex(self.target_doorbell_md_ptrs[idx % self.pipeline_stages])}")
                    raise RuntimeError()
                # self.store.send_obj(info, 1)

                idx += 1
                current = self.buffer_manager.descs[idx % self.pipeline_stages]
                bucket_meta = {}
                offset = 0

                if (idx >= self.pipeline_stages):
                    await self.buffer_manager.wait_spin_wb(idx % self.pipeline_stages)

            assert offset + weight.nbytes <= self.bucket_size, (
                f"Weight {name}({weight.shape}, {weight.dtype}) is too large to fit in the bucket."
            )

            bucket_meta[name] = {
                "name": name,
                "shape": weight.shape,
                "dtype": weight.dtype,
                "offset": offset,
            }
            # start = time.perf_counter()
            current.data_buffer[offset : offset + weight.nbytes].view(weight.dtype).copy_(weight.view(-1), non_blocking=True)
            # end = time.perf_counter()
            # print(f"sync H2D in: {(weight.nbytes / 1e9 ) / (end - start)} GB/s")
            offset += weight.nbytes

        get_torch_device().synchronize()
        info = {
            "bucket_meta": bucket_meta,
            "ptr": current.data_buffer.data_ptr(),
            "len": offset,
            "door_ptr": current.doorbell_wb.data_ptr(),
            "door_val": current.comp_wb.item(),
            "is_last": True,
        }
        self.buffer_manager.copy_obj_to_buffer(idx % self.pipeline_stages, info)
        prev_door_md_val = self.md_values[idx % self.pipeline_stages]
        current.imm_scratch_md[0] = prev_door_md_val + 1
        self.md_values[idx % self.pipeline_stages] += 1 # IMPORTANT, increment or it doesn't work
        self.engine.transfer_sync_write(
            self.target_addr_meta,
            current.imm_scratch_md.data_ptr(),
            self.target_doorbell_md_ptrs[idx % self.pipeline_stages],
            4
        )

        # self.store.send_obj(info, 1)
        await self.buffer_manager.wait_spin_wb(idx % self.pipeline_stages)

        total_bytes += offset
        time_cost = time.time() - start_time
        bandwidth = total_bytes / time_cost / (1e9)
        logger.info(
            f"Rank {self.rank} send weights done, "
            f"total bytes: {total_bytes} time cost: {time_cost:.2f}s bandwidth: {bandwidth:.2f} GB/s"
        )

    @torch.no_grad()
    async def receive_weights(self) -> AsyncGenerator[tuple[str, torch.Tensor], None]:
        """Receive weights using Mooncake TransferEngine"""
        start_time = time.time()
        total_bytes = 0
        idx = 0
        current = self.buffer_manager.descs[idx]

        while True:
            # 1 receive info from previous rank
            preceding_address = self.target_addr

            await self.buffer_manager.wait_spin_md(idx % self.pipeline_stages)
            meta_start = time.perf_counter()
            ret = self.engine.transfer_sync_read(
                preceding_address,
                current.metadata_buffer.data_ptr(), # our meta buffer,
                self.target_metadata_ptrs[idx % self.pipeline_stages],
                self.buffer_manager.metadata_buffer_size
            )
            meta_end = time.perf_counter()
            print(f"read time: {(meta_end-meta_start)*1000:.2f} ms")
            if (ret != 0):
                raise RuntimeError("read failed!")
            info = self.buffer_manager.get_obj_from_buffer(idx % self.pipeline_stages)
            # info = self.store.recv_obj(self.rank - 1)
            
            if (idx == 0):
                start_time = time.time()
            if idx >= self.pipeline_stages and self.rank < self.world_size - 1:
                await self.buffer_manager.wait_spin_wb(idx % self.pipeline_stages)


            prev_buf_ptr = self.target_data_ptrs[idx % self.pipeline_stages]
            prev_door_val = self.wb_values[idx % self.pipeline_stages]
            prev_door_ptr = self.target_doorbell_wb_ptrs[idx % self.pipeline_stages]
            current.imm_scratch_wb[0] = prev_door_val + 1
            # print(f"value: {prev_door_val}")
            self.wb_values[idx % self.pipeline_stages] += 1 # IMPORTANT, increment or nothing works

            info["ptr"] = current.data_buffer.data_ptr()
            info["door_val"] = current.comp_wb.item()
            info["door_ptr"] = current.doorbell_wb.data_ptr()

            start_transfer_time = time.perf_counter()
            ret = self.engine.transfer_sync_read(
                preceding_address, # previous rank address
                current.data_buffer.data_ptr(), # our buffer
                prev_buf_ptr, # remote buffer
                info["len"], # how much data
            )
            transfer_dur = time.perf_counter() - start_transfer_time
            assert ret == 0, f"transfer_sync_read failed {ret}"

            # benchmark transfer speed
            # time_cost_this_step = transfer_dur
            # bandwidth_this_step = info["len"] / time_cost_this_step / (1e9)
            # print(f"RANK: {self.rank}, step {idx}: this step bandwidth: {bandwidth_this_step} GB/s, target hostname: {self.buffer_info["session_id"]}")

            total_bytes += info["len"]

            # 2 tell the next rank that our buffer is ready to be read from
            if (self.rank < self.world_size - 1):
                self.buffer_manager.copy_obj_to_buffer(idx % self.pipeline_stages, info)
                prev_door_md_val = self.md_values[idx % self.pipeline_stages]
                current.imm_scratch_md[0] = prev_door_md_val + 1
                self.md_values[idx % self.pipeline_stages] += 1 # IMPORTANT, increment or it doesn't work
                ret = self.engine.transfer_sync_write(
                    self.target_addr_meta,
                    current.imm_scratch_md.data_ptr(),
                    self.target_doorbell_md_ptrs[idx % self.pipeline_stages],
                    4
                )
                assert ret == 0, f"doorbell metadata write failed"
                # self.store.send_obj(info, self.rank + 1)

            # 3 yield tensor from current buffer
            for name, meta in info["bucket_meta"].items():
                dtype, shape = meta["dtype"], meta["shape"]
                size = dtype.itemsize * shape.numel()
                tensor = current.data_buffer[meta["offset"] : meta["offset"] + size].view(dtype=dtype).view(shape)
                yield name, tensor

            # 4 tell the previous rank that we read from the buffer
            ret = self.engine.transfer_sync_write(
                preceding_address, # target
                current.imm_scratch_wb.data_ptr(), 
                prev_door_ptr,
                4, # int32 is 4 bytes
            )
            assert ret == 0, f"transfer_sync_write failed {ret}"
            print(f"writeback of value {current.imm_scratch_wb[0]} ok")

            get_torch_device().synchronize()

            # 5 swap buffer
            idx += 1
            current = self.buffer_manager.descs[idx % self.pipeline_stages]
            
            if info["is_last"]:
                break

        time_cost = time.time() - start_time
        bandwidth = total_bytes / time_cost / (1e9)
        logger.info(
            f"Rank {self.rank} receive weights done, time cost: {time_cost:.2f}s, bandwidth: {bandwidth:.2f} GB/s"
        )

from .topology import DataParallelSimpleTopology, RankInfo, SimpleCommGroup

@CheckpointEngineRegistry.register("cxi_sharded")
class ShardedCxiCheckpointEngine(CheckpointEngine):
    def __init__(
        self,
        bucket_size: int,
        device: str = "cuda",
        rollout_dtype: torch.dtype = torch.bfloat16,
        device_name: str = "",
        is_master: bool = False,
        rebuild_group: bool = False,
        is_trainer: bool = True
    ):
        self.bucket_size = bucket_size
        self.device = device
        self.rollout_dtype = rollout_dtype
        self.is_master = is_master
        self.is_trainer = is_trainer
        self.rebuild_group = rebuild_group
        # print(f"initialized cxi, is_trainer={is_trainer}")
        rank = int(os.environ["RANK"])
        device_count = get_torch_device().device_count()
        local_rank = rank % device_count
        get_torch_device().set_device(local_rank)

        self.engine = TransferEngine()
        hostname = ray.util.get_node_ip_address().strip("[]")
        # it's kinda ugly but required, otherwise CPU buffers will all be bound to NIC 0,
        # and throughput will be abysmal
        local_rank_ray = int(ray.get_runtime_context().get_accelerator_ids()["GPU"][0])
        self.cxi_device = f"cxi{local_rank_ray % 4}" 
        ret = self.engine.initialize(
            hostname,
            "P2PHANDSHAKE",
            "cxi",
            self.cxi_device,
        )
        assert ret == 0, f"TransferEngine initialize failed ret={ret}"

        rpc_port = self.engine.get_rpc_port()
        self.session_id = f"{hostname}:{rpc_port}"
        self.hostname = hostname
        self.group = None

        self._prefetch = False
        self._debug = False

        self.weight_info = None
        self.buckets = []

        self.pipeline_stages = 2 # number of buckets to keep
        # to save memory, keep trainer buffers on CPU and offload the weights there
        # (implicitly assumes CPU RAM is abundant)
        buf_device = (self.device if not is_trainer else "cpu")
        # buffers to keep weights
        self.buffers = [Buffer(self.bucket_size, buf_device) for _ in range(self.pipeline_stages)]
        buffer_mrs = [b.buffer.data_ptr() for b in self.buffers]
        buffer_mr_sizes = [self.bucket_size for b  in self.buffers]
        # doorbell counters so that receiver -> sender notifies that the buffer is consumed and the sender is
        # free to reuse it 
        buffer_doorbells_mrs = [b.read_doorbell.counter_buf.data_ptr() for b in self.buffers]
        buffer_doorbells_mr_sizes = [4 for b in self.buffers]
        # temporary buffers for remote doorbell write (limit of libfabric, it's not possible to write/atomic immediate values)
        self.temp_cntr_bufs = [torch.empty((1), dtype=torch.int32, device="cpu") for _ in range(self.pipeline_stages)]
        temp_cntr_mrs = [t.data_ptr() for t in self.temp_cntr_bufs]
        temp_cntr_mr_sizes = [4 for _ in self.temp_cntr_bufs]
        # register memory regions in transfer engine
        ret = self.engine.batch_register_memory(
            buffer_mrs + buffer_doorbells_mrs + temp_cntr_mrs,
            buffer_mr_sizes + buffer_doorbells_mr_sizes + temp_cntr_mr_sizes
        )
        assert ret == 0, f"batch_register_memory failed ret={ret}"
        logger.info(f"__init__ session_id={self.session_id}")

    def prepare(self) -> dict[str, Any]:
        """Prepare send and recv buckets"""
        port, _ = get_free_port(self.hostname)
        return {"addr": self.hostname, "port": port, "session_id": self.session_id}

    @classmethod
    def build_topology(cls, trainer_world_size: int, rollout_world_size: int, metadatas: list[dict]):
        topo = DataParallelSimpleTopology()
        trainer_kwargs, rollout_kwargs = topo.build(
            trainer_ws=trainer_world_size,
            rollout_ws=rollout_world_size,
            metadatas=metadatas
        )
        return trainer_kwargs, rollout_kwargs

    def init_process_group(self, group: SimpleCommGroup, info: RankInfo):
        if (self.group == None or self.rebuild_group):
            self.info = info
            self.group = group
            self.group_local_rank = self.info.local_rank
            self.group.create_store(self.group_local_rank, self.cxi_device)
            self.recvs = self.group.should_recv(self.group_local_rank)
            self.sends = self.group.should_send(self.group_local_rank)
            if (self.recvs):
                self.recv_te_addr = self.group.recv_te_addr(self.group_local_rank)
        else:
            print(f"called initialize on {self.group_local_rank}")

    def finalize(self):
        if (self.rebuild_group):
            self.group.store = None
            self.weight_info = None # if we rebuild, cancel the weight list
            self.buckets = []
        for i in range(self.pipeline_stages):
            self.buffers[i].read_doorbell.counter_buf[0] = 0
            self.buffers[i].read_doorbell.last_counter[0] = 0
        get_torch_device().empty_cache()
        gc.collect()
        logger.info(f"finalize rank={self.group_local_rank}")

    def get_weight_buckets(self, weights: Generator[tuple[str, torch.Tensor], None, None]):
        total_bytes = 0
        offset = 0

        buckets = []
        weight_info = {}

        bucket_meta = {}
        for name, weight in weights:
            weight_info[name] = weight
            weight_bytes_rollout = weight.numel() * (self.rollout_dtype.itemsize)
            if (offset + weight_bytes_rollout > self.bucket_size):
                # bucket filled
                total_bytes += offset
                info = {
                    "len": offset,
                    "is_last": False,
                    "bucket_meta": bucket_meta
                }
                buckets.append(info)

                bucket_meta = {}
                offset = 0

            assert offset + weight_bytes_rollout <= self.bucket_size, (
                f"Weight {name}({weight.shape}, {weight.dtype}) is too large to fit in the bucket."
            ) # might change, but would require staging of weights between iterations

            bucket_meta[name] = {
                "shape": weight.shape,
                "offset": offset
            }
            offset += weight_bytes_rollout

        info = {
            "bucket_meta": bucket_meta,
            "is_last": True,
            "len": offset
        }
        buckets.append(info)

        return buckets, weight_info


    @torch.no_grad()
    async def send_weights(self, weights: Generator[tuple[str, torch.Tensor], None, None], global_steps: int | None = None):
        """Send weights using Mooncake TransferEngine"""
        if not self.sends:
            if (self.weight_info is None):
                self.weight_info = {}
                for name, weight in weights:
                    pass
            logger.info(f"skipping send_weights for rank={self.group_local_rank}")
            return

        total_bytes = 0
        start_time = time.time()

        if (self.weight_info is None):
            # if we're here we must be a sender rank
            wl_sr_start = time.perf_counter()
            b, wi = self.get_weight_buckets(weights)
            self.weight_info = wi
            self.buckets = b
            self.group.send({"buckets": self.buckets}, self.group_local_rank)
            wl_sr_end = time.perf_counter()
            if self._debug:
                print(f"rank: {self.group_local_rank}, weight meta time: {(wl_sr_end-wl_sr_start)*1000:.2f} ms")
        # else: 
        #     # the iterator must be consumed, otherwise everything hangs, don't know if it's a feature or a bug, but
        #     # otherwise doesn't work
        #     for name, weight in weights:
        #         pass
            

        # bucket_meta: dict[str, TensorMeta] = {}
        # offset = 0
        # bufs = self.buffers
        # idx = 0
        # current = bufs[idx]
        for idx, bucket in enumerate(self.buckets):
            current = self.buffers[idx % self.pipeline_stages]
            if (idx >= self.pipeline_stages):
                current.wait_spin()

            meta = bucket["bucket_meta"]
            for name, val in meta.items():
                weight = self.weight_info[name]
                weight = weight.to(self.rollout_dtype)
                offset = val["offset"]
                current.buffer[offset : offset + weight.nbytes].view(weight.dtype).copy_(weight.view(-1), non_blocking=True)
            get_torch_device().synchronize()
            self.group.send({
                "idx": idx, 
                "ptr": current.buffer.data_ptr(), 
                "door_ptr": current.read_doorbell.counter_buf.data_ptr(),
                "door_val": current.read_doorbell.last_counter.item()
            }, self.group_local_rank)
            total_bytes += bucket["len"]


        # finished send, print stats
        time_cost = time.time() - start_time
        bandwidth = total_bytes / time_cost / (1e9)
        logger.info(
            f"Rank {self.group_local_rank} send weights done, "
            f"total bytes: {total_bytes} time cost: {time_cost:.2f}s bandwidth: {bandwidth:.2f} GB/s"
        )

    @torch.no_grad()
    async def receive_weights(self, global_steps: int | None = None) -> AsyncGenerator[tuple[str, torch.Tensor], None]:
        
        # total_bytes = 0
        start_time = time.time()

        stats = {
            "total_bytes": 0,
            "start_time": time.time()
        }

        done_queue = queue.Queue(maxsize=self.pipeline_stages)
        slot_acks = [threading.Event() for _ in range(self.pipeline_stages)]

        def _transfer_loop():

            # initial metadata send/recv
            if (self.weight_info is None):
                wl_sr_start = time.perf_counter()
                obj = self.group.recv(self.group_local_rank)
                self.weight_info = {} # don't need the weight tensors, just the names and shapes, set it to whatever != None
                self.buckets = obj["buckets"]
                if (self.sends):
                    self.group.send(obj, self.group_local_rank)
                wl_sr_end = time.perf_counter()
                if self._debug:
                    print(f"rank: {self.group_local_rank}, weight meta time: {(wl_sr_end-wl_sr_start)*1000:.2f} ms")


            idx = 0
            current = self.buffers[idx]
            temp_cntr_buf = self.temp_cntr_bufs[idx]

            info = None
            info_next = None

            while True:
                slot = idx % self.pipeline_stages

                # 1 receive info from previous rank
                # info = self.store.recv_obj(self.rank - 1)
                t0 = time.perf_counter()
                if (self._prefetch):
                    if (info == None):
                        info = self.group.recv(self.group_local_rank)
                    elif (info_next != None):
                        info = info_next
                        info_next = None
                else:
                    info = self.group.recv(self.group_local_rank)
                t1 = time.perf_counter()

                if self._debug:
                    print(f"[idx={idx}] rank: {self.group_local_rank}, recv in {(t1-t0) * 1000:.2f} ms for metadata")

                if (idx >= self.pipeline_stages):
                    get_torch_device().synchronize()
                    slot_acks[idx % self.pipeline_stages].wait()
                    slot_acks[idx % self.pipeline_stages].clear()

                if (idx == 0):
                    stats["transfer_start_time"] = time.time()
                
                if idx >= self.pipeline_stages and self.sends: 
                    current.wait_spin()

                preceding_address = self.recv_te_addr

                # this could also be shared with an all_gather
                prev_buf_ptr = info["ptr"] # buffer we need to read
                prev_door_val = info["door_val"]
                prev_door_ptr = info["door_ptr"]
                temp_cntr_buf[0] = prev_door_val + 1
                # data
                info["ptr"] = current.buffer.data_ptr()
                info["door_val"] = current.read_doorbell.last_counter.item()
                info["door_ptr"] = current.read_doorbell.counter_buf.data_ptr()

                bucket = self.buckets[info["idx"]]

                start_transfer_time = time.perf_counter()
                track = self.engine.batch_transfer_async_read( 
                    preceding_address,
                    [current.buffer.data_ptr()],
                    [prev_buf_ptr],
                    [bucket["len"]]
                )
                assert track != 0, f"transfer failed!"
                # overlap:
                if (self._prefetch and not bucket["is_last"]):
                    recv_start = time.perf_counter()
                    info_next = self.group.recv(self.group_local_rank)
                    recv_dur = (time.perf_counter() - recv_start) * 1000
                    if self._debug:
                        print(f"[idx={idx}][rank={self.group_local_rank}] recv_time={recv_dur:.2f} ms")
            
                while (True):
                    status = self.engine.transfer_check_status(track)
                    if (status == 1):
                        transfer_dur = time.perf_counter() - start_transfer_time
                        break
                    elif (status < 0):
                        raise RuntimeError(f"Transfer failed with status: {status}")
                    time.sleep(0) # yield

                # benchmark transfer speed
                time_cost_this_step = transfer_dur
                bandwidth_this_step = bucket["len"] / time_cost_this_step / (1e9)
                if self._debug:
                    print(f"RANK: {self.group_local_rank}, step {idx}: this step bandwidth: {bandwidth_this_step} GB/s, duration: {transfer_dur * 1000:.2f} ms")

                done_queue.put(item=(slot, bucket, current.buffer))

                stats["total_bytes"] += bucket["len"]
                # 2 tell the next rank that our buffer is ready to be read from
                if (self.sends):
                    self.group.send(info, self.group_local_rank)

                # 4 tell the previous rank that we read from the buffer
                wb_start = time.perf_counter()
                ret = self.engine.transfer_sync_write(
                    self.recv_te_addr, # target
                    temp_cntr_buf.data_ptr(), 
                    prev_door_ptr,
                    4, # int32 is 4 bytes
                )
                wb_end = time.perf_counter()
                assert ret == 0, f"transfer_sync_write failed {ret}"
                if self._debug:
                    print(f"[idx={idx}] rank: {self.group_local_rank}, wb_duration={(wb_end-wb_start) * 1000:.2f} ms")
                # 5 swap buffer
                idx += 1
                current = self.buffers[idx % self.pipeline_stages]
                temp_cntr_buf = self.temp_cntr_bufs[idx % self.pipeline_stages]
                if bucket["is_last"]:
                    break
        
        thread = threading.Thread(target=_transfer_loop)
        thread.start()

        while True:
            slot, bucket, buffer = done_queue.get()
            for name, meta in bucket["bucket_meta"].items():
                dtype, shape = self.rollout_dtype, meta["shape"]
                numel = prod(shape)
                size = dtype.itemsize * numel
                tensor = buffer[meta["offset"] : meta["offset"] + size].view(dtype=dtype).view(shape)
                yield name, tensor

            slot_acks[slot].set()

            if bucket["is_last"]:
                break

        thread.join()

        # print stats
        idle_cost = stats["transfer_start_time"] - start_time
        transfer_cost = time.time() - stats["transfer_start_time"]
        # time_cost = time.time() - start_time
        bandwidth = stats["total_bytes"] / transfer_cost / (1e9)
        logger.info(
            f"Rank {self.group_local_rank} receive weights done, time cost: {transfer_cost:.2f}s, idle cost: {idle_cost:.2f}s, bandwidth: {bandwidth:.2f} GB/s"
        )
