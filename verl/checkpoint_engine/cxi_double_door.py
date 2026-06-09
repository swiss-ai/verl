from .topology import DataParallelSimpleTopology, RankInfo, SimpleCommGroup

import asyncio
import gc
import logging
import os
import time
from typing import Any, AsyncGenerator, Generator

import ray
import torch
from mooncake.engine import TransferEngine
from verl.checkpoint_engine.base import CheckpointEngine, CheckpointEngineRegistry, TensorMeta
from verl.utils.device import get_torch_device
from verl.utils.net_utils import get_free_port
from verl.checkpoint_engine.buffer_utils import BufferManager


logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))

@CheckpointEngineRegistry.register("cxi_dd")
class DoubleDoorCxiCheckpointEngine(CheckpointEngine):
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

        self.pipeline_stages = 2 # number of buckets to keep
        # to save memory, keep trainer buffers on CPU and offload the weights there
        # (implicitly assumes CPU RAM is abundant)
        self.buffer_device = (self.device if not is_trainer else "cpu")
        
        self.buffer_manager = BufferManager(
            self.pipeline_stages, self.bucket_size, self.engine, self.buffer_device, 8 * 1024
        )
        self.buffer_manager.initialize_buffers()
        
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
        if (self.group != group and not self.rebuild_group):
            self.info = info
            self.group = group
            self.group_local_rank = self.info.local_rank
            self.group.create_store(self.group_local_rank, self.cxi_device)
            self.recvs = self.group.should_recv(self.group_local_rank)
            self.sends = self.group.should_send(self.group_local_rank)
            if (self.recvs):
                self.recv_te_addr = self.group.recv_te_addr(self.group_local_rank)
            if (self.sends):
                self.send_te_addr = self.group.send_te_addr(self.group_local_rank)
            
            info = {
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

            info_list = self.group.store.all_gather_obj(info)

            self.target_data_ptrs = None
            self.target_metadata_ptrs = None
            self.target_doorbell_wb_ptrs = None
            self.target_doorbell_md_ptrs = None
            if (self.recvs):
                recv_from = self.group.recv_from[self.group_local_rank]
                self.target_data_ptrs = info_list[recv_from]["buffer_ptrs"]
                self.target_metadata_ptrs = info_list[recv_from]["metadata_buffer_ptrs"]
                self.target_doorbell_wb_ptrs = info_list[recv_from]["doorbell_wb_ptrs"]
            if (self.sends):
                sends_to = self.group.sends_to[self.group_local_rank]
                self.target_doorbell_md_ptrs = info_list[sends_to]["doorbell_md_ptrs"]

        # always reinit to 0
        self.wb_values = [0] * self.pipeline_stages
        self.md_values = [0] * self.pipeline_stages

    def copy_and_send(self, pipeline_stage: int, info: object):
        assert pipeline_stage < self.pipeline_stages
        current = self.buffer_manager.descs[pipeline_stage]
        self.buffer_manager.copy_obj_to_buffer(pipeline_stage, info)
        prev_door_md_val = self.md_values[pipeline_stage]
        current.imm_scratch_md[0] = prev_door_md_val + 1
        self.md_values[pipeline_stage] += 1 # IMPORTANT, increment or it doesn't work
        ret = self.engine.transfer_sync_write(
            self.send_te_addr,
            current.imm_scratch_md.data_ptr(),
            self.target_doorbell_md_ptrs[pipeline_stage],
            4
        )
        print(f"target md buffer should be: {current.imm_scratch_md[0]}")
        assert ret == 0, "copy and send failed!"

    async def recv_and_copy(self, pipeline_stage: int) -> object:
        assert pipeline_stage < self.pipeline_stages
        await self.buffer_manager.wait_spin_md(pipeline_stage)
        print("wait complete!")
        current = self.buffer_manager.descs[pipeline_stage]
        meta_start = time.perf_counter()
        ret = self.engine.transfer_sync_read(
            self.recv_te_addr,
            current.metadata_buffer.data_ptr(), # our meta buffer,
            self.target_metadata_ptrs[pipeline_stage],
            self.buffer_manager.metadata_buffer_size
        )
        meta_end = time.perf_counter()
        print(f"read time: {(meta_end-meta_start)*1000:.2f} ms")
        assert ret == 0, "recv and copy failed!"
        return self.buffer_manager.get_obj_from_buffer(pipeline_stage)
    
    async def submit_recv(self, pipeline_stage: int):
        assert pipeline_stage < self.pipeline_stages

        spin_start = time.perf_counter()
        await self.buffer_manager.wait_spin_md(pipeline_stage)
        spin_end = time.perf_counter()
        print(f"prefetch, spin time: {(spin_end-spin_start)*1000:.2f}")
        current = self.buffer_manager.descs[pipeline_stage]
        track = self.engine.batch_transfer_async_read(
            self.recv_te_addr,
            [current.metadata_buffer.data_ptr()], # our meta buffer,
            [self.target_metadata_ptrs[pipeline_stage]],
            [self.buffer_manager.metadata_buffer_size]
        )
        assert track != 0, "submit recv failed!"
        return track

    def sync_with_recv(self, pipeline_stage: int, track):
        assert pipeline_stage < self.pipeline_stages
        while (True):
            status = self.engine.transfer_check_status(track)
            if (status == 1):
                break
            elif (status < 0):
                raise RuntimeError(f"Transfer failed with status: {status}")
        return self.buffer_manager.get_obj_from_buffer(pipeline_stage)

    def finalize(self):
        if (self.rebuild_group):
            self.group.store = None

        self.buffer_manager.reset_counters()
        
        get_torch_device().empty_cache()
        gc.collect()
        logger.info(f"finalize rank={self.group_local_rank}")

    @torch.no_grad()
    async def send_weights(self, weights: Generator[tuple[str, torch.Tensor], None, None]):
        """Send weights using Mooncake TransferEngine"""
        if not self.sends:
            for name, weight in weights:
                pass
            logger.info(f"send_weights rank={self.group_local_rank}")
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
                get_torch_device().synchronize() # gpu ops must be finished before telling rollout our buffer is ready
                end = time.perf_counter()
                fill_bucket_dur = end-start
                print(f"filled bucket: {fill_bucket_dur} [s], bw: {(total_bytes / 1e9) / (fill_bucket_dur)} ")
                info = {
                    "bucket_meta": bucket_meta,
                    "len": offset,
                    "is_last": False,
                    # data that must be modified when sent to the next
                    "ptr": current.data_buffer.data_ptr(),
                    "door_ptr": current.doorbell_wb.data_ptr(),
                    "door_val": current.comp_wb.item(),
                }
                # send to target
                send_start = time.perf_counter()
                self.copy_and_send(idx % self.pipeline_stages, info)
                send_end = time.perf_counter()
                print(f"[idx={idx}][rank={self.group_local_rank}], send_time = {(send_end-send_start)*1000:.2f} ms")

                # next iter
                idx += 1
                current = self.buffer_manager.descs[idx % self.pipeline_stages]
                bucket_meta = {}
                offset = 0

                if (idx >= self.pipeline_stages):
                    await self.buffer_manager.wait_spin_wb(idx % self.pipeline_stages) # wait until rollout signals that it has consumed (read and yielded) our buffer

            assert offset + weight.nbytes <= self.bucket_size, (
                f"Weight {name}({weight.shape}, {weight.dtype}) is too large to fit in the bucket."
            ) # might change, but would require staging of weights between iterations

            bucket_meta[name] = {
                "name": name,
                "shape": weight.shape,
                "dtype": weight.dtype,
                "offset": offset,
            }
            # copy aysnc weights D2H, should not consume SM resources but GPU's async copy engines
            current.data_buffer[offset : offset + weight.nbytes].view(weight.dtype).copy_(weight.view(-1), non_blocking=True)
            offset += weight.nbytes
        
        # last bucket logic
        get_torch_device().synchronize()
        
        info = {
            "bucket_meta": bucket_meta,
            "ptr": current.data_buffer.data_ptr(),
            "len": offset,
            "door_ptr": current.doorbell_wb.data_ptr(),
            "door_val": current.comp_wb.item(),
            "is_last": True,
        }
        self.copy_and_send(idx % self.pipeline_stages, info)
        await self.buffer_manager.wait_spin_wb(idx % self.pipeline_stages)
        total_bytes += offset

        # finished send, print stats
        time_cost = time.time() - start_time
        bandwidth = total_bytes / time_cost / (1e9)
        logger.info(
            f"Rank {self.group_local_rank} send weights done, "
            f"total bytes: {total_bytes} time cost: {time_cost:.2f}s bandwidth: {bandwidth:.2f} GB/s"
        )

    @torch.no_grad()
    async def receive_weights(self) -> AsyncGenerator[tuple[str, torch.Tensor], None]:
        """Receive weights using Mooncake TransferEngine"""
        start_time = time.time()
        total_bytes = 0
        idx = 0
        current = self.buffer_manager.descs[idx]

        # prefetch
        info = None
        info_next = None

        while True:
            # 1 receive info from peer
            t0 = time.perf_counter()
            if (info == None):
                info = await self.recv_and_copy(idx % self.pipeline_stages)
            elif (info_next != None):
                info = info_next
                info_next = None
            t1 = time.perf_counter()
            print(f"[idx={idx}] rank: {self.group_local_rank}, recv in {(t1-t0) * 1000:.2f} ms for metadata")
            if (idx == 0):
                start_time = time.time()
            if idx >= self.pipeline_stages and self.sends: 
                await self.buffer_manager.wait_spin_wb(idx % self.pipeline_stages)
            preceding_address = self.recv_te_addr
            
            prev_buf_ptr = self.target_data_ptrs[idx % self.pipeline_stages]
            prev_door_val = self.wb_values[idx % self.pipeline_stages]
            prev_door_ptr = self.target_doorbell_wb_ptrs[idx % self.pipeline_stages]
            current.imm_scratch_wb[0] = prev_door_val + 1
            self.wb_values[idx % self.pipeline_stages] += 1 # IMPORTANT, increment or doesn't work

            start_transfer_time = time.perf_counter()
            track = self.engine.batch_transfer_async_read( 
                preceding_address,
                [current.data_buffer.data_ptr()],
                [prev_buf_ptr],
                [info["len"]]
            )
            assert track != 0, f"transfer failed!"
            # overlap:
            if (not info["is_last"]):
                # recv_start = time.perf_counter()
                track_prefetch = await self.submit_recv((idx + 1) % self.pipeline_stages)
                print("submittato")
                # info_next = await self.recv_and_copy((idx + 1) % self.pipeline_stages)
                # recv_dur = (time.perf_counter() - recv_start) * 1000
                # print(f"[idx={idx}][rank={self.group_local_rank}] recv_time={recv_dur:.2f} ms")

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
            bandwidth_this_step = info["len"] / time_cost_this_step / (1e9)
            print(f"RANK: {self.group_local_rank}, step {idx}: this step bandwidth: {bandwidth_this_step} GB/s, duration: {transfer_dur * 1000:.2f} ms")

            total_bytes += info["len"]
            # 2 tell the next rank that our buffer is ready to be read from
            if (self.sends):
                self.copy_and_send(idx % self.pipeline_stages, info)
            # 3 yield tensor from current buffer
            for name, meta in info["bucket_meta"].items():
                dtype, shape = meta["dtype"], meta["shape"]
                size = dtype.itemsize * shape.numel()
                tensor = current.data_buffer[meta["offset"] : meta["offset"] + size].view(dtype=dtype).view(shape)
                yield name, tensor

            # 4 tell send peer that we read from the buffer
            wb_start = time.perf_counter()
            ret = self.engine.transfer_sync_write(
                preceding_address, # target
                current.imm_scratch_wb.data_ptr(), 
                prev_door_ptr,
                4, # int32 is 4 bytes
            )
            wb_end = time.perf_counter()

            assert ret == 0, f"transfer_sync_write failed {ret}"
            print(f"[idx={idx}] rank: {self.group_local_rank}, wb_duration={(wb_end-wb_start) * 1000:.2f} ms")
            get_torch_device().synchronize()
            # 5 swap buffer
            idx += 1
            current = self.buffer_manager.descs[idx % self.pipeline_stages]
            if (not info["is_last"]):
                info_next = self.sync_with_recv(idx % self.pipeline_stages, track_prefetch)

            if info["is_last"]:
                break
        
        # print stats
        time_cost = time.time() - start_time
        bandwidth = total_bytes / time_cost / (1e9)
        logger.info(
            f"Rank {self.group_local_rank} receive weights done, time cost: {time_cost:.2f}s, bandwidth: {bandwidth:.2f} GB/s"
        )
