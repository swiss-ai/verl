import torch
import pickle
from mooncake.engine import TransferEngine
from dataclasses import dataclass
import time

@dataclass
class BufferDesc:
    data_buffer: torch.Tensor
    metadata_buffer: torch.Tensor
    doorbell_wb: torch.Tensor
    comp_wb: torch.Tensor
    doorbell_md: torch.Tensor
    comp_md: torch.Tensor
    imm_scratch_wb: torch.Tensor
    imm_scratch_md: torch.Tensor

class BufferManager:
    
    def __init__(self, 
                 pipeline_stages: int, 
                 buffer_size: int,
                 engine: TransferEngine,
                 buffer_device: str,
                 metadata_buffer_size: int,
                 metadata_rdma: bool = True):
        self.engine = engine
        self.pipeline_stages = pipeline_stages
        self.buffer_device = buffer_device
        self.metadata_buffer_size = metadata_buffer_size
        self.buffer_size = buffer_size
        self.metadata_rdma = metadata_rdma

    def initialize_buffers(self):
        if (self.buffer_device == "cpu"):
            self.buffer_allocation = torch.zeros(self.buffer_size * self.pipeline_stages, 
                                                 dtype = torch.uint8, 
                                                 device = "cpu").pin_memory()
        else:
            self.buffer_allocation = torch.zeros(self.buffer_size * self.pipeline_stages,
                                                 dtype = torch.uint8,
                                                 device=self.buffer_device) 
        
        self.metadata_buffer_alloc = torch.zeros(self.metadata_buffer_size * self.pipeline_stages, 
                                                 dtype = torch.uint8,
                                                 device = "cpu") # no pin, doesn't interact with cuda
        # two doorbells per pipeline stage
        self.doorbells_alloc = torch.zeros(self.pipeline_stages * 4, dtype=torch.int32, device = "cpu")
        self.local_comparator_alloc = torch.zeros(self.pipeline_stages * 2, dtype=torch.int32, device = "cpu") # not registered by fabric

        self.doorbells_wb = self.doorbells_alloc[0 : self.pipeline_stages]
        self.doorbells_md = self.doorbells_alloc[self.pipeline_stages : 2 * self.pipeline_stages]
        self.comps_wb = self.local_comparator_alloc[0 : self.pipeline_stages]
        self.comps_md = self.local_comparator_alloc[self.pipeline_stages : 2 * self.pipeline_stages]
        self.imm_scratch_wb = self.doorbells_alloc[2 * self.pipeline_stages : 3 * self.pipeline_stages]
        self.imm_scratch_md = self.doorbells_alloc[3 * self.pipeline_stages : 4 * self.pipeline_stages]

        mr_ptrs = [self.buffer_allocation.data_ptr(), self.metadata_buffer_alloc.data_ptr(), self.doorbells_alloc.data_ptr()]
        mr_sizes_bytes = [self.buffer_allocation.nbytes, self.metadata_buffer_alloc.nbytes, self.doorbells_alloc.nbytes]

        ret = self.engine.batch_register_memory(
            mr_ptrs,
            mr_sizes_bytes
        )
        if (ret != 0):
            raise RuntimeError(f"Failed to register buffers! Got error={ret}")
        
        self.descs = [
            BufferDesc(
                data_buffer=self.buffer_allocation[self.buffer_size * idx : self.buffer_size * (idx + 1)],
                metadata_buffer=self.metadata_buffer_alloc[self.metadata_buffer_size * idx : self.metadata_buffer_size * (idx + 1)],
                doorbell_wb=self.doorbells_wb[idx : idx + 1],
                comp_wb=self.comps_wb[idx : idx + 1],
                doorbell_md=self.doorbells_md[idx : idx + 1],
                comp_md=self.comps_md[idx : idx + 1],
                imm_scratch_wb=self.imm_scratch_wb[idx : idx + 1],
                imm_scratch_md=self.imm_scratch_md[idx : idx + 1]
            )
            for idx in range(self.pipeline_stages)
        ]

    def reset_counters(self):
        self.doorbells_wb.zero_()
        self.doorbells_md.zero_()
        self.comps_wb.zero_()
        self.comps_md.zero_()
        self.imm_scratch_wb.zero_()
        self.imm_scratch_md.zero_()

    def copy_obj_to_buffer(self, pipeline_stage: int, obj: object):
        pickled = pickle.dumps(obj)
        sz = len(pickled)
        assert sz <= self.metadata_buffer_size
        self.descs[pipeline_stage].metadata_buffer[0:sz].copy_(torch.frombuffer(pickled, dtype=torch.uint8))
    
    def get_obj_from_buffer(self, pipeline_stage: int):
        return pickle.loads(bytes(self.descs[pipeline_stage].metadata_buffer))

    async def wait_spin_wb(self, pipeline_stage: int):
        idx = pipeline_stage
        while True:
            if (self.doorbells_alloc[idx] == self.local_comparator_alloc[idx] + 1):
                self.local_comparator_alloc[idx] += 1
                break
            time.sleep(0)
    
    async def wait_spin_md(self, pipeline_stage: int):
        idx = self.pipeline_stages + pipeline_stage
        print(f"spinning on metadata for stage: {pipeline_stage}, curr_value: {self.local_comparator_alloc[idx]}")
        while True:
            if (self.doorbells_md[pipeline_stage] == self.local_comparator_alloc[idx] + 1):
                self.local_comparator_alloc[idx] += 1
                break
            time.sleep(0)
    