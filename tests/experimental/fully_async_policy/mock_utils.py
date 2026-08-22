from verl.experimental.fully_async_policy.fully_async_trainer import FullyAsyncTrainer
from verl.experimental.fully_async_policy.fully_async_rollouter import FullyAsyncRollouter
from verl.experimental.fully_async_policy.fully_async_main import FullyAsyncTaskRunner, validate_cluster
from verl.experimental.separation.ray_trainer import SeparateRayPPOTrainer
import ray
from verl.trainer.ppo.utils import Role
from verl.single_controller.ray.base import RayResourcePool, RayWorkerGroup

from verl.experimental.separation.utils import (
    create_resource_pool_manager,
    create_role_worker_mapping,
)

@ray.remote(num_cpus=1, label_selector={"actor": "true", "trainer_head": "true"})
class MockFullyAsyncTrainer(FullyAsyncTrainer):

    def __init__(self, config, tokenizer, role_worker_mapping, resource_pool_manager, ray_worker_group_cls, device_name=None):
        super().__init__(config, tokenizer, role_worker_mapping, resource_pool_manager, ray_worker_group_cls, device_name)

    def _init_models(self):
        return None

    async def get_actor_wg(self):
        return self.all_wg[str(self.train_role)]
    
    async def set_rollouter(self, rollouter):
        self.rollouter = rollouter


@ray.remote(num_cpus=1, label_selector={"rollout": "true", "rollout_head": "true"})
class MockFullyAsyncRollouter(FullyAsyncRollouter):

    def __init__(self, config, tokenizer, processor=None, device_name=None):
        pass

    def _init_models(self):
        return None

    def init_workers(self):
        return None

class MockFullyAsyncTaskRunner(FullyAsyncTaskRunner):

    def run(self, config):
        print("[ASYNC MAIN] Starting fully async PPO training...")
        self.cluster_config = validate_cluster(config)
        self._resolve_nodes()
        self._initialize_components(config)
        

    def _initialize_components(self, config) -> None:

        print("[ASYNC MAIN] Creating worker mapping and resource pools...")
        role_worker_mapping, ray_worker_group_cls = create_role_worker_mapping(config)
        self.components["role_worker_mapping"] = role_worker_mapping
        self.components["ray_worker_group_cls"] = ray_worker_group_cls

        self._create_trainer_rollouter(config)

        print("[ASYNC MAIN] Setting up rollouter reference on trainer")
        ray.get(
            self.components["trainer"].set_rollouter.remote(
                self.components["rollouter"]
            )
        )


    def _create_trainer_rollouter(self, config):
        print("[ASYNC MAIN] Starting create trainer and rollouter...")
        rollouter = MockFullyAsyncRollouter.remote(
            config=config,
            tokenizer=None,
            processor=None,
            device_name=config.trainer.device,
        )
        trainer_role_mapping = {
            role: worker_cls
            for role, worker_cls in self.components["role_worker_mapping"].items()
            if role != Role.Rollout
        }

        trainer = MockFullyAsyncTrainer.remote(
            config=config,
            tokenizer=None,
            role_worker_mapping=trainer_role_mapping,
            resource_pool_manager=create_resource_pool_manager(
                config, roles=list(trainer_role_mapping.keys())
            ),
            ray_worker_group_cls=self.components["ray_worker_group_cls"],
            device_name=config.trainer.device,
        )

        ray.get([trainer.init_workers.remote(), rollouter.init_workers.remote()])

        self.components["rollouter"] = rollouter
        self.components["trainer"] = trainer

        print("[ASYNC MAIN] Trainer and Rollouter created and initialized successfully")
