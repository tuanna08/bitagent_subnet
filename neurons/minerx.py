import time
import importlib
import bittensor as bt
from rich.console import Console
from typing import List, Tuple
import bitagent
import httpx

rich_console = Console()

class Miner(bitagent.BaseNeuron):

    @classmethod
    def add_args(cls, parser: bt.parser.ArgumentParser):
        parser.add_argument(
            "--miner",
            type=str,
            default="t5",
            help="Miner to load. Default choices are 't5' and 'mock'. Pass your custom miner name as appropriate."
        )

    def __init__(self, config=None):
        self.forward_capabilities = [
            {'forward': self.forward_for_task, 'blacklist': self.blacklist_for_task, 'priority': self.priority_for_task},
            {'forward': self.forward_for_result, 'blacklist': self.blacklist_for_result, 'priority': self.priority_for_result},
            {'forward': self.forward_for_alive, 'blacklist': self.blacklist_for_alive, 'priority': self.priority_for_alive},
        ]
        if not config:
            config = self.default_config()

        super(Miner, self).__init__(config=config)

        miner_name = f"bitagent.miners.{config.miner}_miner"
        miner_module = importlib.import_module(miner_name)

        self.miner_init = miner_module.miner_init
        self.miner_process = miner_module.miner_process

        self.miner_init(self)

    async def forward_for_task(
        self, synapse: bitagent.protocol.QnATask
    ) -> bitagent.protocol.QnATask:
        # Make an HTTP request to the FastAPI worker
        async with httpx.AsyncClient() as client:
            response = await client.post("http://127.0.0.1:8080/", json={"input_text": synapse.prompt})
            llm_response = response.json()["result"]

        synapse.response["response"] = llm_response
        return synapse

    async def forward_for_result(
        self, synapse: bitagent.protocol.QnAResult
    ) -> bitagent.protocol.QnAResult:
        if self.config.logging.debug:
            rich_console.print(synapse.results)
        return synapse

    async def forward_for_alive(
        self, synapse: bitagent.protocol.IsAlive
    ) -> bitagent.protocol.IsAlive:
        synapse.response = True
        return synapse

    async def forward(self, synapse: bt.Synapse) -> bt.Synapse:
        # Implement the forward method logic here
        pass

    async def __blacklist(self, synapse: bt.Synapse) -> Tuple[bool, str]:
        # Blacklist logic here
        pass

    async def __priority(self, synapse: bt.Synapse) -> float:
        # Priority logic here
        pass

    async def blacklist_for_task(self, synapse: bitagent.protocol.QnATask) -> Tuple[bool, str]:
        return await self.__blacklist(synapse)

    async def blacklist_for_result(self, synapse: bitagent.protocol.QnAResult) -> Tuple[bool, str]:
        return await self.__blacklist(synapse)

    async def blacklist_for_alive(self, synapse: bitagent.protocol.IsAlive) -> Tuple[bool, str]:
        return await self.__blacklist(synapse)

    async def priority_for_task(self, synapse: bitagent.protocol.QnATask) -> float:
        return await self.__priority(synapse)

    async def priority_for_result(self, synapse: bitagent.protocol.QnAResult) -> float:
        return await self.__priority(synapse)

    async def priority_for_alive(self, synapse: bitagent.protocol.IsAlive) -> float:
        return await self.__priority(synapse)

    def save_state(self):
        pass

    def load_state(self):
        pass

# Create a synapse and make a POST request
if __name__ == "__main__":
    with Miner() as miner:
        synapse = bitagent.protocol.QnATask(prompt="Hello, what's the weather like today?")
        response = miner.forward(synapse)
        print(response)
