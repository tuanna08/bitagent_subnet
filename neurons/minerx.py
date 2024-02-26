import time
import argparse
import importlib
from typing import List, Tuple
import bittensor as bt
from rich.console import Console
import httpx  # Import the httpx library for making HTTP requests

# Bittensor Miner Template:
import bitagent
from common.utils.config import add_args as util_add_args
from common.utils.config import config as util_config

# Import the BaseMinerNeuron class
from common.base.miner import BaseMinerNeuron
rich_console = Console()

class Miner(BaseMinerNeuron):
    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser):
        util_add_args(cls, parser)
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
            config = util_config(self)

        super(Miner, self).__init__(config=config)

        # Dynamic module import based on the 'miner' argument
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

        # Update the synapse with the response from the worker
        synapse.response["response"] = response.json()["result"]
        synapse.response["citations"] = []

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

    # ... (Remaining code remains unchanged)

# This is the main function, which runs the miner.
if __name__ == "__main__":
    with Miner() as miner:
        while True:
            bt.logging.info("Miner running...", time.time())
            time.sleep(15)
