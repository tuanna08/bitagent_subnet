import time
import importlib
from typing import Tuple
import bittensor as bt
from rich.console import Console
import bitagent
import httpx  # Added import for httpx

import time
import glob
import argparse
import importlib
from typing import List, Tuple
import bittensor as bt
from rich.console import Console

# Bittensor Miner Template:
import bitagent
# Sync calls set weights and also resyncs the metagraph.
from common.utils.config import add_args as util_add_args
from common.utils.config import config as util_config


# import base miner class which takes care of most of the boilerplate
from common.base.miner import BaseMinerNeuron
rich_console = Console()

rich_console = Console()

class Miner(bitagent.BaseNeuron):

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser):
        util_add_args(cls, parser)
        parser.add_argument(
            "--miner",
            type=str,
            default="t5",
            help="Miner to load. Default choices are 't5' and 'mock'.  Pass your custom miner name as appropriate."
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
        """
        Processes the incoming BitAgent synapse and returns response.

        Args:
            synapse (bitagent.protocol.QnATask): The synapse object containing the urls and prompt.

        Returns:
            bitagent.protocol.QnATask: The synapse object with the 'response' field set to the generated response and citations
        """
        # Make an HTTP request to the local FastAPI worker
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

    # ... (Other methods remain unchanged)

# This is the main function, which runs the miner.
if __name__ == "__main__":
    with Miner() as miner:
        while True:
            bt.logging.info("Miner running...", time.time())
            time.sleep(15)
