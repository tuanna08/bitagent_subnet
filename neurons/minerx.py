import time
import importlib
import bittensor as bt
from rich.console import Console
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
        """
        Processes the incoming BitAgent synapse and returns response.

        Args:
            synapse (bitagent.protocol.QnATask): The synapse object containing the urls and prompt.

        Returns:
            bitagent.protocol.QnATask: The synapse object with the 'response' field set to the generated response and citations
        """
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

    async def __blacklist(self, synapse: bt.Synapse) -> Tuple[bool, str]:
        """
        Determines whether an incoming request should be blacklisted and thus ignored. Your implementation should
        define the logic for blacklisting requests based on your needs and desired security parameters.

        Blacklist runs before the synapse data has been deserialized (i.e. before synapse.data is available).
        The synapse is instead contructed via the headers of the request. It is important to blacklist
        requests before they are deserialized to avoid wasting resources on requests that will be ignored.

        Args:
            synapse (bitagent.protocol.QnATask): A synapse object constructed from the headers of the incoming request.

        Returns:
            Tuple[bool, str]: A tuple containing a boolean indicating whether the synapse's hotkey is blacklisted,
                            and a string providing the reason for the decision.

        This function is a security measure to prevent resource wastage on undesired requests. It should be enhanced
        to include checks against the metagraph for entity registration, validator status, and sufficient stake
        before deserialization of synapse data to minimize processing overhead.

        Example blacklist logic:
        - Reject if the hotkey is not a registered entity within the metagraph.
        - Consider blacklisting entities that are not validators or have insufficient stake.

        In practice it would be wise to blacklist requests from entities that are not validators, or do not have
        enough stake. This can be checked via metagraph.S and metagraph.validator_permit. You can always attain
        the uid of the sender via a metagraph.hotkeys.index( synapse.dendrite.hotkey ) call.

        Otherwise, allow the request to be processed further.
        """

        # Check if the key has validator permit
        if self.config.blacklist.force_validator_permit:
            if synapse.dendrite.hotkey in self.metagraph.hotkeys:
                uid = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)
                if not self.metagraph.validator_permit[uid]:
                    return True, "validator permit required"
            else:
                return True, "validator permit required, but hotkey not registered"

        # TODO(developer): Define how miners should blacklist requests.
        if synapse.dendrite.hotkey not in self.metagraph.hotkeys:
            # Ignore requests from unrecognized entities.
            bt.logging.trace(
                f"Blacklisting unrecognized hotkey {synapse.dendrite.hotkey}"
            )
            return True, "Unrecognized hotkey"

        bt.logging.trace(
            f"Not Blacklisting recognized hotkey {synapse.dendrite.hotkey}"
        )
        return False, "Hotkey recognized!"

    async def blacklist_for_task(self, synapse: bitagent.protocol.QnATask) -> Tuple[bool, str]:
        return await self.__blacklist(synapse)

    async def blacklist_for_result(self, synapse: bitagent.protocol.QnAResult) -> Tuple[bool, str]:
        return await self.__blacklist(synapse)

    async def blacklist_for_alive(self, synapse: bitagent.protocol.IsAlive) -> Tuple[bool, str]:
        return await self.__blacklist(synapse)

    async def __priority(self, synapse: bt.Synapse) -> float:
        """
        The priority function determines the order in which requests are handled. More valuable or higher-priority
        requests are processed before others. You should design your own priority mechanism with care.

        This implementation assigns priority to incoming requests based on the calling entity's stake in the metagraph.

        Args:
            synapse (bitagent.protocol.QnATask): The synapse object that contains metadata about the incoming request.

        Returns:
            float: A priority score derived from the stake of the calling entity.

        Miners may recieve messages from multiple entities at once. This function determines which request should be
        processed first. Higher values indicate that the request should be processed first. Lower values indicate
        that the request should be processed later.

        Example priority logic:
        - A higher stake results in a higher priority value.
        """
        # TODO(developer): Define how miners should prioritize requests.
        caller_uid = self.metagraph.hotkeys.index(
            synapse.dendrite.hotkey
        )  # Get the caller index.
        prirority = float(
            self.metagraph.S[caller_uid]
        )  # Return the stake as the priority.
        bt.logging.trace(
            f"Prioritizing {synapse.dendrite.hotkey} with value: ", prirority
        )
        return prirority

    async def priority_for_task(self, synapse: bitagent.protocol.QnATask) -> float:
        return await self.__priority(synapse)

    async def priority_for_result(self, synapse: bitagent.protocol.QnAResult) -> float:
        return await self.__priority(synapse)

    async def priority_for_alive(self, synapse: bitagent.protocol.IsAlive) -> float:
        return await self.__priority(synapse)

    async def forward(self, synapse: bt.Synapse) -> bt.Synapse:
        # not being used but required by ABC
        pass

    # no idea what to save for a miner
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
