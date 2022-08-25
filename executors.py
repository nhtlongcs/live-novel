import asyncio
import os
from typing import Dict

from jina import Executor, requests, DocumentArray, Document
from create_fn import create

class StableAIExecutor(Executor):
    skip_event = asyncio.Event()
    stop_event = asyncio.Event()

    @requests(on='/create')
    async def create_artworks(self, parameters: Dict, **kwargs):
        await asyncio.get_event_loop().run_in_executor(
            None, self._create, parameters
        )

    def _create(self, parameters: Dict, **kwargs):
        return create(
            skip_event=self.skip_event, # TODO: this is not working
            stop_event=self.stop_event, # TODO: this is not working
            **parameters
        )

    @requests(on='/skip')
    async def skip_create(self, **kwargs):
        self.skip_event.set()

    @requests(on='/stop')
    async def stop_create(self, **kwargs):
        self.stop_event.set()


class ResultPoller(Executor):
    @requests(on='/result')
    def poll_results(self, parameters: Dict, **kwargs):
        print(parameters['sess_name'])
        output_dir = 'outputs/samples' # hardcoded for now
        path = os.path.join(output_dir, parameters['sess_name'] + '.protobuf.lz4')
        da = DocumentArray.load_binary(path)
        return da
    
# https://docs.jina.ai/fundamentals/executor/executor-serve/
# StableAIExecutor.serve(port=12345)
