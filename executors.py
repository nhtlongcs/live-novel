import os
import configparser

config = configparser.ConfigParser()
config.read('executor_cfg.ini')
# set os environment
os.environ['NOVEL_CORE_MODULE'] = config['DEFAULT']['CORE_MODULE']
os.environ['NOVEL_OUTPUTS_DIR'] = config['DEFAULT']['OUTPUTS_DIR']
os.makedirs(os.environ['NOVEL_OUTPUTS_DIR'], exist_ok=True)

import asyncio
from typing import Dict

from jina import Document, DocumentArray, Executor, requests

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
        sess_name = str(parameters['sess_name'])
        da_save_path = os.path.join(os.environ.get('NOVEL_OUTPUTS_DIR', './'), f"{sess_name}.protobuf.lz4")
        da = DocumentArray.load_binary(da_save_path)
        return da
    
# https://docs.jina.ai/fundamentals/executor/executor-serve/
# StableAIExecutor.serve(port=12345)
