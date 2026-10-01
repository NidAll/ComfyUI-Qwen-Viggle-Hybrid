from comfy_api.latest import ComfyExtension, io

from .nodes import QwenViggleHybridSampler


class QwenViggleExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [QwenViggleHybridSampler]


async def comfy_entrypoint() -> QwenViggleExtension:
    return QwenViggleExtension()
