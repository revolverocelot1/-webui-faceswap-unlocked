import gradio as gr
import modules.scripts as scripts
from modules import scripts, scripts_postprocessing
from modules.processing import StableDiffusionProcessing, StableDiffusionProcessingImg2Img
from modules.shared import cmd_opts, opts, state
from PIL import Image

from scripts.faceswap_logging import logger
from scripts.swapper import swap_face
from scripts.faceswap_version import version_flag


class FaceSwapScript(scripts.Script):
    def title(self):
        return f"FaceSwap"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with gr.Accordion(f"Face Swap {version_flag}", open=False):
            with gr.Column():
                img = gr.inputs.Image(type="pil")
                enable = gr.Checkbox(
                    False, placeholder="enable", label="Enable"
                )
                faces_index = gr.Textbox(
                    value="0",
                    placeholder="Which face to swap (comma separated), start from 0",
                    label="Comma separated face number(s)",
                )

                swap_in_source = gr.Checkbox(
                    False, placeholder="Swap face in source image", label="Swap in source image", visible=is_img2img
                )
                swap_in_generated = gr.Checkbox(
                    True, placeholder="Swap face in generated image", label="Swap in generated image", visible=is_img2img
                )

        return [img, enable, faces_index, swap_in_source, swap_in_generated]

    def process(self, p: StableDiffusionProcessing, img, enable, faces_index, swap_in_source, swap_in_generated):
        self.source = img
        self.enable = enable
        self.swap_in_generated = swap_in_generated
        self.faces_index = {int(x) for x in faces_index.strip(",").split(",") if x.isnumeric()}
        if len(self.faces_index) == 0 :
            self.faces_index = [0]
        if self.enable:
            if self.source is not None:
                if isinstance(p,StableDiffusionProcessingImg2Img) and swap_in_source:
                    for i in range(len(p.init_images)) :
                        p.init_images[i] = swap_face(self.source, p.init_images[i], faces_index = self.faces_index)

                logger.info(f"FaceSwap enabled, face index %s", self.faces_index)
            else :
                logger.error(f"Please provide a source face")

    def postprocess_image(self, p, script_pp: scripts.PostprocessImageArgs, *args):
        if self.enable and self.swap_in_generated:
            if self.source is not None:
                image: Image.Image = script_pp.image
                result = swap_face(self.source, image, faces_index = self.faces_index)
                pp = scripts_postprocessing.PostprocessedImage(result)
                pp.info = {}
                p.extra_generation_params.update(pp.info)
                script_pp.image = pp.image
