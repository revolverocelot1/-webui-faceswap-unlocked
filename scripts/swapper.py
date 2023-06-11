from dataclasses import dataclass
import os
from typing import List
import cv2
import insightface
import onnxruntime
import numpy as np

from PIL import Image, ImageFont, ImageDraw, PngImagePlugin
from scripts.faceswap_logging import logger
from modules.upscaler import Upscaler, UpscalerData
from modules.face_restoration import restore_faces
from modules import scripts, shared, images,  scripts_postprocessing
from modules.face_restoration import FaceRestoration

@dataclass
class UpscaleOptions :
    scale : int = 1
    upscaler : UpscalerData = None
    upscale_visibility : float = 0.5
    face_restorer : FaceRestoration  = None
    restorer_visibility : float = 0.5

def upscale_image(image: Image, upscale_options: UpscaleOptions):
    result_image = image
    
    if(upscale_options.upscaler is not None and upscale_options.upscaler.name != "None") :
        original_image = result_image.copy()
        logger.info("Upscale with %s scale = %s", upscale_options.upscaler.name, upscale_options.scale)
        result_image = upscale_options.upscaler.scaler.upscale(image, upscale_options.scale, upscale_options.upscaler.data_path)
        if upscale_options.scale == 1 :
            result_image = Image.blend(original_image, result_image, upscale_options.upscale_visibility)

    if(upscale_options.face_restorer is not None) :
        original_image = result_image.copy()
        logger.info("Restore face with %s", upscale_options.face_restorer.name())
        numpy_image = np.array(result_image)
        numpy_image = upscale_options.face_restorer.restore(numpy_image)
        restored_image = Image.fromarray(numpy_image)
        result_image = Image.blend(original_image, restored_image, upscale_options.restorer_visibility)

    return result_image


providers = onnxruntime.get_available_providers()
if "TensorrtExecutionProvider" in providers:
    providers.remove("TensorrtExecutionProvider")


def get_face_single(img_data, face_index=0, det_size=(640, 640)):
    face_analyser = insightface.app.FaceAnalysis(name="buffalo_l", providers=providers)
    face_analyser.prepare(ctx_id=0, det_size=det_size)
    face = face_analyser.get(img_data)
    
    if len(face) == 0 and det_size[0] > 320 and det_size[1] > 320:
        det_size_half = (det_size[0] // 2, det_size[1] // 2)
        return get_face_single(img_data, face_index=face_index, det_size=det_size_half)
    
    try:
        return sorted(face, key=lambda x: x.bbox[0])[face_index]
    except IndexError:
        return None




def swap_face(
    source_img: Image, target_img: Image, model: str = "../models/inswapper_128.onnx", faces_index: List[int] = [0], upscale_options: UpscaleOptions = None
) -> Image:
    source_img = cv2.cvtColor(np.array(source_img), cv2.COLOR_RGB2BGR)
    target_img = cv2.cvtColor(np.array(target_img), cv2.COLOR_RGB2BGR)
    source_face = get_face_single(source_img, face_index=0)
    if source_face is not None:
        result = target_img
        model_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), model)
        face_swapper = insightface.model_zoo.get_model(
            model_path, providers=providers
        )
        for face_num in faces_index:
            target_face = get_face_single(target_img, face_index=face_num)
            if target_face is not None:
                result = face_swapper.get(
                    result, target_face, source_face, paste_back=True
                )
            else:
                logger.info(f"No target face found")

        result_image = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))

        result_image = upscale_image(result_image, upscale_options)
        
        return result_image
    else:
        logger.info(f"No source face found")

    return None
