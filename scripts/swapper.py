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

def upscale_image(image: Image, upscaler: UpscalerData):
    result_image = upscaler.scaler.upscale(image, 1, upscaler.data_path)
    numpy_image = np.array(result_image)
    numpy_image = shared.face_restorers[0].restore(numpy_image)
    result_image = Image.fromarray(numpy_image)
    logger.info("Upscale and restore face in result image with %s and %s", upscaler.name, shared.face_restorers[0].name())
    return result_image


def swap_face(
    source_img: Image, target_img: Image, model: str = "../models/inswapper_128.onnx", faces_index: List[int] = [0], upscaler: UpscalerData = None
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

        if upscaler is not None:
            result_image = upscale_image(result_image, upscaler)
        
        return result_image
    else:
        logger.info(f"No source face found")

    return None
