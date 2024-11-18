from io import BytesIO

import numpy as np
from PIL import Image
from fastapi import APIRouter, UploadFile, Depends

from app.model import get_model, Model
from app.schemas.response import ResponseSchemas, Classification

router = APIRouter()


@router.post("/degradai")
async def infer(
    image: UploadFile,
    model: Model = Depends(get_model),
) -> ResponseSchemas:
    file = await image.read()
    image_as_array = np.array(Image.open(BytesIO(file)))

    predictions, label = model.predict(image_as_array)
    label = Classification(label)
    index = np.argmax(predictions)
    mention = None

    if label == Classification.DEGRADANT:
        if predictions[index] > 0.95:
            mention = "ABERRANT"

    return ResponseSchemas(
        classname=label,
        probability=predictions[index],
        mention=mention
    )