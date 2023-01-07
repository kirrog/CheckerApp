from src.services.orthography_service import check_and_transform_orthography
from src.services.punctuation_service import check_and_transform_punctuation
from src.structs.request_structs import CheckerRequestFormat


def check_and_transform(req: CheckerRequestFormat):
    req = check_and_transform_punctuation(req)
    req = check_and_transform_orthography(req)
    req.filled = True
    return req
