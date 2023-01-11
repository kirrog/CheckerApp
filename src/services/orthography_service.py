from src.orthog.cache import ModelCache
from src.services.differ_calc_service import find_changes
from src.structs.request_structs import CheckerRequestFormat


def check_and_transform_orthography(req: CheckerRequestFormat) -> CheckerRequestFormat:
    return req
    # print("Loading punct-cache model")
    # model = ModelCache().model
    # data = req.text
    # print(f"Apply to {data}")
    # output_data = model(data)
    # print(f"Result is {output_data}")
    # l_of_changes = find_changes(req.text, output_data)
    # print(f"Len list of changes: {len(l_of_changes)}")
    # req.list_of_changes.extend(l_of_changes)
    # req.text = output_data
    # return req
