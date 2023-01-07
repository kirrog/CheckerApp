from src.structs.request_structs import CheckerRequestFormat, ChangeFormats


def check_and_transform_orthography(req: CheckerRequestFormat) -> CheckerRequestFormat:
    req.list_of_changes.append(ChangeFormats("нет", "да", "потомучто"))
    return req
