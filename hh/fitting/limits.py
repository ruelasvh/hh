import json
import pyhf
import cabinetry


def calculate_limits(model, data):
    print("Calculating limits")
    # breakpoint()
    limit_results = cabinetry.fit.limit(model, data)
    return limit_results
