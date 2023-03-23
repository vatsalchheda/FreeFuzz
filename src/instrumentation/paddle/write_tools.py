import pymongo

"""
You should configure the database
"""
paddle_db = pymongo.MongoClient(host="localhost", port=27017)["freefuzz-paddle"]

def write_fn(func_name, params, input_signature, output_signature):
    params = dict(params)
    out_fname = "paddle." + func_name
    params['input_signature'] = input_signature
    params['output_signature'] = output_signature
    paddle_db[out_fname].insert_one(params)