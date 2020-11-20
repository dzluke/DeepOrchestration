import parameters
from OrchDataset_2 import RawDatabase

print("Generating raw database")
rdb = RawDatabase('../rdb.pkl',
                  random_granularity=parameters.GLOBAL_PARAMS.rdm_granularity,
                  instr_filter=parameters.GLOBAL_PARAMS.instr_filter)
print("Saving raw database")
rdb.save()
print("Done")