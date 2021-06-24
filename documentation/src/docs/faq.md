# Frequently asked questions


## Getting a Thread error when using a TFRecordDataset

When using the TFdatasetsampler, Tensorflow might throw a thread errors. It is usually because you use too many shards, try to cap the number using the `shards_per_cycle` option of `TFRecordDatasetSampler` init.