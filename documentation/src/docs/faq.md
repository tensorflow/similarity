# Frequently asked questions


## Getting a Thread error when using a TFRecordDataset

When using the TFdatasetsampler, Tensorflow might throw a thread errors `(Check failed: ret == 0 (11 vs. 0)Thread creation via pthread_create() failed..)`. This is usually
because you use too many shards at once, try to cap the number using
the `shards_per_cycle` option of `TFRecordDatasetSampler` init.
