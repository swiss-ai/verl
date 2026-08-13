import pyarrow.parquet as pq
from verl import DataProto
import numpy as np
import pyarrow.compute as pc
import pyarrow as pa

if __name__ == "__main__":

    path = "/capstor/scratch/cscs/atazza/data_rl/val.parquet"
    group_batches = {}
    group_counts = {}
    sources = set()

    group_col = "data_source"
    n_samples = 8

    schema = pq.read_schema(path)
    column_names = schema.names

    print(column_names)

    train_ds = pq.ParquetFile(path)
    
    for batch in train_ds.iter_batches(batch_size=8192):
        filtered = batch.filter(
            pc.equal(batch.column(group_col), "openai/gsm8k")
        )
        
        if len(filtered) > 0:
            # Access the first row as a dictionary
            sample_row = filtered.slice(0, 1).to_pylist()[0]
            # print(sample_row["extra_info"])  # just that field
            print(sample_row)                # whole row as dict
            # sample_prompt = filtered.column("extra_info")[0].to_pylist()
            # print("=== Sample Prompt ===")
            # print(sample_prompt)
            break  # Stop processing once you've found a sample
    #     print(unique_sources)
    #     for source_val in unique_sources:
    #         if source_val in sources and group_counts[source_val] >= n_samples:
    #             continue
            
    #         mask = pc.equal(batch.column(group_col), source_val)
    #         filtered_batch = batch.filter(mask)

    #         gc = 0 if source_val not in sources else group_counts[source_val]
    #         needed = n_samples - gc
    #         sliced_batch = filtered_batch.slice(0, needed)
    #         if (source_val in sources):
    #             group_batches[source_val].append(sliced_batch)
    #             group_counts[source_val] += len(sliced_batch)
    #         else:
    #             group_batches[source_val] = [sliced_batch]
    #             group_counts[source_val] = len(sliced_batch)
    #             sources.add(source_val)
    # filtered = train_ds.filter(
    #     pc.equal(train_ds.column("data_source"), "openai/gsm8k")
    # )
    # print(filtered)

    # result = [b for batches in group_batches.values() for b in batches]
    # for b in result:
    #     for x in b:
    #         x = x.to_pydict()
    #         print(x["data_source"])
