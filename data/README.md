# Hetionet source and two-scale cache (v7)

The repository does **not** distribute the 148 MB original Hetionet ZIP.

To use your own ZIP, copy `hetionet-v1.0.0.zip` into `data/archives/`.
The `study` Compose job extracts only `hetionet-v1.0-nodes.tsv` and
`hetionet-v1.0-edges.sif.gz`, verifies their headers and checksum, and saves
both under `data/raw/`. Missing source data is fetched from the verified
Hetionet official media/raw endpoints instead.

Independently sampled outputs are stored under `data/samples/10000/` and
`data/samples/100000/`. They contain triples, original node names/kinds and
metadata. They are not medical data about patients.

The standard API demo still reads `data/hetionet_demo/` and is independent of
the `study` experiment artifacts.
