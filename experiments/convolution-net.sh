#!/bin/bash

# Create dir where logs and model snapshots are stored
mkdir -p snapshots

# # Convert data to iobes
cat data/conll2003/eng.testa.dev | python utils/iob-iobes.py false iob | python utils/iob-iobes.py false iobes > data/conll2003/eng.testa.dev.iobes
cat data/conll2003/eng.testb.test | python utils/iob-iobes.py false iob | python utils/iob-iobes.py false iobes > data/conll2003/eng.testb.test.iobes
cat data/conll2003/eng.train | python utils/iob-iobes.py false iob | python utils/iob-iobes.py false iobes > data/conll2003/eng.train.iobes

# # Create index from data
python utils/data-indexer.py sentence-convolution

# Convert index to torch tensors
th utils/index-to-torch-tensors-converter.lua -inFile data/embeddings/senna.index -len 50 -outFile data/embeddings/senna.torch -mode win
th utils/index-to-torch-tensors-converter.lua -inFile data/conll2003/eng.testb.test.iobes.index -outFile data/conll2003/eng.testb.test.iobes.torch -mode sentence-convolution
th utils/index-to-torch-tensors-converter.lua -inFile data/conll2003/eng.testa.dev.iobes.index -outFile data/conll2003/eng.testa.dev.iobes.torch -mode sentence-convolution
th utils/index-to-torch-tensors-converter.lua -inFile data/conll2003/eng.train.iobes.index  -outFile data/conll2003/eng.train.iobes.torch -mode sentence-convolution -batchedMode

# Run learning
th conll_learn.lua -version 1-convolution-sentence-net-fixed -mode sent -cuda -model_name convolution-full
