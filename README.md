# Genome-Reconstruction-GNN
The aim of this project is to train a GNN to reconstruct genomes. This repository will be frequently updated as the project advances. 

## The pipeline visualized

flowchart TD
    subgraph Optimize
      KO[K-mer Optimizer<br/>(kmer_optimizer.py)]
    end

    subgraph Sampling
      SF[Sample FASTA<br/>(sample_fasta.py)]
    end

    subgraph Processing
      DP[Data Processing<br/>(data_processing.py)]
    end

    subgraph Exploration
      EDA[EDA & Stats<br/>(eda.py)]
    end

    subgraph Modeling
      TM[Train Model<br/>(train_model.py)]
      AA[Analyze Assembly<br/>(analyze_assembly.py)]
    end

    KO --> SF
    SF --> DP
    DP --> EDA
    DP --> TM
    TM --> AA

k-mer optimization (chooses the optimal k value for the dataset) -> sample the dataset -> process into de Bruijn graph -> train and validate the model
