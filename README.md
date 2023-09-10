McDetect constructs a Convolutional Recurrent Neural Network(CRNN) structure to detect DNA methylation state from Nanopore reads. It is built with Python 3.6 and Tensorflow.

Installation

Prerequisites: Guppy Python >= 3.6 tombo tensorflow

Usage

1.multi to single fast5

multi_to_single_fast5 -i /path/to/raw_fast5/ -s /path/to/single_fast5/ -t 10 --recursive

2.basecalling by guppy

guppy_basecaller -i -s /path/to/single_fast5/ /path/to/outdir --flowcell FLO-MIN106 --kit SQK-LSK109 --cpu_threads_per_caller 2 --num_callers 2 --fast5_out

3.re-squiggle algorithm aligns raw signal for nanopore reads

tombo resquiggle /path/to/outdir/workspace /path/to/genome.fa --processes 12 --ignore-read-locks --corrected-group RawGenomeCorrected_001 --basecall-group Basecall_1D_000 --overwrite

4.extract signals of different kmers

python tomboExtractCG_1D.py -i /path/to/outdir/workspace -l 5 (or 7 9 11 13 15 17) -o 5kmer.signal.tsv

5.training mode by 5mer signal (This step can be skipped if you want to use the model trained with 5mer signal)

python mcDetect_training_CRNN_5mer_ghmccVS_mc_2C_bagging.py -i test.tsv -o test.tsv.result

6.detece 5mC

python mcDetect_test_CRNN_5mer_ghmccVS_mc_2C_bagging.py -i 5kmer.signal.tsv -o 5kmer.result.tsv

7.calculate methylation

perl calculate_methylation.pl 5kmer.result.tsv 5kmer.methylation.result.tsv
