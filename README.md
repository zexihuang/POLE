# POLE: Polarized Embedding for Signed Networks

This repository is a reference implementation of the polarized embedding method as described in the paper:
<br/>
> POLE: Polarized Embedding for Signed Networks.<br>
> Zexi Huang, Arlei Silva, Ambuj Singh.<br>
> To appear, ACM International Conference on Web Search and Data Mining, 2022.
> <Insert paper link>

The proposed method, POLE, captures both topological and signed similarities jointly via signed autocovariance and leverages matrix factorization to generate embedding. 
It achieves state-of-the-art signed link prediction performance especially for predicting conflicts in polarized signed graphs.

## Embedding

### Example Usage
To embed the WoW-EP8 network with default settings:

    python src/embedding.py --graph graph/WoW-EP8.edges --embedding emb/WoW-EP8.embedding

where `graph/WoW-EP8.edges` stores the input graph and `emb/WoW-EP8.embedding` is the target file for output embeddings. 
### Options
You can check out all the available options with:

	python src/embedding.py --help

### Input Graph
The supported input graph format is a list of edges:

	node1_id_int node2_id_int <signed_weight_float>
		
where node ids are should be consecutive integers starting from 0 and weights include link signs. 

### Output Embedding
The output embedding file has *n* lines where *n* is the number of nodes in the graph. Each line stores the learned embedding of the node with its id equal to the line number: 

	emb_dim1 emb_dim2 ... emb_dimd

## Evaluation

The proposed embedding method enables effective signed link prediction. Here, we show how to evaluate it on this task. Full evaluation options are can be found with:
                                              
    python src/slp.py --help

Note that the results shown below may not be identical to those in the paper due to different random seeds, but the conclusions are the same.  

### Link Removal Preparation

We first need to remove a proportion of links in the original graph for testing:
     
     python src/slp.py --mode preparation --graph graph/WoW-EP8.edges --removed-edges graph/WoW-EP8.removed-edges --remaining-edges graph/WoW-EP8.remaining-edges

This takes the original graph `graph/WoW-EP8.edges` as input and output the removed and remaining edges (residual graph) to `graph/WoW-EP8.removed-edges` and `graph/WoW-EP8.remaining-edges`.

### Signed Link Prediction

Vanilla signed link prediction takes the input of POLE embedding based on the residual graph. We first generate the residual embedding. For example:

    python src/embedding.py --graph graph/WoW-EP8.remaining-edges --embedding emb/WoW-EP8.residual-embedding --markov-time 0.5

Then, we can evaluate the performance of signed link prediciton in terms of *positive precision@k* and *negative precision@k*:
    
    python src/slp.py --mode slp --embedding emb/WoW-EP8.residual-embedding --removed-edges graph/WoW-EP8.removed-edges --remaining-edges graph/WoW-EP8.remaining-edges --k 1.0

The results for WoW-EP8 dataset with varying *k* are as follows:

|       k      |   10%  |   20%  |   30%  |   40%  |   50%  |   60%  |   70%  |   80%  |   90%  |  100%  |
|:------------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|
| Positive p@k | 0.9836 | 0.9716 | 0.9434 | 0.9056 | 0.8582 | 0.8108 | 0.7623 | 0.7113 | 0.6670 | 0.6272 |
| Negative p@k | 0.2276 | 0.2290 | 0.2064 | 0.1910 | 0.1809 | 0.1731 | 0.1671 | 0.1583 | 0.1512 | 0.1446 |


### Signed Link Prediction with Link Existence Information

Combining the proposed embedding method with an unsigned embedding method (e.g., [RWE](https://github.com/zexihuang/random-walk-embedding)) can further improve the signed link prediction performance, especially for the negative links. 
To do that, first generate signed embedding (POLE) and unsigned embedding (RWE) with:
    
    python src/embedding.py --graph graph/WoW-EP8.remaining-edges --embedding emb/WoW-EP8.residual-embedding --markov-time 0.5
    python src/embedding.py --graph graph/WoW-EP8.remaining-edges --embedding emb/WoW-EP8.residual-unsigned-embedding --markov-time 0.6 --signed False

Then we can evaluate the signed link prediction performance with both embeddings input:

    python src/slp.py --mode slp-rwe --embedding emb/WoW-EP8.residual-embedding --unsigned-embedding emb/WoW-EP8.residual-unsigned-embedding --removed-edges graph/WoW-EP8.removed-edges --remaining-edges graph/WoW-EP8.remaining-edges --k 1.0

The results for WoW-EP8 dataset with varying *k* are as follows:

|       k      |   10%  |   20%  |   30%  |   40%  |   50%  |   60%  |   70%  |   80%  |   90%  |  100%  |
|:------------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|
| Positive p@k | 0.9905 | 0.9677 | 0.9385 | 0.8975 | 0.8522 | 0.8030 | 0.7521 | 0.7032 | 0.6600 | 0.6203 |
| Negative p@k | 0.4368 | 0.3694 | 0.3438 | 0.3222 | 0.3071 | 0.2978 | 0.2818 | 0.2682 | 0.2566 | 0.2441 |


## Citing
If you find our framework useful, please consider citing the following paper:

	@inproceedings{pole,
	author = {Huang, Zexi and Silva, Arlei and Singh, Ambuj},
	 title = {POLE: Polarized Embedding for Signed Networks},
	 booktitle = {WSDM},
	 year = {2022}
	}