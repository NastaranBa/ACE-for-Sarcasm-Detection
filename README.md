# Affective and Contextual Embedding for Sarcasm Detection
![ACE1F](https://user-images.githubusercontent.com/32373744/96896414-77d59e80-145b-11eb-8d45-c3de7d139fad.png)

**Overview of the Proposed Model ACE 1**

![ACE2F](https://user-images.githubusercontent.com/32373744/96896519-989df400-145b-11eb-8c84-440032d8b5f5.png)

**Overview of the Proposed Model ACE 2**


**Contributions**

1) We present two novel deep neural network language models (ACE 1 and ACE 2) for sarcasm detection.Each model extends the architecture of BERT by incorporating both affective and contextual featuresof text to build a classifier that can determine whether a document is sarcastic or not. To the bestof our knowledge, this is the first attempt to directly alter BERT’s architecture and train it from theground-up (rather than using the already pre-trained BERT embeddings) for sarcasm detection.
2) Integral to our proposed models is a novel model that learns the affective representation of a document,using a Bi-LSTM architecture with multi-head attention.  The resulting representation takes intoaccount the importance of the affect representations of the sentences in the document.
3) We design and evaluate alternatives that materialize each of the two components (affective featureembedding and contextual feature embedding) of the proposed deep neural network architecturemodel. We systematically evaluate the effectiveness of each alternative architecture.
4) We conduct an extensive evaluation of the performance of the proposed models (ACE 1 and ACE 2),which demonstrates that they significantly outperform current state-of-the-art models for sarcasmdetection.

**Corpora for Training Embeddings**

1) News : https://www.kaggle.com/snapcrack/all-the-news
2) Wikipedia : https://www.kaggle.com/jkkphys/english-wikipedia-articles-20170820-sqlite
3) WikiSarc: IMSDB:https://www.imsdb.com/ Scraper:https://github.com/JoeKarlsson/movie-script-scraper
4) Riloff: riloff-etal-2013-sarcasm


** Sarcasm Detection Datasets**

1) Onion : https://github.com/rishabhmisra/News-Headlines-Dataset-For-Sarcasm-Detection
2) IAC : https://nlds.soe.ucsc.edu/sarcasm2
3) Reddit : https://nlp.cs.princeton.edu/SARC/0.0/
4) Pt'acek: Ptcek2014SarcasmDO
5) SemEval-201: van-hee-etal-2018-semeval

