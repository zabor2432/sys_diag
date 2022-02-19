## Movie recommending system

### Data

As an input data publicly available MovieLens Tag Genome Dataset 2021 dataset was used, this dataset have following feature structure saved in csv format: 

    userId,movieId,rating,timestamp

Where rating is a value ranging from 0.5 to 5. For our system we wanted to recommend movies that user will probably like, so every rating above 3 was changed to 1 and every other rating was changed to zero to represent binary infromation about user preferences.

Dataset contains `100837` ratings with following distribution: 
```
0.5: 1370
1.0: 2811
1.5: 1791
2.0: 7551
2.5: 5550
3.0: 20047
3.5: 13136
4.0: 26818
4.5: 8551
5.0: 13211
```

### Model

Neural Collaborative Filtering model was implemented, this particular model combines deep learning approach with matrix factorization, it's architecture is shown on following figure.

![NCF](./ncfp.png)

Item and User are one-hot encoded to a vector of size `user_count + 1` and `item_count + 1` respectively, additional one is required to deal with null user and null item problem.

Model output is a probability that particular user will like the recommended movie, for loss function `binary-crossentropy` was used.

Different hyper-parameters sets have been searched to train this model, including learing rate, number of neurons in hidden layers or dropout rate.

### Results

Main performance metric for this system was `accuracy`, however in many publications authors are using `recall` metric, but for model that is not operating on user sessions this could be misleading.

Detailed resuts can be found in the table below

|lr   |dropout|hidden|epoch_accuracy|epoch_accuracy|epoch_auc|epoch_auc|
|-----|-------|------|--------------|--------------|---------|---------|
|0.1  |0.2    |100.0 |0.5775        |0.5363        |0.5344   |0.5038   |
|0.01 |0.1    |50.0  |0.8854        |0.6852        |0.9527   |0.7293   |
|0.001|0.1    |75.0  |0.8810        |0.6924        |0.9486   |0.7419   |
|0.001|0.2    |50.0  |0.8719        |0.7059        |0.9414   |0.7539   |
|0.01 |0.1    |100.0 |0.8842        |0.6858        |0.9528   |0.7289   |
|0.1  |0.2    |50.0  |0.5753        |0.5477        |0.5298   |0.5017   |
|0.001|0.1    |50.0  |0.8740        |0.7024        |0.9424   |0.7532   |
|0.1  |0.2    |75.0  |0.5763        |0.5488        |0.5325   |0.5029   |
|0.001|0.2    |100.0 |0.8696        |0.7060        |0.9385   |0.7556   |
|0.1  |0.1    |50.0  |0.5809        |0.5558        |0.5377   |0.5037   |
|0.1  |0.1    |100.0 |0.5782        |0.5561        |0.5333   |0.4987   |
|0.001|0.2    |75.0  |0.8760        |0.6982        |0.9447   |0.7461   |
|0.001|0.1    |100.0 |0.8836        |0.6964        |0.9508   |0.7416   |
|0.01 |0.2    |100.0 |0.8742        |0.6957        |0.9448   |0.7391   |
|0.01 |0.2    |75.0  |0.8786        |0.6857        |0.9473   |0.7278   |
|0.01 |0.1    |75.0  |0.8801        |0.6700        |0.9494   |0.7064   |
|0.01 |0.2    |50.0  |0.8808        |0.6839        |0.9493   |0.7322   |
|0.1  |0.1    |75.0  |0.5744        |0.5447        |0.5290   |0.49975156784057617|



