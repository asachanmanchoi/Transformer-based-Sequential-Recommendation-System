# Dataset & code running finding Review Meeting

### Challenge 1: Run and comprehensively master the training and prediction of original model

We need to (1) thoroughly run and (2) familiarize ourselves with the training and prediction principles of the two models (source code) below, in order to design our own model that can enhance the accuracy of predictions.
- [RecFormer](https://github.com/AaronHeee/RecFormer)
- [Longformer](https://github.com/allenai/longformer)

### Challenge 2: Dataset size

The existing dataset is so large that it takes a month to complete one training session. We need to consider how to reduce the size of the dataset (while retaining the necessary and helpful parts) in order to complete our experiment within the specified time.

### Notice

1. You should pay attention to the batch size setting; otherwise, the training may not converge (at least 4 is recommended).
2. When dealing with cold start problems, while working on datasets and designing models, you can consider the possibility of transferring the knowledge from one domain to another. Examples are as follows:
    1. There might be some content in the movie that can assist us in comprehending the product. For instance, if the movie revolves around food, we can include relevant content to acquire knowledge associated with (learn more about) food and predict products related to it.
    2. We can identify products that have a high degree of similarity in order to make accurate predictions.
        - Books <-> Movies and TV <-> CDs and Vinyl
        - Sports and Outdoors <-> Clothing, Shoes and Jewelry <-> Amazon Fashion
3. To minimize complexity, it is advisable not to alter the structure of the original project but rather enhance its existing capabilities, such as addressing their insufficient analysis regarding cold start and time series.
4. Links that could be helpful are as follows:
    - [Amazon Review dataset](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/)
    - [IMDb Movie Reviews](https://paperswithcode.com/dataset/imdb-movie-reviews)
    - [Hugging Face](https://github.com/huggingface)