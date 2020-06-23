# TensorFlow Similarity Serving

Tf.Similarity is a web application that allows you to easily interact with tf.similarity models. Tf.similarity aims to give you a better understanding of how tf.similarity
works by visualizing key concepts of the tf.similarity package such as nearest targets. It also aims to give you an understanding of how tf.similarity models make decisions
through a GradCam++ visualization.

## Demonstration

[![Watch the video](https://img.youtube.com/vi/WSryZYbzDOI/maxresdefault.jpg)](https://www.youtube.com/watch?v=WSryZYbzDOI&feature=youtu.be)

## Getting Started

0. Create a python venv: <pre><code>python3 -m venv venv</code></pre>
1. Activate the venv: <pre><code>. venv/bin/activate</code></pre>
2. Install Flask <pre><code>pip install Flask</code></pre>
3. Install all other dependencies <pre><code>pip install tensorflow==2.1.1 & !pip install keras-tuner confusable-homoglyphs matplotlib==3.1.0 tensorflow-plot uuid & pip install --upgrade grpcio & pip install umap-learn plotly altair MulticoreTSNE & pip install -U altair vega_datasets vega & wget https://anaconda.org/CannyLab/tsnecuda/2.1.0/download/linux-64/tsnecuda-2.1.0-cuda100.tar.bz2 !tar xvjf tsnecuda-2.1.0-cuda100.tar.bz2 & cp -r site-packages/* /usr/local/lib/python3.6/dist-packages/ & ln -s /content/lib/libfaiss.so $LD_LIBRARY_PATH/libfaiss.so
</code></pre>
5. Export the server application <pre><code>export FLASK_APP=main.py</code></pre>
6. Run the application <pre><code>flask run</code></pre>


## Tutorials

Head over to `tensorflow-similarity/experiments` to find tutorials on using TensorFlow Similarity on Cifar100, mnist, fashion_mnist, and iris datasets.

## Terminology

Similarity Model - A Similarity model is trained by comparing tuples of input examples, instead of trying to predict classes for single input examples. 

GradCam - uses the gradients of any target concept (say logits for “dog” or even a caption), flowing into the final convolutional layer to produce a coarse localization map highlighting the important regions in the image for predicting the concept.

## Disclaimer

This is not an official Google product.