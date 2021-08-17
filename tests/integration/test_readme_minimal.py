"Test that the example in the readme work correctly"

from tensorflow_similarity.samplers import TFDatasetMultiShotMemorySampler
from tensorflow.keras import layers
from tensorflow_similarity.models import SimilarityModel
from tensorflow_similarity.losses import TripletLoss
from tensorflow_similarity.visualization import viz_neigbors_imgs


def test_readme_minimal():
    sampler = TFDatasetMultiShotMemorySampler(dataset_name='mnist',
                                              classes_per_batch=10)

    inputs = layers.Input(shape=(sampler.x[0].shape))
    x = layers.experimental.preprocessing.Rescaling(1 / 255)(inputs)
    x = layers.Conv2D(32, 7, activation='relu')(x)
    x = layers.MaxPool2D()(x)
    x = layers.Conv2D(64, 3, activation='relu')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64)(x)
    model = SimilarityModel(inputs, x)

    # using Tripletloss to project in metric space
    tloss = TripletLoss()
    model.compile('adam', loss=tloss)
    model.fit(sampler, epochs=5)

    # index emneddings for fast retrivial via ANN
    model.index(x=sampler.x[:100], y=sampler.y[:100], data=sampler.x[:100])

    # Lookup examples nearest indexed images
    nns = model.single_lookup(sampler.x[3713])

    # visualize results result
    # viz_neigbors_imgs(sampler.x[4242], sampler.y[4242], nns)
