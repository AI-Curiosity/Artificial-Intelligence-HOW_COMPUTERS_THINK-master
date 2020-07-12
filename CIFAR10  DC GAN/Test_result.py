from keras.models import load_model
from numpy.random import randn
from matplotlib import pyplot

# Loading saved model file 
# Try a new model or various saved ones to see difference in generation result.

saved_model='generator_model_080.h5'
def generate_latent_points(latent_dim, n_samples):
	x_input = randn(latent_dim * n_samples)
	x_input = x_input.reshape(n_samples, latent_dim)
	return x_input
 
def create_plot(examples, n):
	for i in range(n * n):
		pyplot.subplot(n, n, 1 + i)
		pyplot.axis('off')
		pyplot.imshow(examples[i, :, :])
	pyplot.show()
 
model = load_model(saved_model)
latent_points = generate_latent_points(100, 100)
X = model.predict(latent_points)
X = (X + 1) / 2.0
create_plot(X, 10)