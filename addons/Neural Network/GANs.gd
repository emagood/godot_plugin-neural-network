

class_name GAN
extends Node

var generator : NNET
var discriminator : NNET
var generator_structure : Array
var discriminator_structure : Array

func _init(generator_structure : Array, discriminator_structure : Array, use_bias : bool):
	self.generator_structure = generator_structure
	self.discriminator_structure = discriminator_structure
	generator = NNET.new(generator_structure, use_bias)
	discriminator = NNET.new(discriminator_structure, use_bias)
	set_algorithms()

func set_algorithms():
	generator.use_Adam(0.001)  # Configura el algoritmo Adam para el generador
	discriminator.use_Adam(0.001)  # discriminador

func train_gan(real_data : Array, epochs : int, batch_size : int):
	for epoch in range(epochs):
		print("Epoch: ", epoch)
		for batch in range(0, real_data.size(), batch_size):
			print("Batch: ", batch)
			if batch + batch_size > real_data.size():
				print("Skipping batch ", batch, " due to size constraints")
				continue

			var real_batch = real_data.slice(batch, batch + batch_size)
			if real_batch.size() == 0:
				print("Empty batch, skipping")
				continue
			
			var generated_batch = []
			for i in range(real_batch.size()):
				var noise = generate_noise(100)
				var gen_data = generator_produce_data(noise)
				generated_batch.append(gen_data)
				if i == 0 and batch == 0:
					print("Generated data sample: ", gen_data)

			var real_labels = []
			var fake_labels = []
			for i in range(real_batch.size()):
				var real_label = []
				var fake_label = []
				for j in range(discriminator_structure[-1]):
					real_label.append(1.0)
					fake_label.append(0.0)
				real_labels.append(real_label)
				fake_labels.append(fake_label)

			print("Training discriminator with real data")
			discriminator.train(real_batch, real_labels)  # Etiquetas de datos reales
			var disc_real_loss = discriminator.get_loss(real_batch, real_labels)
			print("Discriminator real data loss: ", disc_real_loss)
			print("Training discriminator with generated data")
			discriminator.train(generated_batch, fake_labels)  # Etiquetas de datos generados
			var disc_fake_loss = discriminator.get_loss(generated_batch, fake_labels)
			print("Discriminator fake data loss: ", disc_fake_loss)

			var noise_batch = []
			for i in range(batch_size):
				noise_batch.append(generate_noise(100))
			var gen_labels = []
			for i in range(batch_size):
				var gen_label = []
				for j in range(generator_structure[-1]):
					gen_label.append(1.0)
				gen_labels.append(gen_label)

			print("Training generator")
			generator.train(noise_batch, gen_labels)
			var gen_loss = generator.get_loss(noise_batch, gen_labels)
			print("Generator loss: ", gen_loss)
			print("Epoch: ", epoch, " Batch: ", batch, " Generator training complete")

		# Visualizar algunas muestras generadas después de cada epoch
		for i in range(5):
			var noise = generate_noise(100)
			var generated_data = generator_produce_data(noise)
			print("Generated data sample after epoch ", epoch, ": ", generated_data)

func generator_produce_data(input_data : Array) -> Array:
	generator.set_input(input_data)
	generator.propagate_forward()
	return generator.get_output()

func generate_noise(size : int) -> Array:
	var noise = []
	for i in range(size):
		noise.append(randi() % 100 / 100.0)
	return noise

func discriminator_classify(input_data : Array) -> float:
	discriminator.set_input(input_data)
	discriminator.propagate_forward()
	return discriminator.get_output()[0]  # Asumiendo salida única para la clasificación

func set_batch_size(new_batch_size : int):
	generator.set_batch_size(new_batch_size)
	discriminator.set_batch_size(new_batch_size)
	
	
func save_generated_images(gan, num_samples, size):
	for i in range(num_samples):
		var noise = gan.generate_noise(100)
		var generated_data = gan.generator_produce_data(noise)
		var image = Image.new()
		var data = PackedByteArray(generated_data)
		image.create_from_data(size, size, false, Image.FORMAT_L8, data)
		image.save_png("res://generated_image_" + str(i) + ".png")
