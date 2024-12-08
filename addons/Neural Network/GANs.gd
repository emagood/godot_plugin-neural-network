

class_name GAN
extends Node

var generator : NNET
var discriminator : NNET
var generator_structure : Array
var discriminator_structure : Array
var history = ""



func _init(generator_structure : Array, discriminator_structure : Array, use_bias : bool):
	self.generator_structure = generator_structure
	self.discriminator_structure = discriminator_structure
	generator = NNET.new(generator_structure, use_bias)
	discriminator = NNET.new(discriminator_structure, use_bias)
	set_algorithms()

func set_algorithms():
	generator.use_Adam(0.01)  # Configura el algoritmo Adam para el generador
	discriminator.use_Adam(0.01)  # discriminador
	
func train_gan(real_data : Array, epochs : int, batch_size : int):
	for epoch in range(epochs):
		print("Epoch: ", epoch)
		history += "Epoch: " + str(epoch) + "\n"

		for batch in range(0, real_data.size(), batch_size):
			history += "batch: " + str(batch) + "\n"
			print("Batch: ", batch)

			if batch + batch_size > real_data.size():
				history += "Skipping batch " + str(batch) + " due to size constraints" + "\n"
				print("Skipping batch ", batch, " due to size constraints")
				continue

			var real_batch = real_data.slice(batch, batch + batch_size)
			if real_batch.size() == 0:
				print("Empty batch, skipping")
				history += "Empty batch, skipping" + "\n"
				continue

			var generated_batch = []
			for i in range(real_batch.size()):
				var noise = generate_noise(100)
				var gen_data = generator_produce_data(noise)
				generated_batch.append(gen_data)
				if i == 0 and batch == 0:
					history += "generate data sample " + str(gen_data) + "\n"
					print("Generated data sample: ", gen_data)

			var real_labels = []
			var fake_labels = []
			for i in range(real_batch.size()):
				var real_label = []
				var fake_label = []
				for j in range(discriminator_structure.size() - 1):
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
			history += "Discriminator fake data loss:" + str(disc_fake_loss) + "\n"

			var noise_batch = []
			for i in range(batch_size):
				noise_batch.append(generate_noise(100))

			# El generador intenta engañar al discriminador
			var misleading_labels = []
			for i in range(batch_size):
				var gen_label = []
				for j in range(generator_structure.size() - 1):
					gen_label.append(1.0)
				misleading_labels.append(gen_label)

			# Calcular la pérdida del discriminador para las imágenes generadas
			var disc_predictions = []
			for i in range(noise_batch.size()):
				var gen_data = generator_produce_data(noise_batch[i])
				var prediction = discriminator_classify(gen_data)
				disc_predictions.append(prediction)

			# Calcular las etiquetas para la pérdida del generador
			var generator_labels = []
			for i in range(disc_predictions.size()):
				generator_labels.append(1.0)

			# Calcular la pérdida del generador
			var gen_loss = compute_loss(disc_predictions, generator_labels)

			# Retropropagar la pérdida del generador
			generator.set_input(noise_batch)
			generator.set_target(generator_labels)
			generator.propagate_forward()
			generator.propagate_backward()

			# Ajustar los pesos del generador utilizando la pérdida calculada
			generator.apply_gradients(gen_loss)

			print("Epoch: ", epoch, " Batch: ", batch, " Generator training complete")

		# Visualizar algunas muestras generadas después de cada epoch
		for i in range(5):
			var noise = generate_noise(100)
			var generated_data = generator_produce_data(noise)
			print("Generated data sample after epoch ", epoch, ": ", generated_data)
	save_history(history, "res://data/hgans.txt")


func compute_loss(predictions: Array, labels: Array) -> float:
	var loss = 0.0
	for i in range(predictions.size()):
		loss += pow(predictions[i] - labels[i], 2)
	return loss / predictions.size()

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
	
	

func save_gan(path: String):
	#generator.save_binary(path + "_generator.bin")#  2.7 megas 
	#discriminator.save_binary(path + "_discriminator.bin")#  2.8 megas 
	print("GAN model saved at: ", path)
	#generator.compress_nnet(path + "_generator.bin")
	#discriminator.compress_nnet(path + "_discriminator.bin")

func load_gan(path: String):
	generator.load_data(path + "_generator.bin")
	discriminator.load_data(path + "_discriminator.bin")
	print("GAN model loaded from: ", path)


# Función para guardar historial en texto
func save_history(data, path: String):
	var temp_file = FileAccess.open(path, FileAccess.WRITE)
	if temp_file:
		temp_file.store_string(data)
		temp_file.close()
