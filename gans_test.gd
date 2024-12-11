class_name GAN
extends Node

var generator : NNET
var discriminator : NNET
var generator_structure : Array
var discriminator_structure : Array
var history = ""
var current = 0
var min_val = 0.00053494601191 
var max_val = 0.53720839288
var adjusted_value

func _init(generator_structure : Array, discriminator_structure : Array, use_bias : bool):
	self.generator_structure = generator_structure
	self.discriminator_structure = discriminator_structure
	generator = NNET.new(generator_structure, use_bias)
	discriminator = NNET.new(discriminator_structure, use_bias)
	set_algorithms()

func set_algorithms():
	generator.use_Adam(0.001)  # Configura el algoritmo Adam para el generador
	discriminator.use_Adam(0.001)  # Configura el algoritmo Adam para el discriminador

func train_gan(real_data : Array, epochs : int, batch_size : int):
	randomize()
	for epoch in range(epochs):
		prints("Epoch: ", epoch , " size data ", real_data.size())
		#print("Epoch: ", epoch)
		history += "Epoch: " + str(epoch) + "\n"

		for batch in range(0, real_data.size(), batch_size):
			history += "batch: " + str(batch) + "\n"
			print("Batch: ", batch)

			var real_batch: Array

			if batch + batch_size > real_data.size():
				history += "Using last incomplete batch: " + str(batch) + "\n"
				print("Using last incomplete batch: ", batch)
				real_batch = real_data.slice(batch, real_data.size())  # Último lote incompleto
			else:
				real_batch = real_data.slice(batch, batch + batch_size)  # Lotes completos

			if real_batch.size() == 0:
				print("Empty batch, skipping")
				history += "Empty batch, skipping" + "\n"
				continue

			var generated_batch = []
			print(real_batch.size(), "antes de generar datos y entrenar")
			history += "antes de entrenar el discriminador real_batch.size()" + str(real_batch.size()) + "\n"
			for i in range(real_batch.size()):
				var noise = generate_noise(100)
				var gen_data = generator_produce_data(noise)
				generated_batch.append(gen_data)
				if i == 0 and batch == 0:
					history += "generate data sample " + str(gen_data) + "\n"
					print("Generated data sample: ")

			# Definir etiquetas únicas en lugar de matrices
			var real_labels = []
			var fake_labels = []
			for i in range(real_batch.size()):
				real_labels.append([0.9])  # Etiqueta para datos reales
				fake_labels.append([0.1])  # Etiqueta para datos generados

			print("Training discriminator with real data")
			var index_bach = randi() % real_batch.size()
			prints("index de batch :", index_bach)
			discriminator.set_input(real_batch[index_bach])# elije una imagen al azar
			discriminator.set_target([0.9])# no solo 1.0
			discriminator.propagate_forward()
			print("Discriminator propagate_forward with real data done")
			discriminator.propagate_backward()
			print("Discriminator propagate_backward with real data done")
			discriminator.apply_gradients(0.005)
			print("Discriminator apply_gradients with real data done")
			save_training_images(real_batch, 28 , "disriminados")
			real_batch = format_data(real_batch)
			real_labels = format_data(real_labels)
			#print("Type of real_batch: ", typeof(real_batch))
			#print("Type of real_labels: ", typeof(real_labels))
			#print("Type of generated_batch: ", typeof(generated_batch))
			#print("Type of fake_labels: ", typeof(fake_labels))
			#prints(real_batch.size() ," ",real_labels.size())

# Entrenar discriminador con datos reales y generados usando train
			#discriminator.train(real_batch, real_labels)
			#discriminator.train(generated_batch, fake_labels)

			discriminator.train(real_batch, real_labels)
			var errordis = discriminator.get_loss(real_batch, real_labels)
			prints("error del discriminadir aprendiendo de imagen real" , errordis,"   labels etiqueta ", real_labels)

			var disc_real_loss = compute_loss(discriminator.get_output(), real_labels)  # Real labels
			print("Discriminator real data loss: ", disc_real_loss)
			history += "Discriminator real data loss: " + str(disc_real_loss) + "\n"

			print("Training discriminator with generated data")
			discriminator.set_input(generated_batch[0])
			discriminator.set_target([1.0])
			discriminator.propagate_forward()
			print("Discriminator propagate_forward with generated data done")
			discriminator.propagate_backward()
			print("Discriminator propagate_backward with generated data done")
			discriminator.apply_gradients(0.01)
			print("Discriminator apply_gradients with generated data done")
			discriminator.train(generated_batch, fake_labels)
			var disc_fake_loss = compute_loss(discriminator.get_output(), fake_labels)  # Fake labels
			print("Discriminator fake data loss: ", disc_fake_loss)
			discriminator.train(generated_batch, fake_labels)
			prints("   gtamaño del entrenamientro ",generated_batch.size(),fake_labels.size())
			history += "Discriminator fake data loss: " + str(disc_fake_loss) + "\n"

			var noise_batch = []
			for i in range(batch_size):
				noise_batch.append(generate_noise(100))

			var misleading_labels = []
			for i in range(batch_size):
				misleading_labels.append([1.0])  # El generador intenta engañar al discriminador

			var disc_predictions = []
			for i in range(noise_batch.size()):
				var gen_data = generator_produce_data(noise_batch[i])
				var prediction = discriminator_classify(gen_data)
				#var prediction2 = discriminator_classify(real_batch[index_bach])
				disc_predictions.append(prediction)
				prints("prediion de lo generado y lo otro : " , prediction)
				adjusted_value = 1.0 - ((prediction[0] - min_val) / (max_val - min_val)) * (0.999 - 0.001) + 0.001
				prints(adjusted_value , " error de enrtrada salida ")


			var generator_labels = []
			for i in range(disc_predictions.size()):
				#prints("prediion discpredicction  : " , disc_predictions.size())
				generator_labels.append([1.0])

			var gen_loss = compute_loss(disc_predictions, generator_labels)

			generator.set_input(noise_batch[0])
			
			save_training_images(noise_batch, 28,"generador")
			prints("Noise size : " , noise_batch.size()," size de  bach 0: ",noise_batch[0].size())
			
		 #Crear una lista de etiquetas plana con el tamaño adecuado
			generator_labels = []
			for i in range(784):
				generator_labels.append(1.0)

			prints("size label generado ",generator_labels.size())
			generator.set_target(generator_labels)# quiere que sea verdad [1.0]
			generator.propagate_forward()
			generator.propagate_backward()

			generator.apply_gradients(0.01)  # gen_loss * 0.4Uso el error del discriminador para ajustar el generador
			print("Error del generador: ", gen_loss)
			print("size label: ", generator_labels.size(), " is[0]: ", batch, " Generator training complete")
			
			#var format = format_data(generator_labels)
			#prints(format)
			#prints(format_data(noise_batch))
			#generator_labels = []
			#for i in range(784):
				#generator_labels.append([1.0])
			var target_data = []
			#var target_example = []
			#for j in range(784):
				#target_example.append(1.0)
			target_data.append(generator_labels)
			prints(noise_batch.size())
			var pari = 0.0
			for i in range(noise_batch.size()):
				
				var data_train = []
				data_train.append(noise_batch[i])
				
				generator.train(data_train, target_data)
				var gen_data = generator.get_output()
				#var data_imag = []
				#data_imag.append(gen_data)
				#save_training_images(data_imag, 28,"generador")
				var prediction = discriminator_classify(gen_data)
				prints("predicion de ia discrimina ", prediction[0])
				generator.propagate_backward()
				adjusted_value = 1.0 - ((prediction[0] - min_val) / (max_val - min_val)) * (0.999 - 0.001) + 0.001
				generator.apply_gradients(adjusted_value) 
				if pari > adjusted_value:
					prints("se aprendio mas que antes en: " ,adjusted_value - pari )
				else:
					prints("se perdio un promedio de :", pari - adjusted_value)
				pari = adjusted_value
				prints((adjusted_value + randf() * (0.03 - 0.01) + 0.001)  / 3)
			
#
		#for i in range(5):
			#var noise = generate_noise(100)
			#var generated_data = generator_produce_data(noise)
			#
			#print("Generated data sample after epoch ", epoch, ": ", "generated_data")
	save_history(history, "res://data/hgans.txt")

func compute_loss(predictions: Array, labels: Array) -> float:
	var loss = 0.0
	for i in range(predictions.size()):
		if predictions[i] is Array:
			loss += pow(predictions[i][0] - labels[i][0], 2)  # Calcula la pérdida cuadrática
		else:
			loss += pow(predictions[i] - labels[i][0], 2)  # Ajuste para valores float
	return loss / predictions.size()  # Devuelve la pérdida media

func generator_produce_data(input_data: Array) -> Array:
	generator.set_input(input_data)
	generator.propagate_forward()
	return generator.get_output()

func generate_noise(size: int) -> Array:
	var noise = []

	for i in range(size):
		noise.append(randf_range(0.9, 0.0001) + randf_range(0.0001, 0.0) )  # Genera un valor aleatorio entre 0 y 1
	return noise

func discriminator_classify(input_data: Array) -> Array:
	discriminator.set_input(input_data)
	discriminator.propagate_forward()
	return [discriminator.get_output()[0]]  # Devuelve un array con un solo valor

func set_batch_size(new_batch_size : int):
	generator.set_batch_size(new_batch_size)
	discriminator.set_batch_size(new_batch_size)

func save_gan(path: String):
	generator.save_binary(path + "_generator.bin")
	discriminator.save_binary(path + "_discriminator.bin")
	print("GAN model saved at: ", path)

func load_gan(path: String):
	generator.load_data(path + "_generator.bin")
	discriminator.load_data(path + "_discriminator.bin")
	print("GAN model loaded from: ", path)

func save_history(data: String, path: String):
	var temp_file = FileAccess.open(path, FileAccess.WRITE)
	if temp_file:
		temp_file.store_string(data)
		temp_file.close()

func format_data(data: Array) -> Array:
	var formatted_data = []
	for item in data:
		if typeof(item) != TYPE_ARRAY:
			formatted_data.append([item])
		else:
			formatted_data.append(item)
	return formatted_data


func save_training_images(real_data, size, etiqueta):
	var max = 2
	for i in range(real_data.size()):
		max -= 1
		if max <= 0:
			return
		var image_data = real_data[i]

		if image_data.size() != size * size:
			history += "Training image data has incorrect size. Expected: "+str( size * size ) + " Got: " + str (  image_data.size()) + "\n"
			continue

		var image = Image.create(size, size, false, Image.FORMAT_RGBA8)
		
		if image.get_width() != size or image.get_height() != size:
			history += "Failed to create image with correct dimensions. Width: " + str(image.get_width()) + " Height: "+  str(image.get_height())+ "\n"
			continue
		
		for y in range(size):
			for x in range(size):
				var index = y * size + x
				var value = image_data[index]
				var color = Color(value, value, value)
				image.set_pixel(x, y, color)
		current +=1
		var save_path = "res://img/real_data_trainin_" + etiqueta + str(i + current) + ".png"
		var result = image.save_png(save_path)
		if result != OK:
			print("Failed to save training image: ", save_path)
			history += "Failed to save training image: " + save_path
		else:
			print("Training image saved: ", save_path)
