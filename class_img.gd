extends Node
class_name IMG



var discriminator : NNET

var discriminator_structure : Array
var history = ""
var current = 0
var min_val = 0.00053494601191 
var max_val = 0.53720839288
var adjusted_value


func _init(discriminator_structure : Array , use_bias : bool):
	randomize()

	self.discriminator_structure = discriminator_structure
	discriminator = NNET.new(discriminator_structure, use_bias)
	set_algorithms()

func set_algorithms():
	discriminator.use_Adam(0.004)  # Configura el algoritmo Adam para el discriminador



func train_gan(real_data : Array, tag : Array, epochs : int, batch_size : int):
	
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
				var gen_data 
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

			discriminator.train(real_batch, tag)
			var errordis = discriminator.get_loss(real_batch, tag)
			#prints("error del discriminadir aprendiendo de imagen real" , errordis,"   labels etiqueta ", real_labels)

			var disc_real_loss = compute_loss(discriminator.get_output(), tag)  # Real labels
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
			discriminator.traintrain(real_batch, tag)
			var disc_fake_loss = compute_loss(discriminator.get_output(), fake_labels)  # Fake labels
			print("Discriminator fake data loss: ", disc_fake_loss)
			discriminator.train(generated_batch, fake_labels)
			prints("   gtamaño del entrenamientro ",generated_batch.size(),fake_labels.size())
			history += "entrenamiento " + "\n"



#
			#var disc_predictions = []
			#for i in range(9):
				#var gen_data 
				#var prediction = discriminator_classify(gen_data)
#
				#disc_predictions.append(prediction)
				#prints("prediion de lo generado y lo otro : " , prediction)
				#adjusted_value = 1.0 - ((prediction[0] - min_val) / (max_val - min_val)) * (0.999 - 0.001) + 0.001
				#prints(adjusted_value , " error de enrtrada salida ")
#
#
			#var generator_labels = []
			#for i in range(disc_predictions.size()):
				##prints("prediion discpredicction  : " , disc_predictions.size())
				#generator_labels.append([1.0])
#
			#var gen_loss = compute_loss(disc_predictions, generator_labels)

	save_history(history, "res://data/hgans.txt")










func compute_loss(predictions: Array, labels: Array) -> float:
	var loss = 0.0
	for i in range(predictions.size()):
		if predictions[i] is Array:
			loss += pow(predictions[i][0] - labels[i][0], 2)  # Calcula la pérdida cuadrática
		else:
			loss += pow(predictions[i] - labels[i][0], 2)  # Ajuste para valores float
	return loss / predictions.size()  # Devuelve la pérdida media



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

	discriminator.set_batch_size(new_batch_size)

func save_gan(path: String):

	discriminator.save_binary(path + "_img_train.bin")
	print("img model saved at: ", path)

func load_gan(path: String):

	discriminator.load_data(path + "_img_train.bin")
	print("img model loaded from: ", path)

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
	var max = 1
	for i in range(real_data.size()):
		
		if max <= 0:
			return
		var image_data = real_data[randi() % real_data.size()]

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
		max -= 1
