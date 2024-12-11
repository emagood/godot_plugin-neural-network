extends Node
var history = ""
var size = 28
var image_paths = []
var dir_fashion = "C:/Users/Emabe/Desktop/entrena/fashion/train/"
var max = 25
var real_data



func _ready():
	randomize()
	var generator_structure = [100, 256, 512, 980, size*size]  # Estructura de ejemplo para el generador
	var discriminator_structure = [size*size, 256, 256, 64, 1]  # Estructura de ejemplo para el discriminador
	var use_bias = true
	
	var gan = GAN.new(generator_structure, discriminator_structure, use_bias)
	gan.load_gan("res://data/gans")
	
# Ejemplo de uso
	#image_paths = [
	#"res://img/training_image_0.png",
	#"res://img/training_image_1.png",
	#"res://img/training_image_2.png"
#]
	dir_contents(dir_fashion , max)
	var size = 28  # Tamaño de las imágenes
	var real_data = load_training_images(image_paths, size)
	print("Datos cargados: ")

	''' esto es para generar imagenes random '''
	## Crear datos de ejemplo para entrenar
	#real_data = []
	#for i in range(4):  # Generar 1000 imágenes de ejemplo (suponiendo imágenes de 28x28 píxeles)
		#var image_data = []
		#for j in range(size*size):
			#image_data.append(randf())
		#real_data.append(image_data)
		
		
		
	save_training_images(real_data, size)# size imagen
	# Entrenar la GAN
	gan.train_gan(real_data, 1, 25)  # Epochs = 10, batch_size = 32
	gan.save_gan("res://data/gans")

	
	save_generated_images(gan, 5, size) # valor size imagen
	print("Training complete")
	history += "Training complete"
	save_history(history, "res://data/historial.txt")









#func save_training_images(real_data, size):
	#for i in range(real_data.size()):
		#var image_data = real_data[i]
#
		#if image_data.size() != size * size:
			#print("Training image data has incorrect size. Expected: ", size * size, " Got: ", image_data.size())
			#continue
#
		#var image = Image.new()
		#image.create(size, size, false, Image.FORMAT_RGB8)
		#var data = PackedByteArray()
#
		## Normalizar los datos y convertirlos a RGB (24 bits)
		#for value in image_data:
			#var byte_value = clamp(int(value * 255), 0, 255)
			#data.append(byte_value)  # R
			#data.append(byte_value)  # G
			#data.append(byte_value)  # B
		#prints("dato size " , data.size(), " datos " , data )
		## Verificar que el tamaño de los datos es correcto
		#if data.size() != size * size * 3:
			#print("PackedByteArray has incorrect size. Expected: ", size * size * 3, " Got: ", data.size())
			#continue
#
		#image.create_from_data(size, size, false, Image.FORMAT_RGB8, data)
#
		#var save_path = "res://img/training_image_" + str(i) + ".png"
		#var img = Image.new()
		#img.create_from_data(28, 28, false, Image.FORMAT_RGB8, data)#FORMAT_ETC
		#var resave = img.save_jpg(save_path)
		#if resave != OK:
			#print("error de crear y guardar imagen : ", save_path)
		#else:
			#print("Training image saved: ", save_path)
			#
			#
		##var result = image.save_png(save_path)
		##if result != OK:
			##print("Failed to save training image: ", save_path)
		##else:
			##print("Training image saved: ", save_path)




func save_training_images(real_data, size):
	#var img = Image.create(90, 90, false, Image.FORMAT_RGBA8)
	#print("Alto: ", img.get_height(), " Ancho: ", img.get_width())
	var max = 10
	for i in range(real_data.size()):
		max -= 1
		if max <= 0:
			return
		var image_data = real_data[i]

		if image_data.size() != size * size:
			history += "Training image data has incorrect size. Expected: "+str( size * size ) + " Got: " + str (  image_data.size()) + "\n"
			continue

		var image = Image.create(size, size, false, Image.FORMAT_RGBA8)
		#image.create(size, size, false, Image.FORMAT_RGB8)
		
		# Asegurarnos de que la imagen se crea correctamente
		if image.get_width() != size or image.get_height() != size:
			history += "Failed to create image with correct dimensions. Width: " + str(image.get_width()) + " Height: "+  str(image.get_height())+ "\n"
			continue
		
		# Dibujar los píxeles directamente
		for y in range(size):
			for x in range(size):
				var index = y * size + x
				var value = image_data[index]
				var color = Color(value, value, value)  # Usar el mismo valor para R, G y B para escala de grises
				image.set_pixel(x, y, color)
		
		var save_path = "res://img/training_image_" + str(i) + ".png"
		var result = image.save_png(save_path)
		if result != OK:
			print("Failed to save training image: ", save_path)
			history += "Failed to save training image: " + save_path
		else:
			print("Training image saved: ", save_path)






func save_generated_images(gan, num_samples, size):
	for i in range(num_samples):
		var noise = gan.generate_noise(100)
		var generated_data = gan.generator_produce_data(noise)
		var loss = gan.discriminator_classify(generated_data)
		prints("es preciso en " , loss)
		if generated_data.size() != size * size:
			print("Generated data has incorrect size. Expected: ", size * size, " Got: ", generated_data.size())
			continue

		# Crear la imagen con dimensiones correctas y formato RGBA8
		var image = Image.create(size, size, false, Image.FORMAT_RGBA8)

		# Dibujar los píxeles directamente
		for y in range(size):
			for x in range(size):
				var index = y * size + x
				var value = generated_data[index]
				var color = Color(value, value, value, 1)  # Usar el mismo valor para R, G, B y alfa
				image.set_pixel(x, y, color)

		var save_path = "res://img/generated_image_" + str(i) + ".png"
		var result = image.save_png(save_path)
		if result != OK:
			print("Failed to save generated image: ", save_path)
		else:
			print("Generated image saved: ", save_path)
			
			
			
# Función para guardar historial en texto
func save_history(data, path: String):
	var temp_file = FileAccess.open(path, FileAccess.WRITE)
	if temp_file:
		temp_file.store_string(data)
		temp_file.close()
		
		
func compute_loss(predictions: Array, labels: Array) -> float:
	var loss = 0.0
	for i in range(predictions.size()):
		loss += pow(predictions[i] - labels[i], 2)
	return loss / predictions.size()
	



func load_training_images(image_paths: Array, size: int) -> Array:
	var loaded_data = []
	
	for path in image_paths:
		var image = Image.new()
		var error = image.load(path)
		
		if error != OK:
			print("Failed to load image: ", path)
			continue
		
		if image.get_width() != size or image.get_height() != size:
			print("Image size does not match. Width: ", image.get_width(), " Height: ", image.get_height(), " Expected: ", size)
			continue
		
		var image_data = []
		for y in range(size):
			for x in range(size):
				var color = image.get_pixel(x, y)
				var value = color.r  # Asumiendo escala de grises, usar solo el canal rojo
				image_data.append(value)
		
		loaded_data.append(image_data)
	
	return loaded_data




func dir_contents(path , max):
	
	var dir = DirAccess.open(path)
	if dir:
		dir.list_dir_begin()
		var file_name = dir.get_next()
		while file_name != "" and max > 0:
			max -= 1
			if dir.current_is_dir():
				print("⭐️DIRECTORIO ENCONTRADO⭐️ : " + file_name)
			else:
				print("⭐ARCHIVO ENCONTRADO⭐️ : " + file_name)
				print("f⭐️EXTENCION DEL ARCHIVO⭐️: " + file_name.get_extension())

				if file_name.get_extension() == "png" or file_name.get_extension() == "jpg":
					prints("estencion gd")
					image_paths.append(path + "/" + file_name)
					prints( "⭐️DIRECTORIO⭐️  ",path + "/" + file_name)
			file_name = dir.get_next()
	else:
		print("⭐️ EDITOR FALLO : An error occurred when trying to access the path ⭐️")
