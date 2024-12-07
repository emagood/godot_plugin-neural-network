extends Node

func _ready():
	var generator_structure = [100, 128, 256, 512, 28*28]  # Estructura de ejemplo para el generador
	var discriminator_structure = [28*28, 512, 256, 128, 1]  # Estructura de ejemplo para el discriminador
	var use_bias = true
	
	var gan = GAN.new(generator_structure, discriminator_structure, use_bias)
	
	# Crear datos de ejemplo para entrenar
	var real_data = []
	for i in range(50):  # Generar 1000 imágenes de ejemplo (suponiendo imágenes de 28x28 píxeles)
		var image_data = []
		for j in range(28*28):
			image_data.append(randf())
		real_data.append(image_data)
	save_training_images(real_data, 28)
	# Entrenar la GAN
	gan.train_gan(real_data, 1, 32)  # Epochs = 10, batch_size = 32
	save_generated_images(gan, 5, 28)
	print("Training complete")









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
	for i in range(real_data.size()):
		var image_data = real_data[i]

		if image_data.size() != size * size:
			print("Training image data has incorrect size. Expected: ", size * size, " Got: ", image_data.size())
			continue

		var image = Image.create(28, 28, false, Image.FORMAT_RGBA8)
		#image.create(size, size, false, Image.FORMAT_RGB8)
		
		# Asegurarnos de que la imagen se crea correctamente
		if image.get_width() != size or image.get_height() != size:
			print("Failed to create image with correct dimensions. Width: ", image.get_width(), " Height: ", image.get_height())
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
		else:
			print("Training image saved: ", save_path)






func save_generated_images(gan, num_samples, size):
	for i in range(num_samples):
		var noise = gan.generate_noise(100)
		var generated_data = gan.generator_produce_data(noise)

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
